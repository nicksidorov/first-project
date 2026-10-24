"""Физические матрицы статьи из калибровки constants.py и метаданных IMS.

Rs и Rg — базовые отклики R_c^(0) и R_g^(0), до поправок C.
Строки: ROI [1, 2, 3, 5, 6, 7, 8]. Столбцы: Rn222, Xe135, Xe133,
Xe131m, Xe133m. Функция не читает спектры и не изменяет исходные константы.

У радона статья требует калибровочные H_11 для трёх измерительных условий.
Их передают явно. Режим ``legacy`` воспроизводит прежнее предположение
об эффективности 0.3 и распаде с периодом родительского Rn222; это явно
обозначенное приближение, а не новая калибровка радоновой цепочки.
"""

from collections.abc import Mapping
from datetime import datetime
import re
from typing import Any

import numpy as np

from constants import (
    DELTA_EFF, DELTA_V_XE, DETECTOR_RESPONSE_MATRICES,
    FIXED_EFFICIENCIES_SAMPLE, FIXED_EFFICIENCIES_GASBK, ROI_INFO, XENON_ISOTOPES,
)

from .activities import compose_activity_transform
from .detectors import validate_detector_ids
from .model import build_joint_matrix


LEGACY_ROI_NUMBERS = [1, 2, 3, 5, 6, 7, 8]
ISOTOPE_NAMES = ["Rn222", "Xe135", "Xe133", "Xe131m", "Xe133m"]


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: требуется положительное конечное число.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: требуется положительное конечное число.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}: требуется положительное конечное число.")
    return result


def _date(data: Mapping, key: str, label: str) -> datetime:
    result = data.get(key)
    if not isinstance(result, datetime):
        raise ValueError(f"{label}: отсутствует корректная дата {key}.")
    return result


def _elapsed(later: datetime, earlier: datetime, name: str) -> float:
    try:
        return (later - earlier).total_seconds()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: даты должны иметь согласованный часовой пояс.") from exc


def _times(data: Mapping, label: str) -> tuple[float, float]:
    real = _positive(data.get("real_time"), f"{label}: реальное время")
    live = _positive(data.get("live_time"), f"{label}: живое время")
    if live > real:
        raise ValueError(f"{label}: живое время не может превышать реальное.")
    return real, live


def decay_correction(decay_constant, duration):
    """G_lambda(T)=lambda*T/(1-exp(-lambda*T)), eq:decay_correction_function.

    expm1 сохраняет точность при коротких интервалах; G_lambda(0)=1.
    Поддерживается broadcasting NumPy для вектора констант распада.
    """
    decay, time = np.broadcast_arrays(
        np.asarray(decay_constant, dtype=float), np.asarray(duration, dtype=float),
    )
    if (
        np.any(~np.isfinite(decay)) or np.any(decay <= 0)
        or np.any(~np.isfinite(time)) or np.any(time < 0)
    ):
        raise ValueError("G_lambda: нужны конечные lambda>0 и T>=0.")
    with np.errstate(over="ignore", invalid="ignore"):
        argument = decay * time
    if np.any(~np.isfinite(argument)):
        raise ValueError("G_lambda: слишком большой интервал времени.")
    result = np.ones_like(argument)
    np.divide(argument, -np.expm1(-argument), out=result, where=argument != 0)
    return result


def _log_decay_correction(decay_constant: float, duration: float) -> float:
    """log(G) с рядом около нуля для точного вычисления G5/G3-1."""
    argument = decay_constant * duration
    if argument < 1e-3:
        # log(x/(1-exp(-x))) = x/2 - x²/24 + x⁴/2880 + O(x⁶).
        return argument / 2 - argument**2 / 24 + argument**4 / 2880
    return float(np.log(decay_correction(decay_constant, duration)))


def _isotope_identity(label: Any) -> str | None:
    """Выделить изотоп из IMS-метки, сохраняя признак изомера M.

    Например, XE133-10_81:30 содержит изотоп XE133 и числовой суффикс.
    Удалять пунктуацию из ВСЕЙ метки нельзя:
    цифры суффикса тогда становятся частью массового числа.
    Поддерживаются также XE-133M и 133M-XE. Суффикс допускается только
    после полного изотопа, с разделителем и числовым описанием.
    """
    match = re.fullmatch(
        r"(?P<nuclide>XE[-_\s]*[0-9]+M?|[0-9]+M?[-_\s]*XE)"
        r"(?:[-_:][0-9][0-9_:.,+\-]*)?",
        str(label).strip().upper(),
    )
    if match is None:
        return None
    nuclide = re.sub(r"[-_\s]", "", match.group("nuclide"))
    return nuclide if nuclide.startswith("XE") else "XE" + nuclide[:-2]


def _efficiencies(data: Mapping, label: str, *, fixed: bool, detector_id: int) -> np.ndarray:
    """Четыре ε ксенона: из выбранного источника, без множителя branching."""
    if fixed:
        constant_name = f"FIXED_EFFICIENCIES_{label}"
        registry = FIXED_EFFICIENCIES_SAMPLE if label == "SAMPLE" else FIXED_EFFICIENCIES_GASBK
        error = (
            f"constants.py: заполните {constant_name}[{detector_id}] четырьмя "
            "эффективностями в порядке Xe135, Xe133, Xe131m, Xe133m. "
            "Нужны конечные значения 0 < ε ≤ 1 (в долях, без branching)."
        )
        try:
            values = np.array(registry[detector_id], dtype=float, copy=True)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error) from exc
        if values.shape != (4,) or np.any(~np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1):
            raise ValueError(error)
        return values

    efficiencies = data.get("efficiency")
    if not isinstance(efficiencies, Mapping):
        raise ValueError(f"{label}: отсутствуют эффективности #b-gEfficiency.")
    values = np.zeros(4)
    for index, isotope in enumerate(ISOTOPE_NAMES[1:]):
        roi = ROI_INFO[isotope]["roi"]
        identities = data.get("roi_to_isotope", {})
        if isinstance(identities, Mapping) and roi in identities:
            identity = _isotope_identity(identities[roi])
            if identity != isotope.upper():
                raise ValueError(
                    f"{label}: эффективность ROI {roi} подписана как {identities[roi]}, "
                    f"ожидается {isotope}."
                )
        efficiency = _positive(
            efficiencies.get(roi), f"{label}: эффективность {isotope}, ROI {roi}",
        )
        if efficiency > 1:
            raise ValueError(f"{label}: эффективность {isotope} должна быть не больше 1.")
        values[index] = efficiency
    return values


def _sensitivities(efficiencies: np.ndarray) -> np.ndarray:
    """I_j=epsilon_j*BR_j; первые коэффициенты радона задаются через H."""
    values = np.zeros(5)
    for index, isotope in enumerate(ISOTOPE_NAMES[1:], start=1):
        branching = _positive(ROI_INFO[isotope]["br"], f"Ветвление {isotope}")
        if branching > 1:
            raise ValueError(f"Вероятность ветвления {isotope} должна быть не больше 1.")
        values[index] = efficiencies[index - 1] * branching
    return values


def _base_response(value, name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (7, 5) or np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"constants.{name}: нужна неотрицательная конечная матрица 7×5.")
    # Первые пять выбранных ROI — опорные области для пяти столбцов.
    if not np.allclose(np.diag(result[:5]), 1, rtol=1e-12, atol=1e-12):
        raise ValueError(f"constants.{name}: базовые отклики должны иметь опорную нормировку 1.")
    return result


def build_physical_matrices(
    sample: Mapping,
    gasbk: Mapping,
    *,
    detector_live_time: float | None = None,
    use_fixed_efficiencies: bool = False,
    radon_mode: str = "calibrated",
    radon_h: Mapping | None = None,
) -> dict[str, Any]:
    """Построить R_F и преобразование объёмных активностей T_V (ранее T_c).

    Если задано detector_live_time=t_bl, V86 включает параметры zeta
    детекторного фона: [beta;gamma;zeta], kappa_s=t_sl/t_bl,
    kappa_g=t_gl/t_bl. Без t_bl сохранён прежний двухблочный API.

    ``sample``/``gasbk`` — словари IMS-парсера. Нужны одинаковые номера
    detector_id (221 или 222) из #Header, даты начала, реальные и живые
    длительности обоих измерений, эффективности
    четырёх изотопов ксенона; для SAMPLE дополнительно даты пробоотбора
    и xenon_volume (cm³). Эффективности GASBK интерпретируются как
    эффективности памяти и сохраняются до SAMPLE. Это условие калибровки
    должно выполняться для используемых файлов.

    При use_fixed_efficiencies=True четыре ε каждого спектра берутся из
    FIXED_EFFICIENCIES_SAMPLE/GASBK[detector_id] в constants.py. Значения
    из файлов тогда не используются; модель радона выбирается отдельно.

    radon_h в режиме calibrated содержит положительные коэффициенты
    Бк/отсчёт: sample=H[0,0], gasbk=H_g2[0,0],
    sample_memory=H_g1[0,0]. В режиме legacy radon_h не задают.
    Объёмные активности имеют единицу Бк/м³; первая координата радона
    вспомогательная, не атмосферная объёмная активность радона.
    """
    if not isinstance(sample, Mapping) or not isinstance(gasbk, Mapping):
        raise ValueError("SAMPLE и GASBK должны быть словарями метаданных IMS.")
    # Матрицы R зависят от детектора; общего значения по умолчанию нет.
    detector_id = validate_detector_ids({"SAMPLE": sample, "GASBK": gasbk})
    if radon_mode not in ("calibrated", "legacy"):
        raise ValueError("radon_mode: ожидается calibrated или legacy.")
    if radon_mode == "legacy" and radon_h is not None:
        raise ValueError("В режиме legacy нельзя одновременно задавать radon_h.")

    t_sr, t_sl = _times(sample, "SAMPLE")
    t_gr, t_gl = _times(gasbk, "GASBK")
    t_bl = None if detector_live_time is None else _positive(detector_live_time, "DETBK: живое время")
    kappa_s = None if t_bl is None else t_sl / t_bl
    kappa_g = None if t_bl is None else t_gl / t_bl
    sample_start = _date(sample, "start_time", "SAMPLE")
    gasbk_start = _date(gasbk, "start_time", "GASBK")
    collection_start = _date(sample, "collection_start", "SAMPLE")
    collection_end = _date(sample, "collection_end", "SAMPLE")
    t_a = _elapsed(collection_end, collection_start, "Пробоотбор")
    t_p = _elapsed(sample_start, collection_end, "Ожидание SAMPLE")
    tau = _elapsed(sample_start, gasbk_start, "Интервал GASBK–SAMPLE")
    if t_a <= 0:
        raise ValueError("Окончание пробоотбора должно быть позже его начала.")
    if t_p < 0:
        raise ValueError("SAMPLE не может начинаться до окончания пробоотбора.")
    if tau < t_gr:
        raise ValueError("GASBK должен закончиться до начала SAMPLE.")

    V_xe = _positive(sample.get("xenon_volume"), "SAMPLE: объём извлечённого ксенона")
    # Статья: стабильный ксенон 0.087 cm³ на 1 m³ воздуха.
    V0 = V_xe / 0.087
    lambdas = np.array([
        _positive(XENON_ISOTOPES[isotope], f"lambda({isotope})")
        for isotope in ISOTOPE_NAMES
    ])
    lambda_3, lambda_5 = lambdas[2], lambdas[4]
    difference = lambda_5 - lambda_3
    if difference <= 0:
        raise ValueError("Для цепочки Xe133m→Xe133 требуется lambda_5>lambda_3.")
    chain_factor = lambda_3 / difference
    epsilon_c = _efficiencies(sample, "SAMPLE", fixed=use_fixed_efficiencies, detector_id=detector_id)
    epsilon_g = _efficiencies(gasbk, "GASBK", fixed=use_fixed_efficiencies, detector_id=detector_id)
    I_c = _sensitivities(epsilon_c)
    I_g = _sensitivities(epsilon_g)
    Rs, Rg = DETECTOR_RESPONSE_MATRICES[detector_id]
    R_c0 = _base_response(Rs, f"Rs_{detector_id}")
    R_g0 = _base_response(Rg, f"Rg_{detector_id}")

    def daughter_matrix(duration: float, sensitivity: np.ndarray) -> np.ndarray:
        # eq:C_decay_chain, 1-based [3,5] -> Python [2,4].
        log_ratio = (
            _log_decay_correction(lambda_5, duration)
            - _log_decay_correction(lambda_3, duration)
        )
        C = np.eye(5)
        C[2, 4] = sensitivity[2] / sensitivity[4] * chain_factor * np.expm1(log_ratio)
        return C

    # eq:C_matrices: дочь образуется за РЕАЛЬНОЕ время экспозиции.
    C_s_c = daughter_matrix(t_sr, I_c)
    C_s_g = daughter_matrix(t_sr, I_g)
    C_g_g = daughter_matrix(t_gr, I_g)
    # eq:response_factorization; после C пятый столбец НЕ нормируем повторно.
    R_c = R_c0 @ C_s_c
    R_g1 = R_g0 @ C_s_g
    R_g2 = R_g0 @ C_g_g

    H = np.eye(5)
    H_g1 = np.eye(5)
    H_g2 = np.eye(5)
    H[np.arange(1, 5), np.arange(1, 5)] = decay_correction(lambdas[1:], t_sr) / (t_sl * I_c[1:])
    H_g1[np.arange(1, 5), np.arange(1, 5)] = decay_correction(lambdas[1:], t_sr) / (t_sl * I_g[1:])
    H_g2[np.arange(1, 5), np.arange(1, 5)] = decay_correction(lambdas[1:], t_gr) / (t_gl * I_g[1:])
    warnings = []
    if radon_mode == "calibrated":
        if not isinstance(radon_h, Mapping):
            raise ValueError(
                "Для радона нужны калибровочные коэффициенты radon_h: "
                "sample, gasbk, sample_memory (Бк/отсчёт). "
                "В constants.py они не заданы; прежнее приближение выбирается явно как legacy."
            )
        H[0, 0] = _positive(radon_h.get("sample"), "radon_h.sample")
        H_g2[0, 0] = _positive(radon_h.get("gasbk"), "radon_h.gasbk")
        H_g1[0, 0] = _positive(radon_h.get("sample_memory"), "radon_h.sample_memory")
    else:
        # Прежняя модель радона из legacy/calculation.py. Она не заменяет
        # требуемую статьёй калибровку эффективного отклика всей цепочки.
        radon_I = 0.3 * _positive(ROI_INFO["Rn222"]["br"], "Ветвление Rn222")
        I_c[0] = I_g[0] = radon_I
        H[0, 0] = H_g1[0, 0] = decay_correction(lambdas[0], t_sr) / (t_sl * radon_I)
        H_g2[0, 0] = decay_correction(lambdas[0], t_gr) / (t_gl * radon_I)
        warnings.append(
            "Радон: явно выбран режим legacy — эффективность 0.3 и поправка "
            "по периоду Rn222 из старого проекта. Это приближение, не отдельная "
            "калибровка радоновой цепочки; оно влияет и на перенос памяти."
        )

    # eq:F_operator: эволюция АКТИВНОСТЕЙ B, затем возврат в шкалу отсчётов.
    B = np.diag(np.exp(-lambdas * tau))
    B[2, 4] = chain_factor * np.exp(-lambda_3 * tau) * (-np.expm1(-difference * tau))
    F = np.linalg.solve(H_g1, B @ H_g2)

    # eq:Gamma_decay_chain: вычитаем дочь, образовавшуюся ДО SAMPLE.
    # expm1 предотвращает потерю точности при коротких t_a и t_p.
    log_collection_ratio = (
        _log_decay_correction(lambda_5, t_a)
        - _log_decay_correction(lambda_3, t_a)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        Gamma = np.eye(5)
        Gamma[2, 4] = -chain_factor * np.expm1(difference * t_p + log_collection_ratio)
        G_p = np.diag(np.exp(lambdas * t_p))
        G_s = np.diag(decay_correction(lambdas, t_a))
    if not all(np.all(np.isfinite(matrix)) for matrix in (H, H_g1, H_g2, Gamma, G_p, G_s)):
        raise ValueError("Времена/чувствительности приводят к переполнению физических матриц.")
    with np.errstate(over="ignore", invalid="ignore"):
        T_c = compose_activity_transform(H, Gamma, G_p, G_s, V0)
        M = build_joint_matrix(R_c, R_g1, R_g2, F, kappa_s=kappa_s, kappa_g=kappa_g)
    if not np.all(np.isfinite(T_c)):
        raise ValueError("Переполнение T_c: проверьте времена, объём и эффективности.")

    # Используем ЧИСЛА старой конфигурации. 0.001 означает 0.1%, даже если
    # старый комментарий рядом с константой ошибочно говорит о 5% или 10%.
    delta_I = np.array(DELTA_EFF, dtype=float, copy=True)
    delta_V0 = float(DELTA_V_XE)
    if (delta_I.shape != (5,) or np.any(~np.isfinite(delta_I)) or np.any(delta_I < 0)
            or not np.isfinite(delta_V0) or delta_V0 < 0):
        raise ValueError("DELTA_EFF/DELTA_V_XE: нужны конечные неотрицательные относительные неопределённости.")

    metadata = {
        "calibration_detector_id": detector_id,
        "calibration_source": (
            f"constants.py: Rs_{detector_id}, Rg_{detector_id}, ROI_INFO, XENON_ISOTOPES"
        ),
        "radon_mode": radon_mode,
        "radon_h": {"sample": float(H[0, 0]), "gasbk": float(H_g2[0, 0]),
                    "sample_memory": float(H_g1[0, 0])},
        "time_parameters": {"t_sr": t_sr, "t_sl": t_sl, "t_gr": t_gr,
                            "t_gl": t_gl, "t_a": t_a, "t_p": t_p, "tau": tau},
        "xenon_volume_cm3": V_xe,
        "effective_air_volume_m3": V0,
        "efficiency_source": (
            f"constants.py: FIXED_EFFICIENCIES_SAMPLE/GASBK[{detector_id}]"
            if use_fixed_efficiencies else "SAMPLE/GASBK: #b-gEfficiency"
        ),
        "efficiency_isotope_names": ISOTOPE_NAMES[1:],
        "sample_efficiencies": epsilon_c.tolist(),
        "gasbk_efficiencies": epsilon_g.tolist(),
        "efficiency_convention": "I=epsilon×BR; epsilon без branching, BR из constants.ROI_INFO",
        "memory_calibration": "Эффективности GASBK относятся к памяти и сохраняются при SAMPLE",
        "base_response_convention": "Rs=R_c^(0), Rg=R_g1^(0)=R_g2^(0); без повторной нормировки после C",
        "radon_concentration": "Первая координата a_c вспомогательная; не атмосферная активность Rn222",
        "relative_uncertainty_source": "Числа DELTA_EFF/DELTA_V_XE, без использования старых комментариев: 0.001=0.1%",
        "delta_I": delta_I.tolist(), "delta_V0": delta_V0,
        "radon_sensitivity_audit": "В режиме calibrated I_c[0]=I_g[0]=0 служебные; радон задаётся коэффициентами H_11",
    }
    detector_extras = {}
    parameter_names = ([f"beta:{name}" for name in ISOTOPE_NAMES]
                       + [f"gamma:{name}" for name in ISOTOPE_NAMES])
    if t_bl is not None:
        metadata["time_parameters"]["t_bl"] = t_bl
        metadata["detector_background_model"] = "V86: совместные SAMPLE/GASBK/DETBK; zeta — ожидаемые отсчёты DETBK"
        metadata["observation_blocks"] = ["SAMPLE", "GASBK", "DETBK"]
        metadata["kappa_s"] = kappa_s
        metadata["kappa_g"] = kappa_g
        metadata["radon_concentration"] = "Первая координата a_V вспомогательная; не атмосферная активность Rn222"
        parameter_names.extend(f"zeta:ROI{roi}" for roi in LEGACY_ROI_NUMBERS)
        detector_extras = {
            "kappa_s": kappa_s, "kappa_g": kappa_g,
            "detector_roi_numbers": np.array(LEGACY_ROI_NUMBERS, dtype=int),
        }
    return {
        "M": M, "R_c": R_c, "R_g1": R_g1, "R_g2": R_g2, "F": F,
        "R_c0": R_c0, "R_g0": R_g0,
        "C_s_c": C_s_c, "C_s_g": C_s_g, "C_g_g": C_g_g,
        "B": B, "H": H, "H_g1": H_g1, "H_g2": H_g2,
        "Gamma": Gamma, "G_p": G_p, "G_s": G_s, "G_a": G_s,
        "V0": V0, "T_c": T_c, "T_V": T_c,
        "I_c": I_c, "I_g": I_g,
        "delta_I": delta_I, "delta_V0": delta_V0,
        "coordinates": "beta_gamma" if t_bl is None else "beta_gamma_zeta",
        "n_sample_parameters": 5, "n_detector_parameters": 0 if t_bl is None else 7,
        "isotope_names": np.array(ISOTOPE_NAMES),
        "parameter_names": np.array(parameter_names),
        "sample_roi_numbers": np.array(LEGACY_ROI_NUMBERS, dtype=int),
        "gasbk_roi_numbers": np.array(LEGACY_ROI_NUMBERS, dtype=int),
        "mdc_component_indices": np.array([1, 2, 3, 4], dtype=int),
        "warnings": warnings, "metadata": metadata, **detector_extras,
    }
