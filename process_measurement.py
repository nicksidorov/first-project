"""IMS → совместная модель SAMPLE/GASBK → активности и MDC.

Матрицы строятся из constants.py и метаданных трёх спектров либо читаются
из подготовленного NPZ. Старый calculation.py не вызывается.
"""

from pathlib import Path
from typing import Any

import numpy as np

from constants import FIXED_ROI_TABLE, ROI_INFO
from ims_parser import (
    convert_roi_energy_to_channels, fit_energy_calibration,
    get_roi_counts_vector, parse_ims_spectral_file,
)
from xemat.activities import compute_activities
from xemat.calibration import LEGACY_ROI_NUMBERS, build_physical_matrices
from xemat.detection import compute_detection_limits
from xemat.diagnostics import compute_fit_quality
from xemat.detectors import validate_detector_ids
from xemat.io import load_prepared_model
from xemat.model import JointModel
from xemat.solvers import fit


METHOD_NAMES = {
    "wls": "Взвешенный МНК",
    "nnls": "Взвешенный NNLS: физические параметры ≥ 0",
    "poisson": "Пуассоновское МП: свободные параметры",
    "poisson_nonnegative": "Пуассоновское МП: физические параметры ≥ 0",
}


def _roi_order(archive, name: str) -> list[int]:
    """Явный порядок строк связывает подготовленные M и измеренные ROI."""
    key = name if name in archive else "roi_numbers"
    if key not in archive:
        raise ValueError(
            "Для расчёта по спектрам задайте в NPZ sample_roi_numbers и "
            "gasbk_roi_numbers (или общий roi_numbers). Порядок должен совпадать с M."
        )
    numbers = np.asarray(archive[key])
    if (
        numbers.ndim != 1 or numbers.size == 0
        or not np.issubdtype(numbers.dtype, np.number)
        or not np.all(np.isfinite(numbers))
        or np.any(numbers < 1) or np.any(numbers != np.floor(numbers))
    ):
        raise ValueError(f"{key}: нужны положительные целочисленные номера ROI.")
    order = [int(number) for number in numbers]
    if len(set(order)) != len(order):
        raise ValueError(f"{key}: ROI не должны повторяться.")
    return order


def _ordered_counts(counts: dict, order: list[int], label: str) -> np.ndarray:
    missing = [number for number in order if number not in counts]
    if missing:
        raise ValueError(f"{label}: отсутствуют ROI {missing}.")
    return np.asarray([counts[number] for number in order], dtype=float)


def _temporal(data: dict) -> dict:
    return {key: data.get(key) for key in (
        "start_time", "collection_start", "collection_end", "real_time",
        "live_time", "air_volume", "xenon_volume",
    )}


def _check_disjoint_rois(data, order, use_fixed_roi, label):
    """Статья использует непересекающиеся ROI: один отсчёт — одна строка.

    Повторяем только геометрию неизменённого парсера: полином калибровки,
    round, ограничение 0..255 и включение обоих концов диапазона. Даже
    один общий канал нарушает предположение независимости ROI.
    """
    regions = (
        FIXED_ROI_TABLE if use_fixed_roi else
        dict(zip(data["roi_numbers"], data["roi_limits"]))
    )
    missing = [number for number in order if number not in regions]
    if missing:
        raise ValueError(f"{label}: не найдены границы ROI {missing}.")
    gamma_to_ch, beta_to_ch = fit_energy_calibration(data["g_energy"], data["b_energy"])
    channels = convert_roi_energy_to_channels(
        [regions[number] for number in order], gamma_to_ch, beta_to_ch,
    )
    channels = np.clip(np.round(channels), 0, 255).astype(int)
    rectangles = [
        (min(row[0], row[1]), max(row[0], row[1]), min(row[2], row[3]), max(row[2], row[3]))
        for row in channels
    ]
    for first in range(len(order)):
        b0, b1, g0, g1 = rectangles[first]
        for second in range(first + 1, len(order)):
            c0, c1, h0, h1 = rectangles[second]
            if max(b0, c0) <= min(b1, c1) and max(g0, h0) <= min(g1, h1):
                raise ValueError(
                    f"{label}: ROI {order[first]} и {order[second]} пересекаются в каналах. "
                    "Для модели статьи выберите непересекающиеся ROI."
                )


def _read_spectra(paths):
    parsed = {
        name: parse_ims_spectral_file(str(path), is_sample=name == "sample")
        for name, path in zip(("sample", "background", "detector"), paths)
    }
    # Проверка общая для constants.py и NPZ с тремя файлами. Нельзя
    # применять фон одного детектора к пробе другого даже с готовой M.
    validate_detector_ids({
        "SAMPLE": parsed["sample"], "GASBK": parsed["background"],
        "DETBK": parsed["detector"],
    })
    return parsed


def _spectral_arrays(parsed, sample_order, gasbk_order, use_fixed_roi):
    """Исходные отсчёты и фиксированный DETBK в согласованном порядке ROI."""
    counts = {
        name: get_roi_counts_vector(data, use_fixed_roi=use_fixed_roi)
        for name, data in parsed.items()
    }

    for name, order in (("sample", sample_order), ("background", gasbk_order),
                        ("detector", sample_order), ("detector", gasbk_order)):
        _check_disjoint_rois(parsed[name], order, use_fixed_roi, name.upper())

    if not use_fixed_roi:
        # DETBK должен описывать те же энергетические области, а не только
        # совпадающие номера ROI. Сетки каналов при этом могут различаться.
        regions = {
            name: dict(zip(data["roi_numbers"], data["roi_limits"]))
            for name, data in parsed.items()
        }
        for name, order in (("sample", sample_order), ("background", gasbk_order)):
            for number in order:
                if number not in regions["detector"] or number not in regions[name]:
                    raise ValueError(f"Не найдено описание ROI {number} в {name}/DETBK.")
                if not np.allclose(regions[name][number], regions["detector"][number]):
                    raise ValueError(
                        f"ROI {number} в {name} и DETBK имеет разные границы энергии. "
                        "Подготовьте согласованные ROI или используйте готовые x,b в NPZ."
                    )

    sample_start = parsed["sample"].get("start_time")
    gasbk_start = parsed["background"].get("start_time")
    if sample_start and gasbk_start and gasbk_start > sample_start:
        raise ValueError("GASBK должен предшествовать SAMPLE.")

    live_times = []
    for name in ("sample", "background", "detector"):
        live_time = parsed[name].get("live_time")
        if live_time is None or not np.isfinite(live_time) or live_time <= 0:
            raise ValueError(f"{name}: для нормировки DETBK нужно положительное живое время.")
        live_times.append(float(live_time))
    t_s, t_g, t_d = live_times

    # x=[x_s;x_g] — исходные отсчёты. DETBK не вычитается из наблюдений.
    x = np.concatenate([
        _ordered_counts(counts["sample"], sample_order, "SAMPLE"),
        _ordered_counts(counts["background"], gasbk_order, "GASBK"),
    ])
    # b — фиксированный фон в mu=M@theta+b, приведённый по живому времени.
    # Неопределённость оценки DETBK здесь автоматически не добавляется:
    # при необходимости полную covariance пользователь передаёт через NPZ.
    b = np.concatenate([
        (t_s / t_d) * _ordered_counts(counts["detector"], sample_order, "DETBK"),
        (t_g / t_d) * _ordered_counts(counts["detector"], gasbk_order, "DETBK"),
    ])
    metadata = {f"{name}_temporal": _temporal(data) for name, data in parsed.items()}
    metadata.update({f"{name}_roi_counts": value for name, value in counts.items()})
    metadata["detector_info"] = {
        name: {"id": data.get("detector_id"), "name": data.get("detector_name")}
        for name, data in parsed.items()
    }
    metadata["input_detector_id"] = int(parsed["sample"]["detector_id"])
    metadata["sample_roi_numbers"] = sample_order
    metadata["gasbk_roi_numbers"] = gasbk_order
    metadata["background_source"] = "DETBK: b=[t_s/t_d·d_s; t_g/t_d·d_g], фиксированный фон"
    return x, b, metadata


def _spectral_input(model_file, paths, use_fixed_roi):
    parsed = _read_spectra(paths)
    with np.load(model_file, allow_pickle=False) as archive:
        sample_order = _roi_order(archive, "sample_roi_numbers")
        gasbk_order = _roi_order(archive, "gasbk_roi_numbers")
    return _spectral_arrays(parsed, sample_order, gasbk_order, use_fixed_roi)


def _constants_input(paths, use_fixed_roi, radon_mode, radon_h):
    """Привязать Rs/Rg к геометрии ROI и временам конкретного измерения."""
    parsed = _read_spectra(paths)
    reference_rois = [ROI_INFO[name]["roi"] for name in ("Xe135", "Xe133", "Xe131m", "Xe133m")]
    for name, data in parsed.items():
        regions = dict(zip(data["roi_numbers"], data["roi_limits"]))
        # Даже при фиксированных ROI нельзя использовать эффективность
        # из файла для ДРУГОЙ опорной области. У DETBK эффективности не берём.
        checked = (reference_rois if name != "detector" else []) if use_fixed_roi else LEGACY_ROI_NUMBERS
        for number in checked:
            if number not in regions or not np.allclose(
                regions[number], FIXED_ROI_TABLE[number], rtol=0, atol=1e-6,
            ):
                if name != "detector" and number in reference_rois:
                    reason = (
                        "Эффективность из файла относится к этой опорной области; "
                        "нужна калибровка для совпадающих границ или подготовленная модель NPZ."
                    )
                else:
                    reason = "Выберите фиксированные ROI или подготовленную модель с подходящей калибровкой."
                raise ValueError(
                    f"{name.upper()}: границы ROI {number} не соответствуют "
                    f"калибровке constants.py. {reason}"
                )
    order = list(LEGACY_ROI_NUMBERS)
    x, b, metadata = _spectral_arrays(parsed, order, order, use_fixed_roi)
    extras = build_physical_matrices(
        parsed["sample"], parsed["background"],
        radon_mode=radon_mode, radon_h=radon_h,
    )
    model = JointModel(extras["M"], x, b, 5, extras["T_c"], "beta_gamma")
    metadata.update(extras.get("metadata", {}))
    metadata["calibration_source"] += "; времена, объём и эффективности из IMS"
    return model, extras, metadata


def process_prepared_measurement(
    model_file: str | None = None,
    *,
    sample_file: str | None = None,
    background_file: str | None = None,
    detector_file: str | None = None,
    use_fixed_roi: bool = False,
    method: str = "nnls",
    radon_mode: str = "legacy",
    radon_h: dict | None = None,
) -> dict[str, Any]:
    """Обработать три спектра с constants.py либо готовую модель NPZ.

    При трёх спектрах x,b из NPZ заменяются исходными ROI и нормированным
    DETBK. Без NPZ все преобразования строятся из калибровки и метаданных.
    Пользовательский профиль по умолчанию сохраняет прежнюю модель радона
    (эффективность 0.3); допущение записывается в протокол. Для отдельной
    калибровки задаются radon_mode='calibrated' и три коэффициента radon_h.
    """
    paths = (sample_file, background_file, detector_file)
    if any(paths) and not all(paths):
        raise ValueError("Выберите все три спектра или используйте только готовую модель NPZ.")
    if method not in METHOD_NAMES:
        raise ValueError(f"Неизвестный метод: {method}.")
    if model_file is None:
        if not all(paths):
            raise ValueError("Для матриц constants.py выберите все три спектра: SAMPLE, GASBK и DETBK.")
        model, extras, metadata = _constants_input(paths, use_fixed_roi, radon_mode, radon_h)
    else:
        x = b = None
        metadata = {"background_source": "Готовый фиксированный вектор b из NPZ"}
        if all(paths):
            x, b, metadata = _spectral_input(model_file, paths, use_fixed_roi)
        metadata["calibration_source"] = "Подготовленная модель NPZ"
        model, extras = load_prepared_model(model_file, x=x, b=b)
    is_quadratic = method in ("wls", "nnls")
    fitting_options = (
        {"weights": extras.get("weights"), "observation_covariance": extras.get("observation_covariance")}
        if is_quadratic else {}
    )
    solution = fit(model, method=method, **fitting_options)
    if not solution.success:
        raise RuntimeError(f"Решатель не сошёлся: {solution.message}")

    result = {
        "model_file": str(Path(model_file).resolve()) if model_file is not None else None,
        "calibration_mode": "constants" if model_file is None else "prepared",
        "sample_file": sample_file,
        "background_file": background_file,
        "detector_file": detector_file,
        "use_fixed_roi": use_fixed_roi,
        "regression_method_used": method,
        "model": model,
        "fit": solution,
        "fit_quality": compute_fit_quality(model, solution),
        "activities": None,
        "detection": None,
        "warnings": list(extras.get("warnings", [])) + list(solution.warnings),
        "parameter_names": extras.get("parameter_names"),
        "isotope_names": extras.get("isotope_names"),
        "metadata": metadata,
        "calibration_matrices": {
            key: extras[key] for key in (
                "R_c0", "R_g0", "R_c", "R_g1", "R_g2", "C_s_c", "C_s_g", "C_g_g", "B", "F",
                "H_g1", "H_g2", "H", "Gamma", "G_p", "G_s", "I_c", "I_g", "V0",
                "delta_I", "delta_V0",
            ) if key in extras
        },
        "weights_source": (
            "Внешняя ковариационная матрица наблюдений"
            if extras.get("observation_covariance") is not None else
            "Внешние веса" if extras.get("weights") is not None else
            "Статья: w_i=1/x_i при x_i>0; w_i=1 при x_i=0"
        ) if method in ("wls", "nnls") else "Исходные пуассоновские отсчёты; веса не применяются",
    }
    if not is_quadratic and any(key in extras for key in ("weights", "observation_covariance")):
        result["warnings"].append("Веса/ковариация из NPZ относятся к МНК и не применяются в пуассоновском критерии.")

    # Активности всегда вычисляются из beta. Неотрицательные методы
    # ограничивают a_c, поэтому нельзя выдавать beta за a_c.
    n = model.n_sample_parameters
    if extras.get("H") is not None or model.T_c is not None:
        if model.coordinates == "beta_gamma":
            beta = solution.theta[:n]
            K_beta = None if solution.covariance is None else solution.covariance[:n, :n]
        elif model.T_c is not None:
            inverse_T = np.linalg.solve(model.T_c, np.eye(n))
            beta = inverse_T @ solution.theta[:n]
            K_beta = (
                None if solution.covariance is None else
                inverse_T @ solution.covariance[:n, :n] @ inverse_T.T
            )
        else:
            beta = None
            K_beta = None
            result["warnings"].append("Активность в детекторе не рассчитана: для a_c→beta нужна T_c.")
        if beta is not None:
            activity_options = {key: extras[key] for key in (
                "H", "Gamma", "G_p", "G_s", "V0", "delta_I", "delta_V0",
            ) if key in extras}
            if model.T_c is not None:
                activity_options["T_c"] = model.T_c
            try:
                result["activities"] = compute_activities(beta, K_beta, **activity_options)
            except (ValueError, np.linalg.LinAlgError) as exc:
                result["warnings"].append(f"Активности недоступны: {exc}")
    if extras.get("H") is None:
        result["warnings"].append("Активность в детекторе недоступна: в NPZ не задана H.")

    # MDC — отдельное заключение о обнаружении, а не неопределённость
    # ограниченной оценки. Использует свободную и неотрицательную оценки.
    if solution.physical_parameters is None:
        result["warnings"].append("MDC недоступна: отсутствует преобразование beta→a_c (T_c).")
    elif "mdc_component_indices" not in extras:
        result["warnings"].append(
            "MDC не рассчитана: укажите mdc_component_indices в NPZ; включайте "
            "только компоненты радиоактивного ксенона в блоке a_c."
        )
    else:
        free_method = "poisson" if method.startswith("poisson") else "wls"
        positive_method = "poisson_nonnegative" if method.startswith("poisson") else "nnls"
        if method in ("wls", "nnls") and any(
            extras.get(key) is not None for key in ("weights", "observation_covariance")
        ):
            result["warnings"].append(
                "MDC использует пуассоновскую ковариацию модели статьи при фиксированном фоне; "
                "внешние веса/ковариация МНК не заменяют модель K_i(L)."
            )
        try:
            free = solution if method == free_method else fit(model, method=free_method, **fitting_options)
            positive = solution if method == positive_method else fit(model, method=positive_method, **fitting_options)
            if not free.success or not positive.success:
                raise ValueError("Не сошлась одна из вспомогательных оценок для MDC.")
            indices = np.asarray(extras["mdc_component_indices"])
            if (
                indices.ndim != 1 or not np.issubdtype(indices.dtype, np.number)
                or not np.all(np.isfinite(indices)) or np.any(indices != np.floor(indices))
                or np.any(indices < 0) or np.any(indices >= n)
            ):
                raise ValueError("mdc_component_indices должны индексировать только блок a_c.")
            isotope_names = extras.get("isotope_names")
            if isotope_names is not None and np.asarray(isotope_names).shape == (n,):
                for index in indices.astype(int):
                    if str(isotope_names[index]).lower().replace("-", "").startswith("rn"):
                        raise ValueError("Rn — вспомогательная компонента; MDC вычисляется только для ксенона.")
            result["detection"] = compute_detection_limits(
                model.physical_matrix, model.b, free.physical_parameters,
                positive.physical_parameters, component_indices=indices.astype(int),
                alpha=float(extras.get("alpha", 0.05)),
                beta_ii=float(extras.get("beta_ii", 0.05)),
                mode=str(extras.get("mdc_mode", "iterative")),
            )
            result["warnings"].extend(result["detection"].warnings)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            result["warnings"].append(f"MDC недоступна: {exc}")
    return result


def parse_sample_and_background_and_detector(
    sample_file, background_file, detector_file, use_fixed_roi=False,
    regression_method="nnls", *, model_file=None, radon_mode="legacy", radon_h=None,
):
    """Совместимый вход старого GUI; без NPZ используются Rs/Rg constants.py."""
    aliases = {"ls": "wls", "mle_free": "poisson", "mle": "poisson_nonnegative"}
    return process_prepared_measurement(
        model_file, sample_file=sample_file, background_file=background_file,
        detector_file=detector_file, use_fixed_roi=use_fixed_roi,
        method=aliases.get(regression_method, regression_method),
        radon_mode=radon_mode, radon_h=radon_h,
    )
