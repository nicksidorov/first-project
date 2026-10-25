"""Экспериментальная проверка L_C и L_D=MDC параметрическим Монте-Карло.

Физические координаты V86 z=[a_V;gamma;zeta], S_V=M Q^{-1}. Для проверяемого
изотопа i задаём z0=z_+ с z0[i]=0; память gamma и фон zeta остаются.
Генерация всех трёх спектров: x* ~ Poisson(S_V@z0+A*S_V[:,i]).
Прежние модели [a_c;gamma] с известным аддитивным b также поддерживаются.
Каждый повтор заново оценивает свободные и неотрицательные параметры,
включая zeta, строит свой нулевой спектр и L_C. Решение: a_V_free[i] > L_C.
Проверяются A=0 и A=исходный MDC. Это проверка в одном заданном режиме
памяти/фона, а не доказательство уровня для всех возможных измерений.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import binomtest

from .detection import compute_detection_limits
from .model import JointModel
from .solvers import fit


def _family(method: str) -> tuple[str, str]:
    if method in ("wls", "nnls"):
        return "wls", "nnls"
    if method in ("poisson", "poisson_nonnegative"):
        return "poisson", "poisson_nonnegative"
    raise ValueError("Метод Монте-Карло: wls/nnls либо poisson/poisson_nonnegative.")


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}: требуется целое число не меньше {minimum}.")
    return int(value)


def _checked_fit(model, method, options):
    result = fit(model, method=method, **options)
    if not result.success:
        raise RuntimeError(f"{method}: решатель не сошёлся: {result.message}")
    if result.physical_parameters is None or not np.all(np.isfinite(result.physical_parameters)):
        raise ValueError(f"{method}: нет конечных физических параметров {model.physical_coordinate_label}.")
    return result


def _summarize(label, activity, requested, estimates, levels, failures, fixed_lc):
    values = np.asarray(estimates, dtype=float)
    thresholds = np.asarray(levels, dtype=float)
    valid = len(values)
    failed = sum(failures.values())
    attempted = valid + failed
    detected = int(np.count_nonzero(values > thresholds))
    interval = None
    if valid:
        ci = binomtest(detected, valid).proportion_ci(confidence_level=.95, method="exact")
        interval = (float(ci.low), float(ci.high))
    return {
        "label": label, "activity": float(activity), "n_requested": requested,
        "n_attempted": attempted, "n_valid": valid, "n_failed": failed,
        "n_detected": detected,
        "detection_rate": detected / valid if valid else None,
        "confidence_interval": interval,
        "failure_probability_bounds": (
            (detected / attempted, (detected + failed) / attempted) if attempted else None
        ),
        "mean_L_C": float(np.mean(thresholds)) if valid else None,
        "mean_free_activity": float(np.mean(values)) if valid else None,
        "std_free_activity": float(np.std(values, ddof=1)) if valid > 1 else None,
        "fixed_threshold_detection_rate": float(np.mean(values > fixed_lc)) if valid else None,
        "failure_reasons": dict(failures),
        # Сохраняем оценки для гистограмм и независимой проверки решения;
        # исходные большие гистограммы спектров в каждом повторе не копируем.
        "free_activity_samples": values,
        "L_C_samples": thresholds,
    }


def run_monte_carlo(
    model: JointModel,
    physical_positive,
    *,
    component_index: int,
    method: str = "wls",
    n_trials: int = 1000,
    seed: int = 12345,
    alpha: float = .05,
    beta_ii: float = .05,
    mdc_mode: str = "iterative",
    weights=None,
    observation_covariance=None,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Два независимых набора n_trials повторов при фиксированной истинной смеси.

    physical_positive — выбранная неотрицательная физическая смесь; из неё
    удаляется только активность пробы проверяемого изотопа. Для сравнения
    методов передавайте одну смесь и seed: исходные случайные отсчёты будут
    одинаковыми. При стандартном МНК веса 1/max(x*,1) обновляются в каждом
    повторе. Переданные внешние веса/ковариация остаются фиксированными.

    В V86 истинные средние zeta фиксированы, но наблюдения DETBK каждый
    раз разыгрываются и zeta оценивается заново вместе с пробой и памятью.
    Так учитывается конечная статистика измерения DETBK. В прежней модели
    фиксированный b остаётся известным, отдельного розыгрыша DETBK нет.

    Неудачные расчёты учитываются отдельно, без повторного розыгрыша.
    detection_rate и биномиальный интервал условны на успешном расчёте,
    если есть ошибки. failure_probability_bounds показывает диапазон доли
    среди всех выполненных попыток при двух крайних исходах ошибок.
    """
    component_index = _integer(component_index, "Индекс изотопа", 0)
    if component_index >= model.n_sample_parameters:
        raise ValueError("Проверяемый индекс должен относиться к объёмной активности пробы, а не к памяти или фону.")
    n_trials = _integer(n_trials, "Число повторов", 1)
    seed = _integer(seed, "Seed", 0)
    free_method, positive_method = _family(method)
    S = model.physical_matrix
    positive = np.asarray(physical_positive, dtype=float)
    if positive.shape != (S.shape[1],) or np.any(~np.isfinite(positive)) or np.any(positive < 0):
        raise ValueError(f"Нужен конечный неотрицательный вектор {model.physical_coordinate_label} полной модели.")
    positive = positive.copy()
    # Для исходных LC/LD свободная оценка не нужна: она используется
    # compute_detection_limits только в поле detected, здесь не читаемом.
    reference = compute_detection_limits(
        S, model.b, positive, positive, component_indices=[component_index],
        alpha=alpha, beta_ii=beta_ii, mode=mdc_mode,
    )
    if not reference.converged[0] or not np.isfinite(reference.MDC[0]):
        raise ValueError("Исходный MDC не сошёлся; сценарий A=MDC построить нельзя.")
    lc, ld = float(reference.L_C[0]), float(reference.MDC[0])
    null_physical = positive.copy()
    null_physical[component_index] = 0
    null_mu = reference.null_spectra[0].copy()
    means = (null_mu, null_mu + ld * S[:, component_index])
    if any(np.any(~np.isfinite(mu)) or np.any(mu < 0) for mu in means):
        raise ValueError("Ожидаемые отсчёты эксперимента должны быть конечными и неотрицательными.")
    options = {"weights": weights, "observation_covariance": observation_covariance} if free_method == "wls" else {}
    # Проверяем конфигурацию до длинного цикла, чтобы ошибки весов/модели
    # не превращались в тысячи одинаковых неудачных испытаний.
    _checked_fit(model, free_method, options)
    _checked_fit(model, positive_method, options)
    rng = np.random.default_rng(seed)
    scenarios = []
    done, total = 0, 2 * n_trials
    cancelled = False
    if progress is not None:
        progress(0, total)
    for label, activity, mu in zip(("A=0", "A=MDC"), (0., ld), means):
        estimates, levels, failures = [], [], Counter()
        for _ in range(n_trials):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            # Независимые исходные отсчёты всех строк M, включая нули.
            # Для V86 здесь 3k строк: SAMPLE, GASBK и DETBK. Средние заданы
            # исходной истинной смесью, а не новым случайным отсчётом DETBK.
            x = rng.poisson(mu)
            try:
                trial = JointModel(
                    model.M, x, model.b, model.n_sample_parameters, model.T_c,
                    model.coordinates, n_detector_parameters=model.n_detector_parameters,
                )
                free = _checked_fit(trial, free_method, options)
                constrained = _checked_fit(trial, positive_method, options)
                # LC не зависит от mode. fixed здесь экономит ненужную
                # итерацию LD: используются ТОЛЬКО LC и свободная оценка.
                # Исходный LD выше рассчитан в выбранном mdc_mode статьи.
                decision = compute_detection_limits(
                    S, model.b, free.physical_parameters, constrained.physical_parameters,
                    component_indices=[component_index], alpha=alpha, beta_ii=beta_ii, mode="fixed",
                )
                estimates.append(float(decision.free_activity[0]))
                levels.append(float(decision.L_C[0]))
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                failures[f"{type(exc).__name__}: {exc}"] += 1
            done += 1
            if progress is not None:
                progress(done, total)
        scenarios.append(_summarize(label, activity, n_trials, estimates, levels, failures, lc))
        if cancelled:
            break
    sampling_note = (
        "Разыгрываются SAMPLE, GASBK и DETBK; в каждом повторе заново оцениваются проба, память и фон zeta. "
        "Истинная смесь, матрицы, времена, объём и калибровка фиксированы; систематические неопределённости не моделируются."
        if model.has_detector_background else
        "Разыгрываются SAMPLE и GASBK. b (DETBK), матрицы, времена, объём и калибровка фиксированы; их систематические неопределённости не моделируются."
    )
    activity_symbol = "a_V" if model.has_detector_background else "a_c"
    warnings = [
        "Параметрическая проверка для одной фиксированной смеси, матриц и фона; она не доказывает уровень для всех измерений.",
        sampling_note,
        "Уровни относятся к одному выбранному изотопу, без поправки на одновременную проверку нескольких изотопов.",
        "Проверяется вероятность обнаружения при исходном MDC; эмпирический L_D здесь не подбирается.",
    ]
    if any(row["n_failed"] for row in scenarios):
        warnings.append("Есть неудачные расчёты: доли и 95% интервалы рассчитаны только по успешным. Учитывайте диапазон исходов ошибок среди всех попыток.")
    if cancelled:
        warnings.append("Эксперимент остановлен: показаны частичные результаты. Остановка по наблюдаемым результатам нарушает обычную интерпретацию доверительного интервала.")
    if weights is not None or observation_covariance is not None:
        warnings.append(
            "Внешние веса/ковариация МНК фиксированы; генерация остаётся независимой пуассоновской и не моделирует внешнюю корреляцию наблюдений."
            if free_method == "wls" else "Внешние веса/ковариация не используются пуассоновским оценивателем."
        )
    return {
        "experiment_version": 2 if model.has_detector_background else 1,
        "component_name": f"{activity_symbol}[{component_index}]", "component_index": component_index,
        "physical_coordinates": model.physical_coordinate_label,
        "detector_background_estimated": model.has_detector_background,
        "method": free_method, "requested_trials_per_scenario": n_trials, "seed": seed,
        "alpha": float(alpha), "beta_ii": float(beta_ii), "mdc_mode": mdc_mode,
        "reference_L_C": lc, "reference_MDC": ld,
        "reference_positive_physical": positive, "reference_null_physical": null_physical,
        "reference_null_mu": null_mu, "reference_signal_mu": means[1],
        "model": model, "S_c": S,
        "quadratic_options": {key: None if value is None else np.array(value, copy=True)
                              for key, value in (("weights", weights), ("observation_covariance", observation_covariance))},
        "rng": type(rng.bit_generator).__name__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "scenarios": scenarios, "cancelled": cancelled, "warnings": warnings,
    }


def run_from_results(results: Mapping, *, component_index: int, n_trials: int = 1000,
                     seed: int = 12345, method: str | None = None, progress=None,
                     should_cancel=None) -> dict[str, Any]:
    """Использовать снимок последнего расчёта окна; файлы заново не читаются."""
    model, solution = results["model"], results["fit"]
    detection = results.get("detection")
    if detection is None:
        raise ValueError("Сначала выполните обычный расчёт с доступными LC и MDC.")
    component_index = _integer(component_index, "Индекс изотопа", 0)
    selected = np.flatnonzero(detection.component_indices == component_index)
    if len(selected) != 1 or not detection.converged[selected[0]]:
        raise ValueError("Выберите изотоп с рассчитанным MDC из последнего расчёта.")
    names = results.get("isotope_names")
    activity_symbol = "a_V" if model.has_detector_background else "a_c"
    name = str(names[component_index]) if names is not None else f"{activity_symbol}[{component_index}]"
    if name.lower().replace("-", "").replace(" ", "").startswith("rn"):
        raise ValueError("Проверка LC/MDC предназначена для ксенона; радон остаётся мешающей компонентой.")
    quadratic = results.get("quadratic_options")
    if quadratic is None:
        if str(results.get("weights_source", "")).startswith("Внешн"):
            raise ValueError("В сохранённом расчёте отсутствуют сами внешние веса. Повторите основной расчёт новой версией программы.")
        quadratic = {"weights": None, "observation_covariance": None}
    original_family, positive_method = _family(solution.method)
    # Истина задаётся исходным расчётом, а НЕ методом, выбранным в окне MC.
    # Поэтому смена исследуемого метода при том же seed сохраняет смесь
    # и случайные отсчёты: можно сравнивать именно оцениватели.
    positive = solution if solution.method == positive_method else _checked_fit(
        model, positive_method, quadratic if original_family == "wls" else {},
    )
    if not positive.success:
        raise ValueError("Неотрицательная оценка исходной смеси не сошлась.")
    result = run_monte_carlo(
        model, positive.physical_parameters, component_index=component_index,
        method=method or solution.method, n_trials=n_trials, seed=seed,
        alpha=detection.alpha, beta_ii=detection.beta_ii, mdc_mode=detection.mode,
        progress=progress, should_cancel=should_cancel, **quadratic,
    )
    result["component_name"] = name
    result["reference_fit_method"] = positive_method
    result["source"] = {
        key: results.get(key) for key in ("sample_file", "background_file", "detector_file", "model_file",
                                         "use_fixed_roi", "use_fixed_efficiencies", "metadata")
    }
    return result


def format_monte_carlo_report(result: Mapping) -> str:
    """Самостоятельный протокол эксперимента; активности выводятся в мБк/м³."""
    def number(value, scale=1):
        return "—" if value is None else f"{scale * value:.7g}"

    def percent(value):
        return number(value, 100) + "%" if value is not None else "—"

    joint_detector = result.get("detector_background_estimated", False)
    coordinates = result.get("physical_coordinates", "[a_c; gamma]")
    activity_symbol = "a_V" if joint_detector else "a_c"
    generation = (
        "Генерация: [c*; d*; b_DETBK*] ~ Poisson(S_V @ z0 + A * S_V[:,i]); z0[i]=0."
        if joint_detector else "Генерация: x* ~ Poisson(b + S_c @ z0 + A * S_c[:,i]); z0[i]=0."
    )
    fitting = (
        "В каждом повторе заново подгоняются проба, память и детекторный фон zeta и рассчитывается LC."
        if joint_detector else "В каждом повторе заново подгоняются проба и память и рассчитывается LC."
    )
    lines = [
        "XeMat — ЭКСПЕРИМЕНТ МОНТЕ-КАРЛО: LC И LD=MDC", "=" * 90,
        f"Изотоп: {result['component_name']}; метод: {result['method']}",
        f"Координаты: {coordinates}; единицы активностей ниже — мБк/м³.",
        f"Исходный L_C: {number(result['reference_L_C'], 1000)}; исходный MDC: {number(result['reference_MDC'], 1000)}",
        f"alpha={percent(result['alpha'])}; beta_II={percent(result['beta_ii'])}; целевая мощность={percent(1-result['beta_ii'])}",
        f"Повторов на сценарий: {result['requested_trials_per_scenario']}; seed: {result['seed']}",
        f"RNG: {result['rng']}; NumPy {result['numpy_version']}; SciPy {result['scipy_version']}",
        f"Режим исходного MDC: {result['mdc_mode']}; исходная неотрицательная смесь: {result.get('reference_fit_method', 'задана явно')}",
        generation,
        fitting,
        f"Обнаружение: свободная оценка {activity_symbol}[i] > L_C своего повтора.",
        "Интервалы: 95%, точные биномиальные Клоппера—Пирсона.",
    ]
    for row in result["scenarios"]:
        interval = row["confidence_interval"]
        bounds = row["failure_probability_bounds"]
        lines.extend([
            "", row["label"] + f"; истинная активность {number(row['activity'], 1000)} мБк/м³", "-" * 90,
            f"Выполнено: {row['n_attempted']}/{row['n_requested']}; успешно: {row['n_valid']}; ошибок: {row['n_failed']}",
            f"Обнаружений: {row['n_detected']}; доля: {percent(row['detection_rate'])}",
            "95% интервал: " + ("—" if interval is None else f"[{percent(interval[0])}; {percent(interval[1])}]"),
            f"Средний L_C: {number(row['mean_L_C'], 1000)}",
            f"Средняя свободная {activity_symbol}: {number(row['mean_free_activity'], 1000)}; СКО: {number(row['std_free_activity'], 1000)}",
            "Для сравнения, доля при фиксированном исходном LC (те же успешные повторы): " + percent(row["fixed_threshold_detection_rate"]),
        ])
        if row["n_failed"] and bounds is not None:
            lines.append(f"Диапазон доли среди всех попыток при крайних исходах ошибок: [{percent(bounds[0])}; {percent(bounds[1])}]")
        for reason, count in row["failure_reasons"].items():
            lines.append(f"Ошибка ({count}): {reason}")
    lines.extend(["", "ПРИМЕЧАНИЯ", *[f"• {note}" for note in result["warnings"]]])
    lines.append("СТАТУС: остановлено, частичный результат" if result["cancelled"] else "СТАТУС: завершено")
    return "\n".join(lines) + "\n"


def save_monte_carlo_report(result: Mapping, path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_monte_carlo_report(result), encoding="utf-8")
