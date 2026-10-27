"""Распределения свободных оценок активностей при повторных измерениях.

V86: x=[SAMPLE;GASBK;DETBK], физические координаты [a_V;gamma;zeta].
Два генератора: Pois(mu_+) после одной полной неотрицательной ММП-подгонки
либо Pois(x) с наблюдаемыми отсчётами в роли средних. В каждом повторе
выполняется только одна свободная подгонка; Lc/MDC здесь не нужны.
"""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from .model import JointModel
from .solvers import fit


GENERATION_NAMES = {
    "fitted": "По ожидаемым отсчётам неотрицательного ММП",
    "observed": "По наблюдаемым отсчётам",
}
METHOD_NAMES = {"wls": "Свободный взвешенный МНК", "poisson": "Свободный пуассоновский ММП"}


def _component_names(results):
    n = results["model"].n_sample_parameters
    names = results.get("isotope_names")
    if names is None or np.asarray(names).shape != (n,):
        return [f"Компонент {index + 1}" for index in range(n)]
    return [str(name) for name in names]


def eligible_components(results):
    """Компоненты свежей пробы с физическими координатами; LC не требуется."""
    if not isinstance(results, Mapping):
        return []
    model, solution = results.get("model"), results.get("fit")
    if not isinstance(model, JointModel) or solution is None or not solution.success:
        return []
    if model.coordinate_transform is None or solution.physical_parameters is None:
        return []
    if not np.all(np.isfinite(solution.physical_parameters)):
        return []
    return [index for index, name in enumerate(_component_names(results))
            if not name.lower().replace("-", "").replace(" ", "").startswith("rn")]


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}: требуется целое число не меньше {minimum}.")
    return int(value)


def _checked_fit(model, method, options):
    solution = fit(model, method=method, **options)
    if not solution.success:
        # Сообщение SciPy об остановке по шагу ещё не подтверждает ККТ.
        # Сохраняем и причину, по которой наша проверка отвергла решение.
        details = "; ".join([str(solution.message), *map(str, solution.warnings)])
        raise RuntimeError(f"{method}: решатель не сошёлся: {details}")
    physical = solution.physical_parameters
    if physical is None or np.asarray(physical).shape != (model.M.shape[1],) or not np.all(np.isfinite(physical)):
        raise ValueError(f"{method}: нет конечной оценки в {model.physical_coordinate_label}.")
    return solution


def _stat_sigma(solution, indices):
    """Только статистическая σ свободной оценки; систематика не разыгрывается."""
    covariance = solution.physical_covariance
    if covariance is None:
        return None
    variance = np.diag(covariance)[indices]
    if np.any(~np.isfinite(variance)) or np.any(variance < 0):
        return None
    return np.sqrt(variance)


def _quadratic_options(results):
    options = results.get("quadratic_options")
    if options is None:
        if str(results.get("weights_source", "")).startswith("Внешн"):
            raise ValueError("В сохранённом расчёте отсутствуют внешние веса. Повторите основной расчёт.")
        options = {}
    return {key: None if options.get(key) is None else np.array(options[key], dtype=float, copy=True)
            for key in ("weights", "observation_covariance")}


def _summaries(samples, sigma_samples, indices, names):
    summaries = []
    for column, (index, name) in enumerate(zip(indices, names)):
        values = samples[:, column]
        sigmas = sigma_samples[:, column]
        sigmas = sigmas[np.isfinite(sigmas)]
        summaries.append({
            "component_index": int(index), "name": name,
            "mean": float(np.mean(values)) if values.size else None,
            "std": float(np.std(values, ddof=1)) if values.size > 1 else None,
            "median": float(np.median(values)) if values.size else None,
            "q025": float(np.quantile(values, .025)) if values.size else None,
            "q975": float(np.quantile(values, .975)) if values.size else None,
            "mean_stat_sigma": float(np.mean(sigmas)) if sigmas.size else None,
            "sigma_count": int(sigmas.size),
        })
    return summaries


def run_from_results(
    results: Mapping, *, generation: str = "fitted", method: str = "wls",
    n_trials: int = 1000, seed: int = 12345,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
):
    """Генерировать все наблюдения и собирать свободные объёмные активности.

    Средние генерации не зависят от выбранного свободного метода. При
    одинаковых средних и seed последовательность исходных векторов одинакова,
    даже если один из методов сообщил ошибку: повторного розыгрыша нет.
    Неотрицательный ММП выполняется один раз только в режиме fitted.
    В режиме observed исходный ноль остаётся нулём во всех реализациях.

    Все активности/σ результата — Бк/м³. Процентили описывают полученную
    выборку оценок, а не доверительный интервал неизвестной активности.
    """
    if generation not in GENERATION_NAMES:
        raise ValueError("Генерация: выберите fitted или observed.")
    if method not in METHOD_NAMES:
        raise ValueError("Гистограммы доступны только для свободных методов wls и poisson.")
    n_trials = _integer(n_trials, "Число повторов", 1)
    seed = _integer(seed, "Seed", 0)
    indices = np.asarray(eligible_components(results), dtype=int)
    if not indices.size:
        raise ValueError("Сначала выполните успешный расчёт с физическими координатами активностей пробы.")
    model = results["model"]
    names = [_component_names(results)[index] for index in indices]
    options = _quadratic_options(results)
    fitting_options = options if method == "wls" else {}
    if progress is not None:
        progress(0, n_trials)
    # Эта свободная оценка нужна для линии исходного измерения и его σ.
    # Выбранный в основном окне ограниченный метод не меняет её смысл.
    original = _checked_fit(model, method, fitting_options)
    generator_fit = None
    if generation == "fitted":
        # Одна общая физически допустимая смесь для обоих свободных методов.
        # Никакой выбранный нуклид не обнуляется: это не эксперимент LC/LD.
        generator_fit = _checked_fit(model, "poisson_nonnegative", {})
        if np.any(generator_fit.physical_parameters < 0):
            raise ValueError("Генератор требует неотрицательных физических параметров.")
        mu = np.array(generator_fit.mu, dtype=float, copy=True)
    else:
        mu = np.array(model.x, dtype=float, copy=True)
    if np.any(~np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError("Средние генерации должны быть конечными и неотрицательными; проверьте подгонку.")

    rng = np.random.default_rng(seed)
    samples, sigma_samples, trial_indices = [], [], []
    failures = Counter()
    covariance_warnings = Counter()
    attempted, cancelled = 0, False
    for trial_index in range(n_trials):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        # В V86 вектор имеет 21 элемент: все три измерения независимы
        # при заданных средних. Нельзя строить SAMPLE из случайного DETBK.
        x = rng.poisson(mu)
        attempted += 1
        try:
            trial = JointModel(model.M, x, model.b, model.n_sample_parameters,
                               model.T_c, model.coordinates, model.n_detector_parameters)
            solution = _checked_fit(trial, method, fitting_options)
            values = solution.physical_parameters[indices].copy()
            sigmas = _stat_sigma(solution, indices)
            samples.append(values)  # Отрицательные свободные оценки сохраняются.
            sigma_samples.append(np.full(indices.size, np.nan) if sigmas is None else sigmas)
            trial_indices.append(trial_index)
            covariance_warnings.update(solution.warnings)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failures[f"{type(exc).__name__}: {exc}"] += 1
        if progress is not None:
            progress(attempted, n_trials)

    values = np.asarray(samples, dtype=float).reshape(-1, indices.size)
    sigmas = np.asarray(sigma_samples, dtype=float).reshape(-1, indices.size)
    warnings = [
        "Показано распределение свободных оценок при заданных средних отсчётах. Отрицательные оценки сохранены.",
        "Матрицы, времена, объём и калибровка фиксированы; систематические погрешности не разыгрываются. Сравниваются только статистические σ.",
        "Процентили 2.5%–97.5% описывают центральные 95% реализаций; это не доверительный интервал истинной активности.",
    ]
    if model.has_detector_background:
        warnings.append("В каждом повторе разыгрываются SAMPLE, GASBK и DETBK и заново оцениваются проба, память и детекторный фон ζ.")
    else:
        warnings.append("Прежняя модель NPZ: разыгрываются наблюдения модели; аддитивный фон b считается известным, отдельного блока DETBK нет.")
    if generation == "observed":
        warnings.append("Средние равны наблюдаемому x: ROI с нулевыми отсчётами остаются нулевыми во всех повторах. Заданная истинная активность для этой генерации неизвестна.")
    if failures:
        warnings.append("Есть неуспешные подгонки: гистограммы и статистика построены только по успешным повторам; повторного розыгрыша не было.")
    if values.shape[0] < 2:
        warnings.append("Для оценки стандартного отклонения нужны хотя бы две успешные реализации.")
    if values.shape[0] and np.any(~np.isfinite(sigmas)):
        warnings.append("В части успешных подгонок статистическая σ недоступна; эти оценки сохранены в гистограммах.")
    warnings.extend(f"Исходная свободная оценка: {message}" for message in original.warnings)
    if generator_fit is not None:
        # Обычная ковариация ограниченного оценивателя здесь не используется;
        # в протоколе ниже сохраняем все диагностические сообщения подгонки.
        generator_warnings = list(generator_fit.warnings)
    else:
        generator_warnings = []
    return {
        "experiment": "activity_distributions", "experiment_version": 1,
        "generation": generation, "method": method, "units": "Bq/m3",
        "component_indices": indices, "component_names": names,
        "activity_samples": values, "trial_stat_sigma_samples": sigmas,
        "successful_trial_indices": np.asarray(trial_indices, dtype=int),
        "original_activity": original.physical_parameters[indices].copy(),
        "original_stat_sigma": _stat_sigma(original, indices),
        "generation_activity": (None if generator_fit is None else generator_fit.physical_parameters[indices].copy()),
        "generation_mu": mu,
        "generator_fit_method": None if generator_fit is None else "poisson_nonnegative",
        "generator_physical_parameters": None if generator_fit is None else generator_fit.physical_parameters.copy(),
        "generator_warnings": generator_warnings,
        "summary": _summaries(values, sigmas, indices, names),
        "n_requested": n_trials, "n_attempted": attempted, "n_valid": len(samples),
        "n_failed": sum(failures.values()), "failure_reasons": dict(failures),
        "trial_fit_warnings": dict(covariance_warnings),
        "seed": seed, "cancelled": cancelled, "warnings": warnings,
        "detector_background_estimated": model.has_detector_background,
        "physical_coordinates": model.physical_coordinate_label,
        "model": model, "quadratic_options": options,
        "rng": type(rng.bit_generator).__name__, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "source": {key: results.get(key) for key in (
            "sample_file", "background_file", "detector_file", "model_file",
            "use_fixed_roi", "use_fixed_efficiencies", "metadata",
        )},
    }


def format_distribution_report(result: Mapping) -> str:
    """Самостоятельный TXT; значения в файле переведены в мБк/м³."""
    def number(value):
        return "—" if value is None or not np.isfinite(value) else f"{1000 * value:.7g}"

    rows = [
        "XeMat — МОНТЕ-КАРЛО: РАСПРЕДЕЛЕНИЯ СВОБОДНЫХ АКТИВНОСТЕЙ",
        "=" * 112,
        f"Генерация: {GENERATION_NAMES[result['generation']]}",
        f"Метод: {METHOD_NAMES[result['method']]}; координаты: {result['physical_coordinates']}",
        "Единицы всех активностей и σ ниже: мБк/м³.",
        f"Seed: {result['seed']}; RNG: {result['rng']}; NumPy {result['numpy_version']}; SciPy {result['scipy_version']}",
        f"Попыток: {result['n_attempted']} / {result['n_requested']}; успешно: {result['n_valid']}; неуспешно: {result['n_failed']}",
    ]
    if result.get("cancelled"):
        rows.append("Эксперимент остановлен; результаты частичные.")
    for key, label in (("sample_file", "SAMPLE"), ("background_file", "GASBK"),
                       ("detector_file", "DETBK"), ("model_file", "NPZ")):
        if result.get("source", {}).get(key):
            rows.append(f"{label}: {result['source'][key]}")
    rows.extend(["", "Опорные значения:",
                 f"{'Изотоп':<16}{'Исходная свободная':>24}{'σ исходная, стат.':>24}{'Задано в генерации':>24}"])
    for i, name in enumerate(result["component_names"]):
        sigma = None if result["original_stat_sigma"] is None else result["original_stat_sigma"][i]
        truth = None if result["generation_activity"] is None else result["generation_activity"][i]
        rows.append(f"{name:<16}{number(result['original_activity'][i]):>24}{number(sigma):>24}{number(truth):>24}")
    rows.extend(["", "Распределения успешных оценок:",
                 f"{'Изотоп':<16}{'Среднее':>16}{'СКО MC':>16}{'Медиана':>16}{'2.5%':>16}{'97.5%':>16}{'Средняя σ стат.':>20}"])
    for row in result["summary"]:
        rows.append(f"{row['name']:<16}" + "".join(f"{number(row[key]):>16}" for key in ("mean", "std", "median", "q025", "q975"))
                    + f"{number(row['mean_stat_sigma']):>20}")
        rows.append(f"  σ доступна в {row['sigma_count']} из {result['n_valid']} успешных подгонок.")
    rows.extend(["", "Замечания:"])
    rows.extend(f"- {message}" for message in result.get("warnings", []))
    for message, count in result.get("failure_reasons", {}).items():
        rows.append(f"Неуспешные подгонки ({count}): {message}")
    for message, count in result.get("trial_fit_warnings", {}).items():
        rows.append(f"Диагностика успешных подгонок ({count}): {message}")
    rows.extend(["", "Средние генерации по ROI, в отсчётах:",
                 np.array2string(np.asarray(result["generation_mu"]), precision=12, threshold=np.inf)])
    return "\n".join(rows) + "\n"


def save_distribution_report(result: Mapping, path) -> None:
    Path(path).write_text(format_distribution_report(result), encoding="utf-8")
