"""Читаемый текстовый экспорт XeMat, сохранённый отдельно от GUI и ядра."""

from pathlib import Path
from textwrap import fill

import numpy as np

from process_measurement import METHOD_NAMES
from xemat.activities import covariance_to_correlation
from xemat.diagnostics import compute_fit_quality


def quality_for_results(results):
    """Одна диагностика для окна и TXT; поддерживаются прежние словари результатов."""
    quality = results.get("fit_quality")
    return quality if quality is not None else compute_fit_quality(results["model"], results["fit"])


def names_for_results(results):
    """Названия задаются моделью; порядок из старой таблицы не угадывается."""
    model = results["model"]
    names = results.get("isotope_names")
    if names is None or np.asarray(names).shape != (model.n_sample_parameters,):
        names = [f"Компонент {i + 1}" for i in range(model.n_sample_parameters)]
    return [str(value) for value in names]


def parameter_names_for_results(results):
    model = results["model"]
    names = results.get("parameter_names")
    if names is None or np.asarray(names).shape != (model.M.shape[1],):
        first = "beta" if not model.is_activity_coordinates else _activity_symbol(model)
        n = model.n_sample_parameters
        names = ([f"{first}[{i}]" for i in range(n)]
                 + [f"gamma[{i}]" for i in range(model.n_memory_parameters)]
                 + [f"zeta[{i}]" for i in range(model.n_detector_parameters)])
    return [str(value) for value in names]


def _activity_symbol(model):
    """V86 использует a_V; прежние NPZ сохраняют обозначение a_c."""
    return "a_V" if model.has_detector_background else "a_c"


# Ширина основного отчёта рассчитана на моноширинный шрифт.
_WIDTH = 100
_MISSING = "—"


def _number(value, precision=7):
    """Не превращать отсутствующие/неопределённые результаты в нули."""
    if value is None or np.isnan(value):
        return _MISSING
    return f"{float(value):.{precision}g}"


def _section(lines, title):
    lines.extend(["", "=" * _WIDTH, title, "=" * _WIDTH])


def _field(lines, label, value):
    lines.append(f"   {label + ':':<33} {value if value is not None else _MISSING}")


def _table(lines, headers, rows, *, min_width=14):
    """Колонки растут по содержимому; числа справа, подписи слева."""
    rows = [[str(value) for value in row] for row in rows]
    headers = [str(value) for value in headers]
    widths = [max(min_width, len(head), *(len(row[i]) for row in rows))
              for i, head in enumerate(headers)]

    def line(row):
        return "  ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                         for i, value in enumerate(row)).rstrip()

    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    lines.extend([rule, line(headers), rule])
    lines.extend(line(row) for row in rows)
    lines.append(rule)


def _estimates(lines, title, names, values, std):
    lines.extend(["", title])
    if values is None:
        lines.append("Не рассчитаны: преобразование не задано или недоступно.")
        return
    rows = []
    for i, (name, value) in enumerate(zip(names, values)):
        sigma = None if std is None else std[i]
        # Относительная неопределённость при нулевой оценке не равна нулю.
        relative = _MISSING if sigma is None or value == 0 else _number(100 * sigma / abs(value), 5)
        rows.append([name, _number(value), _number(sigma), relative])
    _table(lines, ["Компонента", "Значение", "±σ", "σ / |значение|, %"], rows, min_width=17)


def _matrix(lines, title, value, row_names=None, column_names=None, *, correlation=False):
    """Подписи осей и панели колонок вместо переносов строк внутри ndarray."""
    if value is None:
        lines.extend(["", title, "Не рассчитана: ковариация оценки недоступна."])
        return
    lines.extend(["", title])
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        lines.append("Некорректное числовое преобразование; см. предупреждения.")
        lines.append(np.array2string(np.asarray(value), threshold=np.inf))
        return
    if matrix.ndim == 0:
        lines.append(f"   {_number(matrix.item(), 12)}")
        return
    if matrix.ndim > 2:
        # Ошибочная необязательная H/Gamma не отменяет корректную регрессию.
        # Сохраняем диагностические данные, как и предупреждение процесса.
        lines.append(f"Некорректная размерность преобразования: {matrix.shape}; см. предупреждения.")
        lines.append(np.array2string(matrix, precision=12, threshold=np.inf))
        return
    if matrix.ndim == 1:
        matrix = matrix[:, None]
        column_names = ["Значение"]
    nr, nc = matrix.shape
    if row_names is None or len(row_names) != nr:
        row_names = [f"[{i}]" for i in range(nr)]
    if column_names is None or len(column_names) != nc:
        column_names = [f"[{i}]" for i in range(nc)]
    # Полная матрица сохраняется; широкие матрицы печатаются блоками.
    block_size = 5
    for first in range(0, nc, block_size):
        last = min(first + block_size, nc)
        if nc > block_size:
            lines.append(f"   Столбцы {first + 1}–{last} из {nc}")
        rows = []
        for i, label in enumerate(row_names):
            cells = []
            for value in matrix[i, first:last]:
                cells.append(f"{value:.4f}" if correlation and np.isfinite(value) else _number(value, 12))
            rows.append([label, *cells])
        _table(lines, ["Строка / столбец", *column_names[first:last]], rows,
               min_width=10 if correlation else 16)


def _observation_names(results):
    metadata = results.get("metadata", {})
    labels = [f"{role} ROI {roi}" for role, key in
              (("SAMPLE", "sample_roi_numbers"), ("GASBK", "gasbk_roi_numbers"),
               ("DETBK", "detector_roi_numbers"))
              for roi in metadata.get(key, [])]
    if len(labels) != results["model"].x.size:
        labels = [f"x[{i}]" for i in range(results["model"].x.size)]
    return labels


def _input_data(lines, results):
    metadata = results.get("metadata", {})
    _section(lines, "1. ФАЙЛЫ И УСЛОВИЯ ИЗМЕРЕНИЙ")
    for label, key in (("Проба (SAMPLE)", "sample_file"), ("Газовый фон (GASBK)", "background_file"),
                       ("Фон детектора (DETBK)", "detector_file")):
        _field(lines, label, results.get(key) or "данные из NPZ")
    _field(lines, "Модель", results.get("model_file") or "построена из constants.py и спектров")
    if results.get("sample_file"):
        _field(lines, "Границы ROI", "фиксированные, constants.py" if results.get("use_fixed_roi") else "из файлов спектров")
    _field(lines, "Калибровка", metadata.get("calibration_source", "подготовленная модель NPZ"))
    if metadata.get("efficiency_source"):
        _field(lines, "Эффективности ε", metadata["efficiency_source"])
    if results["model"].has_detector_background:
        _field(lines, "Фон детектора ζ", metadata.get("background_source", "оценивается совместно по SAMPLE/GASBK/DETBK"))
    else:
        _field(lines, "Фон b", metadata.get("background_source", "из NPZ"))
    detector_info = metadata.get("detector_info", {})
    for role, key in (("ПРОБА — SAMPLE", "sample"), ("ГАЗОВЫЙ ФОН — GASBK", "background"),
                      ("ФОН ДЕТЕКТОРА — DETBK", "detector")):
        temporal = metadata.get(f"{key}_temporal", {})
        detector = detector_info.get(key, {})
        if not temporal and not detector:
            continue
        lines.extend(["", role, "-" * _WIDTH])
        if detector:
            _field(lines, "Детектор", f"{detector.get('id', _MISSING)} ({detector.get('name', _MISSING)})")
        for name, label, unit in (
            ("start_time", "Начало измерения", ""),
            ("collection_start", "Начало пробоотбора", ""),
            ("collection_end", "Конец пробоотбора", ""),
            ("real_time", "Реальное время", "с"),
            ("live_time", "Живое время", "с"),
            ("air_volume", "Объём воздуха из файла", "м³"),
            ("xenon_volume", "Объём ксенона", "см³"),
        ):
            value = temporal.get(name)
            if value is not None:
                _field(lines, label, f"{_number(value, 10)} {unit}" if unit else str(value))
    times = metadata.get("time_parameters", {})
    if times:
        lines.extend(["", "ВРЕМЕННЫЕ ИНТЕРВАЛЫ", "-" * _WIDTH])
        for key, label in (("tau", "τ: старт SAMPLE − старт GASBK"),
                           ("t_a", "t_a: длительность пробоотбора"), ("t_p", "t_p: время ожидания")):
            if key in times:
                _field(lines, label, f"{_number(times[key], 10)} с  ({_number(times[key] / 3600)} ч)")

    _section(lines, "2. ИСХОДНЫЕ ОТСЧЁТЫ В ROI")
    counts = [metadata.get(f"{key}_roi_counts", {}) for key in ("sample", "background", "detector")]
    if any(counts):
        rows = [[number, *[_MISSING if number not in item else f"{item[number]:.0f}" for item in counts]]
                for number in sorted(set().union(*(set(item) for item in counts)))]
        _table(lines, ["ROI", "Проба SAMPLE", "Фон GASBK", "Фон DETBK"], rows, min_width=18)
        _field(lines, "ROI SAMPLE в расчёте", ", ".join(map(str, metadata.get("sample_roi_numbers", []))))
        _field(lines, "ROI GASBK в расчёте", ", ".join(map(str, metadata.get("gasbk_roi_numbers", []))))
        if results["model"].has_detector_background:
            _field(lines, "ROI DETBK в расчёте", ", ".join(map(str, metadata.get("detector_roi_numbers", []))))
        lines.append("Все ROI приведены для справки; общая сумма не выводится, так как области могут пересекаться.")
    else:
        lines.append("Исходные отсчёты взяты из NPZ и приведены построчно в разделе 8.")


def format_results(results: dict) -> str:
    """Отчёт в стиле старого проекта; форматирование не меняет расчёты."""
    model, solution = results["model"], results["fit"]
    metadata = results.get("metadata", {})
    names = names_for_results(results)
    parameters = parameter_names_for_results(results)
    n = model.n_sample_parameters
    memory_end = n + model.n_memory_parameters
    activity_symbol = _activity_symbol(model)
    lines = ["=" * _WIDTH, "XeMat — РЕЗУЛЬТАТЫ ОБРАБОТКИ БЕТА-ГАММА СПЕКТРОВ".center(_WIDTH).rstrip(),
             "=" * _WIDTH]
    _field(lines, "Метод", METHOD_NAMES[solution.method])
    _field(lines, "Совместная обработка", "SAMPLE + предшествующий GASBK" +
           (" + DETBK" if model.has_detector_background else ""))
    _field(lines, "Сходимость", "достигнута" if solution.success else "НЕ достигнута")
    lines.append("Обозначение «—»: величина не рассчитана или не определена; это не ноль.")
    _input_data(lines, results)

    _section(lines, "3. РЕЗУЛЬТАТЫ РЕГРЕССИИ")
    if model.has_detector_background:
        lines.append("Модель V86: mu = M @ theta; x = [SAMPLE; GASBK; DETBK].")
        lines.append("ζ — оцениваемые средние отсчёты DETBK; фон входит в модель всех трёх спектров.")
        lines.append("Статистическая неопределённость DETBK включена через совместное оценивание ζ.")
    else:
        lines.append("Модель: mu = M @ theta + b; фон входит в ожидаемые отсчёты.")
    coordinates = model.input_coordinate_label
    _field(lines, "Координаты theta", f"{coordinates} ({model.coordinates})")
    quality = quality_for_results(results)
    _field(lines, quality.label, _number(quality.reduced_statistic))
    _field(lines, f"Исходная статистика {quality.symbol}", _number(quality.statistic, 12))
    _field(lines, "Степени свободы ν = N − ранг",
           f"{quality.n_observations} − {quality.rank} = {quality.degrees_of_freedom}"
           + (" (номинальные)" if quality.nominal_dof else ""))
    lines.append("Около 1 — ориентир при достаточных отсчётах и корректной модели дисперсий;")
    lines.append("при малых отсчётах χ²/ν и D/ν могут заметно отклоняться от 1 даже для верной модели.")
    lines.extend(quality.warnings)
    _field(lines, "Ранг", solution.rank)
    _field(lines, "Число обусловленности", _number(solution.condition_number))
    _field(lines, "Сообщение решателя", solution.message)
    _field(lines, "Веса", results["weights_source"])
    if solution.covariance is None:
        lines.append("Ковариация оценки недоступна: ±σ и относительная неопределённость не выдаются.")
    else:
        lines.append("±σ параметров: локальная статистическая ковариация оценки.")
    if solution.method in ("nnls", "poisson_nonnegative"):
        lines.append(f"Неотрицательность наложена на z={model.physical_coordinate_label.replace(' ', '')}. "
                     f"Условие beta ≥ 0 не используется вместо {activity_symbol} ≥ 0.")
    std = None if solution.covariance is None else np.sqrt(np.maximum(np.diag(solution.covariance), 0))
    first_title = ("ПАРАМЕТРЫ ПРОБЫ β (амплитуды в отсчётах)" if not model.is_activity_coordinates else
                   f"ПАРАМЕТРЫ ПРОБЫ {activity_symbol} (Бк/м³)")
    _estimates(lines, first_title, parameters[:n], solution.theta[:n], None if std is None else std[:n])
    if memory_end > n:
        _estimates(lines, "ПАРАМЕТРЫ ПАМЯТИ γ", parameters[n:memory_end], solution.theta[n:memory_end],
                   None if std is None else std[n:memory_end])
    if model.has_detector_background:
        _estimates(lines, "ПАРАМЕТРЫ ФОНА ДЕТЕКТОРА ζ (средние отсчёты за измерение DETBK)",
                   parameters[memory_end:], solution.theta[memory_end:],
                   None if std is None else std[memory_end:])

    _section(lines, "4. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ РАСЧЁТА АКТИВНОСТЕЙ")
    if "xenon_volume_cm3" in metadata:
        _field(lines, "Объём ксенона V_Xe", f"{_number(metadata['xenon_volume_cm3'])} см³")
    if "effective_air_volume_m3" in metadata:
        _field(lines, "V0 = V_Xe / 0.087", f"{_number(metadata['effective_air_volume_m3'])} м³")
    elif "V0" in results.get("calibration_matrices", {}):
        _field(lines, "Объём V0 из модели", f"{_number(results['calibration_matrices']['V0'])} м³")
    transform_symbol = "T_V" if model.has_detector_background else "T_c"
    activity = results.get("activities")
    physical_activity_only = ((activity is None or activity.a_c is None)
                              and solution.physical_parameters is not None)
    if physical_activity_only:
        # Подготовленная S_V может сразу оценивать a_V без H и T_V:
        # объёмная активность доступна, даже если нельзя восстановить beta и a.
        concentration = solution.physical_parameters[:n]
        concentration_covariance = (None if solution.physical_covariance is None else
                                    solution.physical_covariance[:n, :n])
        concentration_std = (None if concentration_covariance is None else
                             np.sqrt(np.maximum(np.diag(concentration_covariance), 0)))
        concentration_correlation = (None if concentration_covariance is None else
                                     covariance_to_correlation(concentration_covariance))
        lines.append(f"Объёмная активность {activity_symbol} взята из оценённых физических координат (Бк/м³).")
        lines.append("Её ±σ, если доступна, — статистическая; систематические вклады здесь не рассчитаны.")
    else:
        concentration = None if activity is None else activity.a_c
        concentration_std = None if activity is None else activity.std_A_c
        concentration_correlation = None if activity is None else activity.correlation_A_c
        lines.append(f"a = H @ beta (Бк); {activity_symbol} = {transform_symbol} @ beta (Бк/м³).")
        lines.append("Если ±σ рассчитана, она включает статистическую часть и заданные систематические вклады.")
    if any(name.lower().replace("-", "").startswith("rn") for name in names):
        lines.append(f"Радоновая компонента {activity_symbol} — вспомогательный параметр нормировки,")
        lines.append("не объёмная активность радона в атмосфере.")
    if activity is None or activity.a is None:
        lines.extend(["", "АКТИВНОСТЬ В ДЕТЕКТОРЕ a (Бк)",
                      "Активность в детекторе не рассчитана; см. предупреждения."])
    else:
        _estimates(lines, "АКТИВНОСТЬ В ДЕТЕКТОРЕ a НА НАЧАЛО SAMPLE (Бк)", names, activity.a, activity.std_A)
    _estimates(lines, f"ОБЪЁМНАЯ АКТИВНОСТЬ {activity_symbol} (Бк/м³)", names,
               concentration, concentration_std)

    _section(lines, "5. ПОРОГ ОБНАРУЖЕНИЯ L_C И MDC")
    detection = results.get("detection")
    if detection is None:
        lines.append("Не рассчитаны; причина приведена в предупреждениях.")
    else:
        mode = {"iterative": "итерационный (iterative)", "fixed": "фиксированная ковариация (fixed)"}
        _field(lines, "Режим MDC", mode.get(detection.mode, detection.mode))
        _field(lines, "Вероятности ошибок", f"α = {_number(detection.alpha)}; β_II = {_number(detection.beta_ii)}")
        lines.append("Все активности и пороги в этой таблице — в мБк/м³ (1 Бк/м³ = 1000 мБк/м³).")
        lines.append(f"Обнаружение: свободная оценка {activity_symbol} > L_C, в том числе при выборе ограниченного метода.")
        rows = []
        for i, component in enumerate(detection.component_indices):
            converged = detection.converged[i]
            rows.append([names[component], _number(1000 * detection.free_activity[i]),
                         _number(1000 * detection.L_C[i]),
                         _number(1000 * detection.MDC[i]) if converged else "не сошёлся",
                         ("да" if detection.detected[i] else "нет") if converged else _MISSING,
                         detection.iterations[i]])
        _table(lines, ["Изотоп", f"{activity_symbol}, свободная", "L_C", "MDC", "Обнаружен", "Итераций"], rows, min_width=12)

    _section(lines, "6. КОРРЕЛЯЦИОННЫЕ МАТРИЦЫ ПИРСОНА")
    lines.append("Связь между оценками: r_ij = K_ij / sqrt(K_ii × K_jj). Коэффициенты безразмерны.")
    lines.append("От +1 (прямая связь) до −1 (обратная связь); 0 — отсутствие линейной корреляции.")
    lines.append("При нулевой дисперсии или недоступной ковариации корреляция не определена: «—».")
    if physical_activity_only:
        lines.append(f"Для объёмных активностей {activity_symbol} приведена статистическая корреляция; систематические вклады не рассчитаны.")
    else:
        lines.append("Для активностей учтены статистическая часть и заданные систематические вклады.")
    _matrix(lines, "Активности в детекторе a", None if activity is None else activity.correlation_A,
            names, names, correlation=True)
    _matrix(lines, f"Объёмные активности {activity_symbol}", concentration_correlation,
            names, names, correlation=True)
    _matrix(lines, f"Параметры theta={coordinates} (статистическая корреляция)",
            None if solution.covariance is None else covariance_to_correlation(solution.covariance),
            parameters, parameters, correlation=True)

    _section(lines, "7. ПРЕДУПРЕЖДЕНИЯ И ПРИНЯТЫЕ ДОПУЩЕНИЯ")
    warnings = results.get("warnings", [])
    if warnings:
        for i, message in enumerate(warnings, 1):
            prefix = f"{i}. "
            lines.append(fill(str(message), width=_WIDTH, initial_indent=prefix,
                              subsequent_indent=" " * len(prefix), break_long_words=False,
                              break_on_hyphens=False))
    else:
        lines.append("Предупреждений нет.")

    _section(lines, "8. ПОДРОБНЫЙ ПРОТОКОЛ: ФИЗИЧЕСКИЕ КООРДИНАТЫ И ОСТАТКИ")
    lines.extend(["", f"ФИЗИЧЕСКИЕ КООРДИНАТЫ z={model.physical_coordinate_label.replace(' ', '')}",
                  f"{activity_symbol} — Бк/м³; gamma — параметры памяти. Индексы начинаются с нуля."])
    if model.has_detector_background:
        lines.append("zeta — средние отсчёты фона за измерение DETBK, отдельный параметр для каждой ROI.")
    if solution.physical_parameters is None:
        lines.append(f"Физические координаты {activity_symbol} недоступны: {transform_symbol} не задана.")
    else:
        labels = ([f"{activity_symbol}[{i}] {names[i]}" for i in range(n)]
                  + [f"gamma[{i}]" for i in range(model.n_memory_parameters)]
                  + [f"zeta[{i}]" for i in range(model.n_detector_parameters)])
        _table(lines, ["Координата", "Значение"],
               [[label, _number(value, 12)] for label, value in zip(labels, solution.physical_parameters)], min_width=18)
    observations = _observation_names(results)
    lines.extend(["", "ИСХОДНЫЕ ОТСЧЁТЫ, ФОН, ОЖИДАЕМЫЕ ОТСЧЁТЫ И ОСТАТКИ"])
    if model.has_detector_background:
        background = model.M[:, memory_end:] @ solution.theta[memory_end:]
        background_label = "Фон, оценка"
        lines.append("x — исходные отсчёты трёх спектров; mu — совместная модель; остаток = x − mu.")
        lines.append("Фон, оценка: κ_s ζ в SAMPLE, κ_g ζ в GASBK, ζ в DETBK; это оценка, а не вычитание из x.")
    else:
        background, background_label = model.b, "b"
        lines.append("x — исходные отсчёты; b — фиксированный фон; mu — модель; остаток = x − mu.")
    _table(lines, ["Наблюдение", "x", background_label, "mu", "x − mu"],
           [[label, f"{x:.0f}", _number(b, 12), _number(mu, 12), _number(x - mu, 12)]
            for label, x, b, mu in zip(observations, model.x, background, solution.mu)], min_width=16)

    _section(lines, "9. ПОДРОБНЫЙ ПРОТОКОЛ: МАТРИЦЫ И КАЛИБРОВКА")
    lines.append("В матрицах и подробных векторах — до 12 значащих цифр. Индексы начинаются с нуля.")
    if results.get("calibration_mode") == "constants":
        lines.append("Исходные Rs/Rg: ROI 1, 2, 3, 5, 6, 7, 8; изотопы Rn222, Xe135, Xe133, Xe131m, Xe133m.")
        lines.append("Чувствительности ксенона: I_c = ε_SAMPLE × BR; I_g = ε_GASBK × BR.")
        if metadata.get("sample_efficiencies") is not None and metadata.get("gasbk_efficiencies") is not None:
            lines.append("Эффективности ε в долях, без branching:")
            _table(lines, ["Изотоп", "ε SAMPLE", "ε GASBK"], [
                [name, _number(sample, 12), _number(gasbk, 12)]
                for name, sample, gasbk in zip(
                    metadata["efficiency_isotope_names"], metadata["sample_efficiencies"], metadata["gasbk_efficiencies"],
                )
            ], min_width=16)
        _field(lines, "Модель радона", metadata.get("radon_mode"))
        for key, label in (("sample", "SAMPLE"), ("gasbk", "GASBK"), ("sample_memory", "память в SAMPLE")):
            _field(lines, f"H_Rn: {label}", f"{_number(metadata.get('radon_h', {}).get(key), 12)} Бк/отсчёт")
        if metadata.get("radon_mode") == "calibrated":
            lines.append("Нулевые первые элементы I_c и I_g служебные: радон задан отдельными H_Rn.")
    _matrix(lines, "M — совместная матрица в координатах theta", model.M, observations, parameters)
    if model.T_c is not None:
        _matrix(lines, f"{transform_symbol} — переход beta → {activity_symbol}", model.T_c, names, names)
    titles = {
        "F": "F — перенос памяти GASBK → SAMPLE", "H": "H — нормировка beta → a (Бк)",
        "Gamma": "Gamma — поправка на цепочку распада", "G_p": "G_p — поправка за время ожидания",
        "G_s": ("G_a — поправка за время пробоотбора (в коде G_s)" if model.has_detector_background else
                "G_s — поправка за время пробоотбора"), "V0": "V0 — объём воздуха (м³)",
        "delta_I": "delta_I — относительные неопределённости чувствительностей",
        "delta_V0": "delta_V0 — относительная неопределённость объёма",
        "kappa_s": "κ_s = t_sl / t_bl — перенос фона DETBK → SAMPLE",
        "kappa_g": "κ_g = t_gl / t_bl — перенос фона DETBK → GASBK",
    }
    for key, value in results.get("calibration_matrices", {}).items():
        array = np.asarray(value)
        row_names = names
        if key in ("R_c0", "R_c", "R_g0", "R_g1", "R_g2"):
            roi_key = "gasbk_roi_numbers" if key == "R_g2" else "sample_roi_numbers"
            row_names = [f"ROI {roi}" for roi in metadata.get(roi_key, [])]
        _matrix(lines, titles.get(key, key), array, row_names, names)
    lines.extend(["", "=" * _WIDTH,
                  "АНАЛИЗ ЗАВЕРШЁН" if solution.success else "РАСЧЁТ НЕ СОШЁЛСЯ",
                  f"Метод: {METHOD_NAMES[solution.method]}", "=" * _WIDTH])
    return "\n".join(lines) + "\n"


def save_results_to_file(results: dict, output_file: str = "analysis_results.txt") -> None:
    """Сохранение в выбранный пользователем UTF-8 файл."""
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_results(results), encoding="utf-8")
