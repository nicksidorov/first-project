"""Читаемый текстовый экспорт XeMat, сохранённый отдельно от GUI и ядра."""

from pathlib import Path

import numpy as np

from process_measurement import METHOD_NAMES


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
        first = "beta" if model.coordinates == "beta_gamma" else "a_c"
        n = model.n_sample_parameters
        names = [f"{first}[{i}]" if i < n else f"gamma[{i - n}]" for i in range(model.M.shape[1])]
    return [str(value) for value in names]


def format_results(results: dict) -> str:
    """Полный протокол; отсутствующие результаты явно помечаются."""
    model, solution = results["model"], results["fit"]
    lines = [
        "РЕЗУЛЬТАТЫ XeMat",
        "=" * 80,
        f"Метод: {METHOD_NAMES[solution.method]}",
        f"Координаты входа и theta: {model.coordinates}",
        "Модель: mu = M @ theta + b; фон входит в ожидаемые отсчёты",
        f"Критерий: {solution.objective:.12g}",
        f"Сходимость: {solution.success}; {solution.message}",
        f"Ранг: {solution.rank}; число обусловленности: {solution.condition_number:.6g}",
        f"Веса: {results['weights_source']}",
        "Неопределённость: локальная ковариация свободной оценки"
        if solution.covariance is not None else
        "Ковариация оценки недоступна; ±σ не выдаётся",
        "",
        "ФАЙЛЫ ДАННЫХ",
        "-" * 80,
        f"Готовая модель: {results.get('model_file')}",
    ]
    for label, key in (("SAMPLE", "sample_file"), ("GASBK", "background_file"), ("DETBK", "detector_file")):
        lines.append(f"{label}: {results.get(key) or 'вектор из NPZ'}")
    metadata = results.get("metadata", {})
    lines.append(f"Фон b: {metadata.get('background_source', 'из NPZ')}")
    for label in ("sample", "background", "detector"):
        temporal = metadata.get(f"{label}_temporal")
        if temporal:
            lines.extend(["", f"ВРЕМЕННЫЕ ДАННЫЕ {label.upper()}"])
            lines.extend(f"{key}: {value}" for key, value in temporal.items() if value is not None)
    if metadata.get("sample_roi_counts"):
        lines.extend(["", "ОТСЧЁТЫ В ROI", f"{'ROI':>6} {'SAMPLE':>15} {'GASBK':>15} {'DETBK':>15}"])
        counts = [metadata.get(f"{label}_roi_counts", {}) for label in ("sample", "background", "detector")]
        for number in sorted(set().union(*(set(item) for item in counts))):
            lines.append(f"{number:6} " + " ".join(f"{item.get(number, 0):15g}" for item in counts))
        lines.append(f"Порядок SAMPLE в M: {metadata.get('sample_roi_numbers')}")
        lines.append(f"Порядок GASBK в M: {metadata.get('gasbk_roi_numbers')}")

    lines.extend(["", "ОЦЕНКИ ПАРАМЕТРОВ theta", "-" * 80])
    std = None if solution.covariance is None else np.sqrt(np.maximum(np.diag(solution.covariance), 0))
    for index, (name, value) in enumerate(zip(parameter_names_for_results(results), solution.theta)):
        uncertainty = "не рассчитана" if std is None else f"{std[index]:.9g}"
        lines.append(f"{name}: {value:.12g}; sigma={uncertainty}")
    if solution.physical_parameters is not None:
        lines.extend(["", "ФИЗИЧЕСКИЕ КООРДИНАТЫ z=[a_c;gamma]", "a_c — Бк/м³; gamma — параметры памяти"])
        if any(name.lower().replace("-", "").startswith("rn") for name in names_for_results(results)):
            lines.append("Радоновая компонента a_c — вспомогательный параметр нормировки, не объёмная активность радона в атмосфере.")
        for index, value in enumerate(solution.physical_parameters):
            label = "a_c" if index < model.n_sample_parameters else "gamma"
            lines.append(f"{label}[{index if label == 'a_c' else index - model.n_sample_parameters}]: {value:.12g}")
    else:
        lines.extend(["", "Физические координаты a_c недоступны: T_c не задана."])

    activity = results.get("activities")
    if activity is not None:
        lines.extend(["", "АКТИВНОСТИ", "-" * 80, "a = H @ beta (Бк); a_c = T_c @ beta (Бк/м³)"])
        if any(name.lower().replace("-", "").startswith("rn") for name in names_for_results(results)):
            lines.append("Для радона значение a_c является вспомогательным параметром, а не атмосферной концентрацией.")
        for index, name in enumerate(names_for_results(results)):
            activity_a = "не вычислена" if activity.a is None else f"{activity.a[index]:.12g}"
            sigma_a = "не вычислена" if activity.std_A is None else f"{activity.std_A[index]:.9g}"
            concentration = "не вычислена" if activity.a_c is None else f"{activity.a_c[index]:.12g}"
            sigma_c = "не вычислена" if activity.std_A_c is None else f"{activity.std_A_c[index]:.9g}"
            lines.append(f"{name}: a={activity_a} Бк; sigma_a={sigma_a}; a_c={concentration} Бк/м³; sigma_c={sigma_c}")
        if activity.K_A is not None:
            lines.extend(["", "Ковариация активностей K_A (Бк²)", np.array2string(activity.K_A, precision=10)])
        if activity.K_A_c is not None:
            lines.extend(["", "Ковариация объёмных активностей K_A_c ((Бк/м³)²)", np.array2string(activity.K_A_c, precision=10)])

    detection = results.get("detection")
    lines.extend(["", "ПОРОГ ОБНАРУЖЕНИЯ L_C И MDC (Бк/м³)", "-" * 80])
    if detection is None:
        lines.append("Не рассчитаны; причина приведена в предупреждениях.")
    else:
        names = names_for_results(results)
        lines.append(f"Режим: {detection.mode}")
        lines.append("Обнаружение проверяется по свободной оценке a_c > L_C, даже при выборе ограниченного метода.")
        for index, component in enumerate(detection.component_indices):
            if detection.converged[index]:
                lines.append(
                    f"{names[component]}: a_c,free={detection.free_activity[index]:.12g}; L_C={detection.L_C[index]:.12g}; "
                    f"MDC={detection.MDC[index]:.12g}; обнаружен={bool(detection.detected[index])}; "
                    f"итераций={detection.iterations[index]}"
                )
            else:
                lines.append(f"{names[component]}: расчёт MDC не сошёлся, число итераций={detection.iterations[index]}")

    lines.extend(["", "ВЕКТОРЫ И МАТРИЦЫ", "-" * 80])
    for name, value in (("M", model.M), ("x (исходные отсчёты)", model.x), ("b", model.b), ("mu", solution.mu), ("x-mu", model.x - solution.mu)):
        lines.extend([name, np.array2string(value, precision=12, threshold=np.inf)])
    if model.T_c is not None:
        lines.extend(["T_c (beta→a_c)", np.array2string(model.T_c, precision=12)])
    if solution.covariance is not None:
        lines.extend(["Ковариация theta", np.array2string(solution.covariance, precision=12)])
    if results.get("warnings"):
        lines.extend(["", "ПРЕДУПРЕЖДЕНИЯ", "-" * 80])
        lines.extend(str(message) for message in results["warnings"])
    return "\n".join(lines) + "\n"


def save_results_to_file(results: dict, output_file: str = "analysis_results.txt") -> None:
    """Сохранение в выбранный пользователем UTF-8 файл."""
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_results(results), encoding="utf-8")
