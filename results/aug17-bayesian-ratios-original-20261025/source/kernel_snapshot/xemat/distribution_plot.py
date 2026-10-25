"""Гистограммы свободных оценок для окна и экспорта Монте-Карло.

Модуль работает непосредственно с Figure: не выбирает backend Matplotlib,
не открывает окна и не зависит от Qt. Входные активности имеют единицы
Бк/м³; перевод в мБк/м³ выполняется только здесь, при отображении.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np
from matplotlib.figure import Figure


_ACTIVITY_SCALE = 1000.0
_ORIGINAL_LABEL = "Исходная свободная оценка"
_GENERATION_LABEL = "Задано при генерации"


def _number(value: Any) -> str:
    """Единый формат сводки; None означает недоступное СКО."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value) * _ACTIVITY_SCALE:.4g}"


def _component_value(values: Any, index: int) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if index >= array.size or not np.isfinite(array[index]):
        return None
    return float(array[index]) * _ACTIVITY_SCALE


def _histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Сохраняем отрицательные оценки и корректно показываем константу."""
    low, high = float(values.min()), float(values.max())
    if low == high:
        # У константы нет наблюдаемой ширины распределения. Небольшой
        # интервал нужен только для рисования столбика и не является σ.
        padding = max(abs(low) * .05, .001)
        low, high = low - padding, high + padding
    return np.linspace(low, high, bins + 1)


def render_distribution_figure(
    result: Mapping[str, Any],
    figure: Figure | None = None,
    bins: int = 30,
) -> Figure:
    """Построить гистограммы, повторно используя ``figure`` при её передаче.

    ``result`` — результат эксперимента распределений. Одна строка массива
    ``activity_samples`` соответствует одной успешной подгонке; столбцы
    соответствуют ``component_names``. Данные и результат не изменяются.

    Линия заданной активности имеет смысл только при генерации по модели.
    При Poisson(x) наблюдения не задают отдельную «истинную активность»,
    поэтому такая линия не показывается даже при наличии лишних метаданных.
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("Число интервалов гистограммы должно быть положительным целым.")

    names = list(result.get("component_names", []))
    n_components = len(names)
    samples = np.asarray(result.get("activity_samples", []), dtype=float)
    if samples.size == 0:
        samples = np.empty((0, n_components))
    elif samples.ndim == 1 and n_components == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[1] != n_components:
        raise ValueError("Столбцы activity_samples должны соответствовать component_names.")

    if figure is None:
        figure = Figure(figsize=(10, 7), facecolor="white")
    else:
        figure.clear()
    # Layout рассчитывается по фактическим размерам текста при каждом
    # рисовании, включая изменение размера Qt-окна. Постоянные доли полей
    # перекрывали подписи, когда высота встроенного canvas уменьшалась.
    figure.set_layout_engine("constrained", h_pad=.04, w_pad=.04,
                             hspace=.04, wspace=.03)

    n_attempted = int(result.get("n_attempted", samples.shape[0]))
    n_valid = int(result.get("n_valid", samples.shape[0]))
    n_requested = int(result.get("n_requested", n_attempted))
    n_failed = int(result.get("n_failed", n_attempted - n_valid))
    method = {"wls": "свободный МНК", "poisson": "свободный ММП Пуассона"}.get(
        result.get("method"), str(result.get("method", ""))
    )
    generation = result.get("generation", "fitted")
    generator_label = (
        "Poisson(μ), средние из неотрицательного ММП"
        if generation == "fitted" else "Poisson(x), наблюдаемые отсчёты"
    )
    status = f"Успешно: {n_valid}/{n_attempted}; запрошено: {n_requested}; seed: {result.get('seed', '—')}"
    if result.get("cancelled"):
        status += "; остановлено"
    elif n_attempted < n_requested:
        status += "; частичный результат"
    if n_failed:
        status += f"; ошибок: {n_failed}"
    figure.suptitle(
        f"Распределения оценок — {method}\nГенерация: {generator_label}\n{status}",
        fontsize=10,
    )
    if not n_components:
        figure.text(.5, .5, "Нет выбранных компонентов", ha="center", va="center")
        return figure

    n_columns = min(2, n_components)
    n_rows = (n_components + n_columns - 1) // n_columns
    axes = figure.subplots(n_rows, n_columns, squeeze=False).ravel()

    legend_handles: dict[str, Any] = {}
    for column, (axis, name) in enumerate(zip(axes, names)):
        source_values = samples[:, column]
        finite = np.isfinite(source_values)
        values = source_values[finite] * _ACTIVITY_SCALE
        extent: list[float] = []
        if values.size:
            edges = _histogram_edges(values, int(bins))
            axis.hist(values, bins=edges, color="#5588ba", edgecolor="white", linewidth=.5)
            extent.extend((float(edges[0]), float(edges[-1])))
        else:
            axis.text(.5, .48, "Нет успешных реализаций", transform=axis.transAxes,
                      ha="center", va="center", color="#555555", fontsize=10)
            axis.set_ylim(0, 1)

        references = [
            (_component_value(result.get("original_activity"), column),
             _ORIGINAL_LABEL, "#b64e25", "--"),
        ]
        if generation == "fitted":
            references.append((
                _component_value(result.get("generation_activity"), column),
                _GENERATION_LABEL, "#178273", "-",
            ))
        for reference, label, color, linestyle in references:
            if reference is not None:
                line = axis.axvline(reference, color=color, linestyle=linestyle,
                                    linewidth=1.7, label=label)
                legend_handles.setdefault(label, line)
                extent.append(reference)
        if extent:
            low, high = min(extent), max(extent)
            padding = .06 * (high - low) if high > low else max(abs(low) * .1, .001)
            axis.set_xlim(low - padding, high + padding)

        # Среднее и выборочное СКО вычисляются по тем же показанным данным.
        # ddof=1: при одной реализации СКО ещё не определено.
        mean = float(np.mean(source_values[finite])) if values.size else None
        std = float(np.std(source_values[finite], ddof=1)) if values.size > 1 else None
        axis.set_title(f"{name} — Среднее: {_number(mean)}; СКО: {_number(std)}", fontsize=9, pad=7)
        if np.count_nonzero(~finite):
            axis.text(.99, .97, f"Не конечных оценок: {np.count_nonzero(~finite)}",
                      transform=axis.transAxes, ha="right", va="top", fontsize=8)
        # Единицы общие для столбца, поэтому повторяем подпись X только
        # в нижней панели столбца; подпись Y — только в левых панелях.
        # Численные деления остаются видимыми на каждой гистограмме.
        if column + n_columns >= n_components:
            axis.set_xlabel("Объёмная активность, мБк/м³", fontsize=9)
        if column % n_columns == 0:
            axis.set_ylabel("Число реализаций", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    for axis in axes[n_components:]:
        axis.set_visible(False)
    if legend_handles:
        figure.legend(
            list(legend_handles.values()), list(legend_handles),
            # Matplotlib >= 3.7 резервирует здесь отдельное место, чтобы
            # легенда не пересекалась ни с заголовками, ни с единицами.
            loc="outside lower center", ncol=len(legend_handles),
            frameon=False, fontsize=8,
        )
    return figure
