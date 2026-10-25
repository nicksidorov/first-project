"""Парные распределения и эмпирические корреляции оценок Монте-Карло.

Используются только уже сохранённые строки ``activity_samples``: один
порядковый номер строки означает одну и ту же реализацию для всех изотопов.
Решатели и генератор случайных чисел здесь не вызываются. Qt и глобальный
backend Matplotlib не нужны; готовую Figure можно встроить в окно.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .distribution_plot import _component_value, _histogram_edges


def _empirical_correlation(samples: np.ndarray) -> np.ndarray:
    """Пирсон по общим конечным строкам; неопределённые значения — NaN.

    Масштабирование столбца перед центрированием предотвращает переполнение
    суммы квадратов. Оно не меняет корреляцию. У константы дисперсия равна
    нулю, поэтому ставить r=0 для неё было бы неверно.
    """
    count, size = samples.shape
    correlation = np.full((size, size), np.nan)
    if count < 2:
        return correlation
    standardized = np.zeros_like(samples, dtype=float)
    variable = np.zeros(size, dtype=bool)
    for column in range(size):
        values = samples[:, column]
        if np.all(values == values[0]):
            continue
        scaled = values / np.max(np.abs(values))
        centered = scaled - np.mean(scaled)
        norm = np.linalg.norm(centered)
        if norm > 0:
            standardized[:, column] = centered / norm
            variable[column] = True
    valid = np.flatnonzero(variable)
    if valid.size:
        block = standardized[:, valid].T @ standardized[:, valid]
        correlation[np.ix_(valid, valid)] = np.clip(block, -1.0, 1.0)
    return correlation


def _correlation_color(value: float):
    if not np.isfinite(value):
        return "#f3f3f3"
    # Цвет только дополняет знак и число: плюс — голубой, минус — красный.
    color = np.array(to_rgb("#4575b4" if value >= 0 else "#d6604d"))
    return tuple(1 - .30 * abs(value) * (1 - color))


def render_activity_pair_figure(
    result: Mapping[str, Any],
    figure: Figure | None = None,
    bins: int = 30,
) -> Figure:
    """Диагональ — гистограммы; ниже — пары оценок; выше — Pearson r.

    Все панели используют одинаковый набор полных конечных строк. Если
    хотя бы одна компонента строки не конечна, исключается вся строка,
    а число исключений выводится в заголовке. Так корреляции и точки нельзя
    случайно вычислить по разным наборам реализаций.

    Входные активности в Бк/м³ переводятся в мБк/м³ только для рисования.
    Каждому изотопу соответствует общий диапазон активности во всём его
    столбце и в парных графиках его строки; оси Y гистограмм — число
    реализаций, и с диапазонами активностей они не связываются.
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("Число интервалов гистограммы должно быть положительным целым.")
    names = list(result.get("component_names", []))
    size = len(names)
    samples = np.asarray(result.get("activity_samples", []), dtype=float)
    if samples.ndim == 1 and samples.size == 0:
        samples = np.empty((0, size))
    elif samples.ndim == 1 and size == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[1] != size:
        raise ValueError("Столбцы activity_samples должны соответствовать component_names.")
    complete = np.all(np.isfinite(samples), axis=1)
    finite_samples = samples[complete]
    n_shown = int(np.count_nonzero(complete))
    n_excluded = samples.shape[0] - n_shown
    correlations = _empirical_correlation(finite_samples)
    activity = finite_samples * 1000.0

    if figure is None:
        figure = Figure(figsize=(9, 9), facecolor="white")
    else:
        figure.clear()
    figure.set_layout_engine("constrained", h_pad=.04, w_pad=.04,
                             hspace=.015, wspace=.015)
    method = {"wls": "свободный МНК", "poisson": "свободный ММП Пуассона"}.get(
        result.get("method"), str(result.get("method", ""))
    )
    generation = result.get("generation", "fitted")
    generator_label = (
        "Poisson(μ), средние из неотрицательного ММП"
        if generation == "fitted" else "Poisson(x), наблюдаемые отсчёты"
    )
    n_attempted = int(result.get("n_attempted", samples.shape[0]))
    n_valid = int(result.get("n_valid", samples.shape[0]))
    n_requested = int(result.get("n_requested", n_attempted))
    n_failed = int(result.get("n_failed", n_attempted - n_valid))
    status = (
        f"Показано: {n_shown}; успешно: {n_valid}/{n_attempted}; "
        f"запрошено: {n_requested}; seed: {result.get('seed', '—')}"
    )
    notes = []
    if result.get("cancelled"):
        notes.append("Остановлено")
    elif n_attempted < n_requested:
        notes.append("Частичный результат")
    if n_failed:
        notes.append(f"ошибок: {n_failed}")
    if n_excluded:
        notes.append(f"исключено строк с не конечными оценками: {n_excluded}")
    title = f"Парные распределения — {method}\nГенерация: {generator_label}\n{status}"
    if notes:
        title += "\n" + "; ".join(notes)
    figure.suptitle(title, fontsize=10)
    if not size:
        figure.text(.5, .5, "Нет выбранных компонентов", ha="center", va="center")
        return figure

    axes = figure.subplots(size, size, squeeze=False)
    references: list[list[tuple[float | None, str, str, str]]] = []
    ranges: list[tuple[float, float]] = []
    histogram_edges: list[np.ndarray | None] = []
    for column in range(size):
        lines = [(_component_value(result.get("original_activity"), column),
                  "Исходная свободная оценка", "#b64e25", "--")]
        if generation == "fitted":
            lines.append((_component_value(result.get("generation_activity"), column),
                          "Задано при генерации", "#178273", "-"))
        references.append(lines)
        edges = _histogram_edges(activity[:, column], int(bins)) if n_shown else None
        histogram_edges.append(edges)
        extent = [float(edges[0]), float(edges[-1])] if edges is not None else []
        extent.extend(value for value, *_ in lines if value is not None)
        low, high = (min(extent), max(extent)) if extent else (-.001, .001)
        padding = .06 * (high - low) if high > low else max(abs(low) * .1, .001)
        ranges.append((low - padding, high + padding))

    legend_handles: dict[str, Any] = {}
    for row in range(size):
        for column in range(size):
            axis = axes[row, column]
            axis.set_xlim(ranges[column])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(labelsize=7, labelbottom=row == size - 1,
                             labelleft=column == 0 or row == column)
            if row == 0:
                axis.set_title(str(names[column]), fontsize=10, pad=7)
            if row == size - 1:
                axis.set_xlabel(f"{names[column]}\nмБк/м³", fontsize=9)
            if column == 0 and row > 0:
                axis.set_ylabel(f"{names[row]}\nмБк/м³", fontsize=9)

            if row == column:
                if n_shown:
                    axis.hist(activity[:, column], bins=histogram_edges[column],
                              color="#5588ba", edgecolor="white", linewidth=.4)
                else:
                    axis.set_ylim(0, 1)
                    axis.text(.5, .43, "Нет реализаций", transform=axis.transAxes,
                              ha="center", va="center", fontsize=8, color="#555555")
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
                axis.text(.03, .97, "Число реализаций", transform=axis.transAxes,
                          ha="left", va="top", fontsize=7, color="#444444",
                          bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1))
                for value, label, color, linestyle in references[column]:
                    if value is not None:
                        line = axis.axvline(value, color=color, linestyle=linestyle,
                                            linewidth=1.3, label=label)
                        legend_handles.setdefault(label, line)
            elif row > column:
                # Не сортируем и не перемешиваем столбцы: обе координаты
                # точки принадлежат одной исходной реализации Монте-Карло.
                axis.scatter(activity[:, column], activity[:, row],
                             s=12 if n_shown < 100 else 7,
                             alpha=.65 if n_shown < 100 else .30,
                             color="#477da8", edgecolors="none", rasterized=True)
                axis.set_ylim(ranges[row])
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                value = correlations[row, column]
                axis.set_facecolor(_correlation_color(value))
                label = f"r = {value:+.3f}" if np.isfinite(value) else "r = —"
                reason = "Пирсон, по реализациям" if np.isfinite(value) else (
                    "Недостаточно реализаций" if n_shown < 2 else "Нет разброса компоненты"
                )
                axis.text(.5, .55, label, transform=axis.transAxes,
                          ha="center", va="center", fontsize=14, color="#253342")
                axis.text(.5, .38, reason, transform=axis.transAxes,
                          ha="center", va="center", fontsize=7, color="#555555")
                axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            axis.grid(axis="y" if row == column else "both", alpha=.15)
            if row < column:
                axis.grid(False)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)

    if legend_handles:
        figure.legend(list(legend_handles.values()), list(legend_handles),
                      loc="outside lower center", ncol=len(legend_handles),
                      frameon=False, fontsize=8)
    return figure
