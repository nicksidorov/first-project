"""Render the calibrated clean-background study; no fitting or inference here.

Input: summary.json from the independent analysis and the complete manifest.json.
Activities in the summary are mBq/m3; probabilities are fractions, not percent.
Each curve has dataset/target/isotope/nominal_MDC_mBq_m3 and two methods.
Each method provides calibration, independent validation (standard/calibrated),
the standard/calibrated Ld95, and formal power points. The conditional interval
holds c fixed; the optional total interval also includes its calibration error.
Missing intervals are displayed as unavailable, never replaced by other ones.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FAMILIES = ("wls", "poisson")
LABELS = {"wls": "МНК", "poisson": "ММП"}
DATES = {"may22": "22 мая", "may26": "26 мая"}
ISOTOPES = {1: "Xe135", 2: "Xe133", 3: "Xe131m", 4: "Xe133m"}
COLORS = {"wls": "#087eaa", "poisson": "#d9650b"}
MARKERS = {"wls": "o", "poisson": "s"}
ARCHIVE = "XeMat_calibrated_nominal_MDC_May.zip"


def finite(value):
    return value is not None and math.isfinite(float(value))


def number(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "—"


def integer(value):
    return f"{int(value):,}".replace(",", "\u202f")


def interval_text(interval, scale=1.0, digits=4):
    if interval is None or len(interval) != 2:
        return "не оценён"
    low, high = interval
    if finite(low) and finite(high):
        return f"{number(low * scale, digits)}–{number(high * scale, digits)}"
    if finite(low):
        return f"от {number(low * scale, digits)}, верхняя граница не определена"
    if finite(high):
        return f"до {number(high * scale, digits)}, нижняя граница не определена"
    return "не оценён"


def rate_text(record):
    if not finite(record.get("rate")):
        return "не рассчитано"
    return f"{record['rate'] * 100:.2f}% ({interval_text(record.get('ci95'), 100, 2)}%)"


def curves(summary):
    return sorted(summary["curves"], key=lambda c: (list(DATES).index(c["dataset"]), c["target"]))


def has_total_intervals(summary):
    return any(
        m.get("total_ci95_mBq_m3") is not None
        and len(m["total_ci95_mBq_m3"]) == 2
        and all(finite(v) for v in m["total_ci95_mBq_m3"])
        for c in curves(summary) for m in c["methods"].values()
    )


def opening_findings(summary):
    """Compute the numerical opening from the final study, never placeholders."""
    ordered = curves(summary)
    rates = [m["validation"]["calibrated"]["rate"]
             for c in ordered for m in c["methods"].values()]
    xe133 = [c for c in ordered if c["target"] == 2]
    examples = []
    for curve in xe133:
        method = curve["methods"]["wls"]
        validation = method["validation"]
        delta = 100 * (method["calibrated_ld_mBq_m3"] / method["standard_ld_mBq_m3"] - 1)
        examples.append(f"{DATES[curve['dataset']]}: {validation['standard']['rate'] * 100:.2f}% → "
                        f"{validation['calibrated']['rate'] * 100:.2f}%, Ld95 {delta:+.1f}%")
    method_comparison = []
    for curve in xe133:
        wls = curve["methods"]["wls"]["calibrated_ld_mBq_m3"]
        poisson = curve["methods"]["poisson"]["calibrated_ld_mBq_m3"]
        method_comparison.append(f"{DATES[curve['dataset']]}: {100 * (1 - poisson / wls):.1f}%")
    other_changes = [100 * (m["calibrated_ld_mBq_m3"] / m["standard_ld_mBq_m3"] - 1)
                     for c in ordered for f, m in c["methods"].items()
                     if not (c["target"] == 2 and f == "wls")]
    return [
        f"**После калибровки независимая проверка дала {min(rates) * 100:.2f}–{max(rates) * 100:.2f}% "
        "ложных обнаружений.** Уровень приблизился к целевым 5%; конечная калибровочная серия не задаёт его точно.", "",
        "Самое заметное изменение — Xe133 при МНК (" + "; ".join(examples) + "). "
        f"В остальных сочетаниях изотопа, метода и комплекта изменение Ld95 составило от {min(other_changes):+.1f}% "
        f"до {max(other_changes):+.1f}%. Эти результаты относятся к заданному чистому фону и майским условиям "
        "измерения; для других смесей они не устанавливают уровень ошибки автоматически.", "",
        "Для Xe133 исходное правило давало меньший Ld95 у МНК при большей частоте ложных обнаружений. "
        "После отдельной калибровки обоих методов в этих двух сценариях меньший Ld95 получен у ММП (разница "
        + "; ".join(method_comparison) + "). Поэтому сравнение одного Ld95 без проверки уровня ошибки "
        "могло привести к обратному выводу о чувствительности. Неопределённость парной разности приведена ниже.", "",
    ]


def require_schema(summary):
    expected = {(d, i) for d in DATES for i in ISOTOPES}
    found = {(c["dataset"], c["target"]) for c in summary["curves"]}
    if found != expected or len(summary["curves"]) != len(expected):
        raise ValueError("Report requires exactly four isotopes for each May dataset")
    for curve in curves(summary):
        if curve["isotope"] != ISOTOPES[curve["target"]]:
            raise ValueError("Isotope name and target index disagree")
        for family in FAMILIES:
            method = curve["methods"][family]
            for key in ("calibration", "validation", "formal_points", "standard_ld_mBq_m3", "calibrated_ld_mBq_m3"):
                if key not in method:
                    raise ValueError(f"Missing method field: {key}")
            if method["validation"] is None:
                raise ValueError("Run the final analysis with independent validation before rendering the report")
            if not finite(method["calibration"].get("c")) or method["calibration"]["c"] <= 0:
                raise ValueError("Calibration multiplier must be positive and finite")


def interval_line(ax, y, interval, denominator, color, width=2, alpha=1.0):
    if interval is None or not all(finite(v) for v in interval):
        return
    ax.hlines(y, interval[0] / denominator, interval[1] / denominator,
              color=color, linewidth=width, alpha=alpha)


def save_figure(fig, name):
    fig.savefig(ROOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"{name}.png", dpi=170, bbox_inches="tight")


def plots(summary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "plot-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "plot-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    method_handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f], label=LABELS[f]) for f in FAMILIES]
    stage_handles = [
        Line2D([], [], color=".35", marker="o", linestyle="none", markerfacecolor="white", label="Исходный Lc"),
        Line2D([], [], color=".35", marker="o", linestyle="none", label="После калибровки: c · Lc"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5), sharex=True, sharey=True)
    all_limits = [5.0]
    for dataset, ax in zip(DATES, axes):
        for curve in (c for c in curves(summary) if c["dataset"] == dataset):
            for family, method_offset in (("wls", -.16), ("poisson", .16)):
                method = curve["methods"][family]
                for stage, stage_offset in (("standard", -.045), ("calibrated", .045)):
                    record = method["validation"][stage]
                    if not finite(record.get("rate")):
                        continue
                    y = curve["target"] - 1 + method_offset + stage_offset
                    ci = record.get("ci95")
                    if ci is not None and all(finite(v) for v in ci):
                        all_limits.extend(v * 100 for v in ci)
                        ax.hlines(y, ci[0] * 100, ci[1] * 100, color=COLORS[family],
                                  lw=1.4 if stage == "standard" else 2.4,
                                  alpha=.45 if stage == "standard" else 1.0)
                    ax.plot(record["rate"] * 100, y, marker=MARKERS[family], ls="none",
                            ms=6, markeredgecolor=COLORS[family],
                            markerfacecolor="white" if stage == "standard" else COLORS[family])
        ax.axvline(5, color=".3", ls="--", lw=1.2)
        ax.set_title(DATES[dataset], fontsize=14)
        ax.set_yticks(range(4), list(ISOTOPES.values()))
        ax.set_xlabel("Ложные обнаружения на независимой проверке, %")
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    span = max(all_limits) - min(all_limits)
    axes[0].set_xlim(max(0, min(all_limits) - .12 * span), max(all_limits) + .12 * span)
    fig.suptitle("Проверка уровня 5% на новых пустых пробах", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .915, "Исходное и откалиброванное правила проверяются на одних и тех же независимых наблюдениях", ha="center", fontsize=10)
    fig.legend(handles=method_handles + stage_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .045))
    fig.text(.5, .018, "Полосы — индивидуальные точные биномиальные 95% интервалы; пунктир — целевые 5%.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .90])
    save_figure(fig, "calibrated_alpha")
    plt.close(fig)

    total_present = has_total_intervals(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8), sharex=True, sharey=True)
    for dataset, ax in zip(DATES, axes):
        for curve in (c for c in curves(summary) if c["dataset"] == dataset):
            denominator = curve["nominal_MDC_mBq_m3"]
            for family, offset in (("wls", -.13), ("poisson", .13)):
                method = curve["methods"][family]
                y = curve["target"] - 1 + offset
                original = method["standard_ld_mBq_m3"]
                calibrated = method["calibrated_ld_mBq_m3"]
                if finite(original) and finite(calibrated):
                    ax.plot([original / denominator, calibrated / denominator], [y, y],
                            color=COLORS[family], ls=":", lw=1.1, alpha=.5)
                interval_line(ax, y, method.get("total_ci95_mBq_m3"), denominator, COLORS[family], width=1.4, alpha=.8)
                interval_line(ax, y, method.get("conditional_ci95_mBq_m3"), denominator, COLORS[family], width=4.0)
                if finite(original):
                    ax.plot(original / denominator, y, marker=MARKERS[family], ls="none", ms=7,
                            markerfacecolor="white", markeredgecolor=COLORS[family], zorder=4)
                if finite(calibrated):
                    ax.plot(calibrated / denominator, y, marker=MARKERS[family], ls="none", ms=6,
                            color=COLORS[family], zorder=5)
        ax.axvline(1, color=".3", ls="--", lw=1.2)
        ax.set_title(DATES[dataset], fontsize=14)
        ax.set_yticks(range(4), list(ISOTOPES.values()))
        ax.set_xlabel("Ld95 по мощности / формульный MDC")
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Как калибровка Lc к 5% меняет предел обнаружения", fontsize=17, fontweight="bold", y=.98)
    interval_handles = [Line2D([], [], color=".35", lw=4, label="95% ДИ при фиксированном c")]
    if total_present:
        interval_handles.append(Line2D([], [], color=".35", lw=1.4, label="95% ДИ с учётом неопределённости c"))
    fig.legend(handles=method_handles + stage_handles + interval_handles,
               loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, .045))
    fig.text(.5, .018, "Интервалы Монте-Карло приближённые; погрешность интерполяции и систематические ошибки не включены.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .19, 1, .93])
    save_figure(fig, "calibrated_ld")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex=True, sharey=True)
    y_values = [95.0]
    for curve in curves(summary):
        ax = axes[list(DATES).index(curve["dataset"]), curve["target"] - 1]
        for family in FAMILIES:
            method = curve["methods"][family]
            points = sorted(method["formal_points"], key=lambda p: p["factor"])
            points = [p for p in points if finite(p["calibrated"].get("rate"))]
            factors = [p["factor"] for p in points]
            probabilities = [p["calibrated"]["rate"] * 100 for p in points]
            y_values.extend(probabilities)
            ax.plot(factors, probabilities, color=COLORS[family], marker=MARKERS[family], ms=4, lw=1.5)
            for factor, probability, point in zip(factors, probabilities, points):
                ci = point["calibrated"].get("ci95")
                if ci is not None and all(finite(v) for v in ci):
                    y_values.append(ci[0] * 100)
                    ax.errorbar(factor, probability,
                                yerr=np.array([[probability - ci[0] * 100], [ci[1] * 100 - probability]]),
                                fmt="none", color=COLORS[family], lw=1, capsize=3)
            root = method["calibrated_ld_mBq_m3"]
            if finite(root):
                ax.plot(root / curve["nominal_MDC_mBq_m3"], 95, marker="|", ms=16, mew=2, color=COLORS[family])
        ax.axhline(95, color=".3", ls=":", lw=1.3)
        ax.axvline(1, color=".4", ls="--", lw=1)
        ax.set_title(f"{DATES[curve['dataset']]} — {curve['isotope']}", fontsize=12)
        ax.text(.98, .04, f"Формула: {curve['nominal_MDC_mBq_m3']:.3f} мБк/м³", transform=ax.transAxes, ha="right", fontsize=9)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        if curve["target"] == 1:
            ax.set_ylabel("Вероятность обнаружения, %")
        if curve["dataset"] == "may26":
            ax.set_xlabel("Истинная A / формульный MDC")
    axes[0, 0].set_ylim(max(0, min(85, min(y_values) - 2)), 100.5)
    fig.suptitle("Мощность правила после калибровки Lc к 5%", fontsize=17, fontweight="bold", y=.985)
    fig.text(.5, .94, "Множитель c фиксирован; Lc, активности, память и фон оцениваются заново в каждой реализации", ha="center", fontsize=10.5)
    fig.legend(handles=method_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .025))
    fig.text(.5, .008, "Точки — Монте-Карло при заданной истинной активности; полосы — индивидуальные биномиальные 95% интервалы.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .085, 1, .91], h_pad=2)
    save_figure(fig, "calibrated_power")
    plt.close(fig)


def text_report(summary, manifest):
    stages = summary.get("n_new_by_stage", {})
    total_new = sum(stages.values())
    calibration_n = stages.get("calibration", 0)
    validation_n = stages.get("validation", 0)
    extra_n = stages.get("signal_extension", 0)
    reused = summary.get("n_reused_signal", 0)
    lines = [
        "# Номинальный MDC после калибровки Lc к 5%", "",
        *opening_findings(summary),
        "Цель — проверить предел обнаружения на чистом детекторном фоне при целевом уровне ложных обнаружений 5% для одного проверяемого изотопа. Это не вероятность хотя бы одного ложного обнаружения среди всех четырёх изотопов. Для каждого изотопа, метода и майского комплекта множитель порога определён на обучающей серии пустых проб, затем проверен на независимой нулевой серии. После фиксации множителя найден Ld95 — активность, при которой вероятность обнаружения достигает примерно 95%.", "",
        "При генерации нет пробы, газовой памяти и радона. Для проверки мощности вводится только один свежий изотоп Xe. Все 21 наблюдение SAMPLE/GASBK/DETBK разыгрываются по Пуассону; все 17 параметров полной модели продолжают оцениваться. Истинные средние детекторного фона заданы по прежнему DETBK 221. Систематические ошибки не моделируются.", "",
        f"**Новые реализации: {integer(total_new)}** — {integer(calibration_n)} для калибровки, {integer(validation_n)} для независимой проверки нуля и {integer(extra_n)} для дополнительных точек мощности. Повторно использовано {integer(reused)} независимых от калибровки сигнальных реализаций предыдущего исследования. Поисковые точки прежнего исследования в формальную оценку Ld не входят.", "",
        "## Что изменилось в правиле обнаружения", "",
        r"Исходное правило: $\widehat A>L_c$. После калибровки: $\widehat A>cL_c$. На обучающих пустых пробах строится распределение $R=\widehat A/L_c$; множитель $c$ выбирается как порядковая статистика ранга $k=\lceil(n+1)0.95\rceil$. При 20000 наблюдений это 19001-е значение по возрастанию. Равенство порогу не считается обнаружением. Такое правило задаёт примерно 5% в среднем по повторным калибровочным сериям; уровень конкретного зафиксированного c проверяется отдельно. Совпадения значений могут сделать правило консервативнее.", "",
        "Множитель c фиксируется до независимой проверки и не меняется при введении сигнала. Сам Lc остаётся случайным: в каждой реализации он пересчитывается прежней процедурой с удалением целевого свежего изотопа из неотрицательной подгонки. Решение об обнаружении использует свободную оценку активности. МНК и ММП обрабатывают одни и те же наблюдения.", "",
        "Поправка относится к конкретному чистому фону и условиям измерения. Это не универсальная поправка для произвольных смесей, памяти или других фоновых уровней.", "",
        "## Независимая проверка уровня ложных обнаружений", "",
        "Оба правила проверены на одних и тех же новых пустых пробах. Интервалы — индивидуальные точные биномиальные 95% ДИ. Наблюдаемая частота имеет погрешность Монте-Карло: калибровка не гарантирует ровно 5.000% в конечной проверочной серии.", "",
        "| Комплект | Изотоп | Метод | Множитель c | Исходный Lc: частота и 95% ДИ | После калибровки: частота и 95% ДИ |",
        "|---|---|---|---:|---|---|",
    ]
    for curve in curves(summary):
        for family in FAMILIES:
            method = curve["methods"][family]
            lines.append(f"| {DATES[curve['dataset']]} | {curve['isotope']} | {LABELS[family]} | {method['calibration']['c']:.4f} | {rate_text(method['validation']['standard'])} | {rate_text(method['validation']['calibrated'])} |")
    checked = [(curve, family, curve["methods"][family]["validation"]["calibrated"])
               for curve in curves(summary) for family in FAMILIES]
    outside = [(curve, family, record) for curve, family, record in checked
               if record.get("ci95") is not None and not record["ci95"][0] <= .05 <= record["ci95"][1]]
    if outside:
        lines.extend(["", f"У {len(checked) - len(outside)} из {len(checked)} отдельных 95% интервалов целевые 5% лежат внутри. "
            "Случаи вне интервала приведены с дополнительными знаками, чтобы округление не скрыло границу: "
            + "; ".join(f"{DATES[c['dataset']]}, {LABELS[f]}, {c['isotope']}: {r['rate'] * 100:.2f}%, "
                         f"95% ДИ {interval_text(r['ci95'], 100, 3)}%" for c, f, r in outside)
            + ". Это индивидуальные интервалы, без одновременной гарантии для всех 16 проверок. "
            "По этим проверочным результатам множители не перенастраивались."])
    lines.extend(["", "![Независимая проверка уровня 5%](calibrated_alpha.png)", "",
        "## Формула, исходная процедура и откалиброванный порог", "",
        "Все активности — **мБк/м³**. Формульный номинальный MDC рассчитан при заданной фоновой смеси по полной информации Фишера. Ld95 исходной и откалиброванной процедур найден из мощности обработки случайных наблюдений. Для каждого метода оба правила применяются к одной сигнальной серии; изменение Ld связано с изменением критического порога.", "",
        "| Комплект | Изотоп | Метод | Формульный MDC | Ld95 с исходным Lc | Ld95 после калибровки | Изменение Ld95 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for curve in curves(summary):
        for family in FAMILIES:
            method = curve["methods"][family]
            old = method["standard_ld_mBq_m3"]
            new = method["calibrated_ld_mBq_m3"]
            change = f"{(new / old - 1) * 100:+.1f}%" if finite(old) and finite(new) and old != 0 else "—"
            lines.append(f"| {DATES[curve['dataset']]} | {curve['isotope']} | {LABELS[family]} | {number(curve['nominal_MDC_mBq_m3'])} | {number(old)} | {number(new)} | {change} |")
    lines.extend(["", "![Изменение Ld95](calibrated_ld.png)", "",
        "Меньший Ld при одинаковом подтверждённом уровне ложных обнаружений означает более высокую чувствительность в выбранных условиях. При сравнении методов нужно учитывать неопределённость и Ld, и независимой оценки ложноположительной частоты.", "",
        "## Неопределённость калибровки и Ld95", "",
        r"Два вида интервалов отвечают на разные вопросы. Условный интервал оценивает $Ld95(\widehat c)$ для фактически зафиксированного множителя и учитывает погрешность сигнальной серии Монте-Карло. Интервал с учётом неопределённости калибровки дополнительно разыгрывает обучающую серию: он оценивает неопределённость $Ld95(c_0)$, где $c_0$ — истинный 95-й квантиль нулевого распределения статистики. Это интервалы оценки предела обнаружения, а не распределения индивидуальных активностей или MDC будущих проб. Систематические ошибки и неопределённость принятого фонового среднего в них не входят.", "",
        "Интерполяция кривой мощности между рассчитанными точками приближённая. Bootstrap-интервалы условны относительно итоговой сетки; ошибка формы кривой между узлами не включена. Отсутствующие интервалы обозначены явно и не заменяются другим видом интервала. Результаты независимой проверки приводятся без повторной настройки c по проверочным данным, даже если отдельный интервал частоты не включает 5%.", "",
        "| Комплект | Изотоп | Метод | c: 95% ДИ | Ld95 | 95% ДИ при фиксированном c | 95% ДИ с учётом неопределённости c |",
        "|---|---|---|---|---:|---|---|",
    ])
    for curve in curves(summary):
        for family in FAMILIES:
            method = curve["methods"][family]
            lines.append(f"| {DATES[curve['dataset']]} | {curve['isotope']} | {LABELS[family]} | {interval_text(method['calibration'].get('ci95_c'))} | {number(method['calibrated_ld_mBq_m3'])} | {interval_text(method.get('conditional_ci95_mBq_m3'))} | {interval_text(method.get('total_ci95_mBq_m3'))} |")
    grid_rows = [(curve, family, curve["methods"][family]) for curve in curves(summary) for family in FAMILIES
                 if any(curve["methods"][family].get(key) is not None for key in
                        ("conditional_grid_bound95_mBq_m3", "total_grid_bound95_mBq_m3"))]
    if grid_rows:
        lines.extend(["", "### Консервативные сеточные границы", "",
            "Эти границы получены из одновременных биномиальных интервалов на рассчитанных активностях при предположении монотонной истинной мощности. Для варианта с неопределённостью c дополнительно использован интервал коэффициента c. Сеточные границы консервативны и могут оставаться открытыми, если сетка недостаточно широка. Бюджеты вероятностей ошибок заданы в протоколе.", "",
            "| Комплект | Изотоп | Метод | Границы 95% при фиксированном c | Границы 95% с учётом неопределённости c |",
            "|---|---|---|---|---|",
        ])
        for curve, family, method in grid_rows:
            lines.append(f"| {DATES[curve['dataset']]} | {curve['isotope']} | {LABELS[family]} | {interval_text(method.get('conditional_grid_bound95_mBq_m3'))} | {interval_text(method.get('total_grid_bound95_mBq_m3'))} |")
    paired = [c for c in curves(summary) if c.get("paired_ld_difference") is not None]
    if paired:
        lines.extend(["", "### Парное сравнение методов", "",
            "Разность Δ = Ld95(МНК) − Ld95(ММП), мБк/м³. Положительная разность означает больший предел обнаружения у МНК. Парные интервалы учитывают, что оба метода анализируют одинаковые реализации.", "",
            "| Комплект | Изотоп | Δ | 95% ДИ при фиксированных c | 95% ДИ с учётом неопределённости c |",
            "|---|---|---:|---|---|",
        ])
        for curve in paired:
            difference = curve["paired_ld_difference"]
            lines.append(f"| {DATES[curve['dataset']]} | {curve['isotope']} | {number(difference.get('point'))} | {interval_text(difference.get('conditional_ci95'))} | {interval_text(difference.get('total_ci95'))} |")
    lines.extend(["", "## Кривые мощности после калибровки", "",
        "Вертикальная отметка на уровне 95% показывает интерполированную оценку Ld95. Горизонтальная ось нормирована на прежний формульный MDC. Интервалы точек мощности условны относительно зафиксированных множителей c.", "",
        "![Кривые мощности](calibrated_power.png)", "",
        "## Сохранённые условия измерения", "",
        "| Параметр | 22 мая | 26 мая |", "|---|---:|---:|",
    ])
    datasets = manifest.get("datasets", {})
    if all(d in datasets and "metadata" in datasets[d] for d in DATES):
        for label, key in (
            ("SAMPLE: живое время, с", "t_sl"), ("SAMPLE: реальное время, с", "t_sr"),
            ("GASBK: живое время, с", "t_gl"), ("GASBK: реальное время, с", "t_gr"),
            ("DETBK: живое время, с", "t_bl"), ("Сбор, с", "t_a"),
            ("Задержка после сбора, с", "t_p"), ("Разность начал SAMPLE/GASBK, с", "tau"),
        ):
            values = [datasets[d]["metadata"]["time_parameters"][key] for d in DATES]
            lines.append(f"| {label} | {values[0]} | {values[1]} |")
        for label, getter in (
            ("Объём Xe, см³", lambda m: m["xenon_volume_cm3"]),
            ("Эффективный объём воздуха, м³", lambda m: m["effective_air_volume_m3"]),
            ("Поле air_volume исходного SAMPLE, м³", lambda m: m["sample_temporal"]["air_volume"]),
        ):
            values = [getter(datasets[d]["metadata"]) for d in DATES]
            lines.append(f"| {label} | {values[0]:.6f} | {values[1]:.6f} |")
    lines.extend(["", "Сохранена конвенция объёма существующего ядра: объёмная активность рассчитывается через эффективный объём воздуха, полученный из объёма Xe. Использованы фиксированные ROI и эффективности, калибровки детектора 221 и прежняя модель радона. Полные матрицы и метаданные сохранены в manifest.json.", "",
        "## Пропуски и воспроизводимость", "",
        f"Зарегистрированных ошибок подгонки/расчёта: {len(summary.get('errors', []))}. Отсутствующие решения не заменяются новыми реализациями; числа доступных решений и пропусков приведены в summary.json.", "",
        "Калибровка, независимая проверка нуля и сигнальные серии разделены. При добавлении сигнальных точек Ld95 исходного правила пересчитывается на той же итоговой сетке, что и Ld95 откалиброванного правила. Исторические значения предыдущего исследования, если они отличаются, сохранены отдельно в summary.json. Для генерации принят фиксированный фон по исходному DETBK, а при каждой обработке его параметры оцениваются заново совместно с SAMPLE и GASBK. Поэтому статистическая погрешность нового измерения DETBK учтена. Неопределённость выбранного фонового среднего и калибровок не моделировалась.", "",
        "## Файлы", "",
        "- [Проверка уровня 5% — PDF](calibrated_alpha.pdf)",
        "- [Изменение Ld95 — PDF](calibrated_ld.pdf)",
        "- [Кривые мощности — PDF](calibrated_power.pdf)",
        "- [Таблица CSV](results.csv)",
        "- [Полная численная сводка](summary.json)",
        "- [Протокол](protocol_ru.md)",
        f"- [Архив с данными и скриптами]({ARCHIVE})", "",
    ])
    report = "\n".join(lines)
    (ROOT / "report_ru.md").write_text(report, encoding="utf-8")
    (ROOT / "report_ru.txt").write_text(report, encoding="utf-8")


def export_csv(summary):
    fields = ["dataset", "isotope", "method", "nominal_MDC_mBq_m3", "c", "c_ci95_low", "c_ci95_high",
              "calibration_n", "calibration_rank", "validation_n", "validation_standard_rate", "validation_standard_ci95_low",
              "validation_standard_ci95_high", "validation_calibrated_rate", "validation_calibrated_ci95_low",
              "validation_calibrated_ci95_high", "standard_Ld95_mBq_m3", "calibrated_Ld95_mBq_m3",
              "conditional_Ld95_ci95_low", "conditional_Ld95_ci95_high", "total_Ld95_ci95_low", "total_Ld95_ci95_high"]
    with (ROOT / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for curve in curves(summary):
            for family in FAMILIES:
                method = curve["methods"][family]
                calibration = method["calibration"]
                standard = method["validation"]["standard"]
                calibrated = method["validation"]["calibrated"]
                writer.writerow([curve["dataset"], curve["isotope"], family, curve["nominal_MDC_mBq_m3"], calibration["c"],
                    *(calibration.get("ci95_c") or [None, None]), calibration.get("n"), calibration.get("rank", calibration.get("k_order")),
                    calibrated.get("n"), standard.get("rate"), *(standard.get("ci95") or [None, None]),
                    calibrated.get("rate"), *(calibrated.get("ci95") or [None, None]), method["standard_ld_mBq_m3"],
                    method["calibrated_ld_mBq_m3"], *(method.get("conditional_ci95_mBq_m3") or [None, None]),
                    *(method.get("total_ci95_mBq_m3") or [None, None])])


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    require_schema(summary)
    plots(summary)
    text_report(summary, manifest)
    export_csv(summary)
    print("Rendered calibrated study:", len(summary["curves"]), "curves,", len(summary.get("errors", [])), "recorded errors")


if __name__ == "__main__":
    main()
