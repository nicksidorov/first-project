"""Plot the saved A/B false-detection comparison without fitting or simulation.

The CSV must contain 16 rows: two variants, two estimation methods, and four
xenon isotopes. Frequencies and exact intervals are read as proportions and
displayed as percentages. No thresholds or statistical intervals are recomputed.

Example:
    python plot_comparison.py --input summary.csv --output-dir figures
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

_CACHE = Path(tempfile.gettempdir()) / "xemat-response-comparison-cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
ISOTOPES = ("Xe135", "Xe133", "Xe131m", "Xe133m")
LABELS = (r"$^{135}\mathrm{Xe}$", r"$^{133}\mathrm{Xe}$",
          r"$^{131\mathrm{m}}\mathrm{Xe}$", r"$^{133\mathrm{m}}\mathrm{Xe}$")
METHODS = ("LS", "Poisson ML")
VARIANTS = {
    "A": {"color": "#1f77b4", "marker": "o", "offset": -.105,
          "label": "A: original responses"},
    "B": {"color": "#ff7f0e", "marker": "s", "offset": .105,
          "label": "B: equal base responses"},
}


def read_summary(path: Path) -> dict:
    """Validate the 16 saved rows and their available-decision denominators."""
    required = {"variant", "method", "isotope", "n_generated", "n_valid",
                "n_false", "frequency", "ci95_low", "ci95_high", "n_fail"}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        sources = list(reader)
    expected = {(variant, method, isotope) for variant in VARIANTS
                for method in METHODS for isotope in ISOTOPES}
    rows = {}
    for source in sources:
        key = (source["variant"], source["method"], source["isotope"])
        if key not in expected or key in rows:
            raise ValueError(f"Unexpected or duplicate summary row: {key}")
        row = dict(source)
        for name in ("n_generated", "n_valid", "n_false", "n_fail"):
            row[name] = int(source[name])
        n, valid, count, failed = (row[name] for name in
                                  ("n_generated", "n_valid", "n_false", "n_fail"))
        if n <= 0 or not 0 <= count <= valid <= n or failed != n - valid:
            raise ValueError(f"Inconsistent counts in {key}")
        for name in ("frequency", "ci95_low", "ci95_high"):
            row[name] = float(source[name]) if source[name].strip() else float("nan")
        rate, low, high = (row[name] for name in ("frequency", "ci95_low", "ci95_high"))
        if valid:
            if not all(math.isfinite(value) for value in (rate, low, high)):
                raise ValueError(f"Non-finite reported frequency or interval in {key}")
            if not 0 <= low <= rate <= high <= 1:
                raise ValueError(f"Invalid reported interval in {key}")
            if not math.isclose(rate, count / valid, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Frequency is not n_false / n_valid in {key}")
        elif not all(math.isnan(value) for value in (rate, low, high)):
            raise ValueError(f"No available decisions but finite statistics in {key}")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError(f"Missing rows: {sorted(expected - set(rows))}")
    if len({row["n_generated"] for row in rows.values()}) != 1:
        raise ValueError("The comparison must use the same planned observations for all rows")
    return rows


def available_note(rows: dict, method: str) -> str:
    parts = []
    for variant in VARIANTS:
        values = [rows[variant, method, isotope]["n_valid"] for isotope in ISOTOPES]
        amount = (f"{values[0]:,}" if len(set(values)) == 1
                  else f"{min(values):,}–{max(values):,}")
        parts.append(f"{variant}: {amount}")
    return "Available decisions per isotope — " + "; ".join(parts)


def make_figure(rows: dict, *, font_size: float = 12, width: float = 12.4,
                height: float = 5.8,
                title: str = "Effect of equal xenon response shapes in a strong-memory scenario"):
    """Build two method panels with shared limits and saved exact intervals."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": font_size, "mathtext.fontset": "cm", "text.usetex": False,
        "axes.labelsize": font_size, "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size + 1, "ytick.labelsize": font_size - .5,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.245, top=.765, wspace=.12)
    plotted_bounds = [5.0]
    x = np.arange(len(ISOTOPES), dtype=float)
    for panel, (ax, method) in enumerate(zip(axes, METHODS)):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e7e7e7", linewidth=.7)
        ax.axhline(5.0, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
        missing = []
        for variant, style in VARIANTS.items():
            selected = [rows[variant, method, isotope] for isotope in ISOTOPES]
            valid = np.array([row["n_valid"] > 0 for row in selected], dtype=bool)
            rates = np.array([row["frequency"] * 100 for row in selected])
            lows = np.array([row["ci95_low"] * 100 for row in selected])
            highs = np.array([row["ci95_high"] * 100 for row in selected])
            plotted_bounds.extend(lows[valid].tolist())
            plotted_bounds.extend(highs[valid].tolist())
            if np.any(valid):
                ax.errorbar(x[valid] + style["offset"], rates[valid],
                    yerr=np.stack((rates[valid]-lows[valid], highs[valid]-rates[valid])),
                    fmt=style["marker"], linestyle="none", color=style["color"],
                    markerfacecolor=style["color"], markeredgecolor="white",
                    markeredgewidth=.55, markersize=6.4, elinewidth=1.25,
                    capsize=3.5, capthick=1.0, zorder=4, label=variant)
            missing.extend(f"{variant}: {ISOTOPES[j]}" for j in np.flatnonzero(~valid))
        ax.set_xlim(-.5, 3.5)
        ax.set_xticks(x, LABELS)
        ax.set_xlabel("Isotope", labelpad=7)
        ax.set_title(f"({chr(ord('a')+panel)}) {method}", loc="left", pad=10)
        if missing:
            ax.text(.02, .98, "No decisions: " + "; ".join(missing),
                transform=ax.transAxes, va="top", fontsize=font_size-2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    lower, upper = min(plotted_bounds), max(plotted_bounds)
    padding = max(.25, .09 * (upper-lower))
    axes[0].set_ylim(max(0, lower-padding), min(100, upper+padding))
    axes[0].set_ylabel("False-detection frequency (%)", labelpad=9)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    handles = [Line2D([], [], color=style["color"], marker=style["marker"],
        linestyle="none", markersize=6.4, markeredgecolor="white",
        markeredgewidth=.55, label=style["label"]) for style in VARIANTS.values()]
    handles.append(Line2D([], [], color="#555555", linestyle="--", linewidth=1.1,
                          label="Target: 5%"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .89),
               frameon=False, ncol=3, handlelength=2, columnspacing=1.8,
               fontsize=font_size-1)
    fig.suptitle(title, x=.53, y=.974, fontsize=font_size+1)
    n = next(iter(rows.values()))["n_generated"]
    for panel, method in enumerate(METHODS):
        position = axes[panel].get_position()
        fig.text((position.x0+position.x1)/2, .125, available_note(rows, method),
                 ha="center", va="center", fontsize=font_size-2.5)
    fig.text(.53, .075,
        f"Same {n:,} observations generated with A; each variant uses its own " + r"$L_C$" + ".",
        ha="center", va="center", fontsize=font_size-1.5)
    fig.text(.53, .035,
        "Points: frequencies among available decisions; intervals: exact 95% Clopper–Pearson.",
        ha="center", va="center", fontsize=font_size-1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--font-size", type=float, default=12)
    parser.add_argument("--width", type=float, default=12.4)
    parser.add_argument("--height", type=float, default=5.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="Effect of equal xenon response shapes in a strong-memory scenario")
    args = parser.parse_args()
    if args.width < 9 or args.height < 4.5 or args.font_size <= 0 or args.dpi < 72:
        parser.error("Choose width >= 9, height >= 4.5, a positive font size, and dpi >= 72.")
    rows = read_summary(args.input.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows, font_size=args.font_size, width=args.width,
                      height=args.height, title=args.title)
    fig.savefig(output / "false_detection_comparison.pdf", dpi=args.dpi,
        metadata={"Title": args.title, "Author": "XeMat",
                  "Subject": "Paired response-model comparison on saved strong-memory observations",
                  "CreationDate": None, "ModDate": None})
    fig.savefig(output / "false_detection_comparison.png", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved false_detection_comparison.pdf and .png ({args.dpi} dpi) in {output}")


if __name__ == "__main__":
    main()
