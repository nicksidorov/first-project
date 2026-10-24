"""Plot saved false-detection frequencies and their exact confidence intervals.

This standalone script reads summary.csv only. It does not simulate observations,
fit spectra, change thresholds, or recompute the reported confidence intervals.
Requirements: NumPy and Matplotlib. No external LaTeX installation is needed.

Example:
    python plot_false_detections.py --input summary.csv --output-dir figures
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xemat-false-detections-mpl"))
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
METHODS = {
    "LS": {"color": "#1f77b4", "marker": "o", "offset": -.105},
    "Poisson ML": {"color": "#ff7f0e", "marker": "s", "offset": .105},
}


def read_summary(path: Path) -> dict:
    """Validate the eight saved rows without making statistical estimates."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "isotope", "n_generated", "n_valid", "n_false",
                    "frequency", "ci95_low", "ci95_high", "n_fail"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing summary columns: {sorted(required - set(reader.fieldnames or ())) }")
        source_rows = list(reader)
    expected = {(method, isotope) for method in METHODS for isotope in ISOTOPES}
    rows = {}
    for source in source_rows:
        key = (source["method"], source["isotope"])
        if key not in expected or key in rows:
            raise ValueError(f"Unexpected or repeated summary row: {key}")
        row = dict(source)
        for field in ("n_generated", "n_valid", "n_false", "n_fail"):
            row[field] = int(source[field])
        n, valid, count, failed = (row[name] for name in
                                  ("n_generated", "n_valid", "n_false", "n_fail"))
        if n <= 0 or not 0 <= count <= valid <= n or failed != n - valid:
            raise ValueError(f"Inconsistent realization counts for {key}")
        for field in ("frequency", "ci95_low", "ci95_high"):
            row[field] = float(source[field]) if source[field].strip() else float("nan")
        frequency, low, high = (row[name] for name in ("frequency", "ci95_low", "ci95_high"))
        if valid:
            if not all(math.isfinite(value) for value in (frequency, low, high)):
                raise ValueError(f"Non-finite frequency or confidence interval for {key}")
            if not 0 <= low <= frequency <= high <= 1:
                raise ValueError(f"Invalid frequency or confidence interval for {key}")
            if not math.isclose(frequency, count / valid, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Frequency differs from n_false / n_valid for {key}")
        elif not all(math.isnan(value) for value in (frequency, low, high)):
            raise ValueError(f"A row without available decisions must have missing statistics: {key}")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError(f"Missing summary rows: {sorted(expected - set(rows))}")
    return rows


def make_figure(rows: dict, *, font_size: float = 12, width: float = 8,
                height: float = 5.2, title: str = "False detections with strong gas memory"):
    """Return a figure whose plotted values are read directly from the CSV."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": font_size,
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1.5,
        "xtick.labelsize": font_size + 1,
        "ytick.labelsize": font_size - .5,
        "axes.linewidth": .8,
        "xtick.major.width": .8,
        "ytick.major.width": .8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.125, right=.975, bottom=.19, top=.78)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e7e7e7", linewidth=.7)
    ax.axhline(5.0, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
    x = np.arange(len(ISOTOPES), dtype=float)
    plotted_bounds = [5.0]
    missing_labels = []
    for method, style in METHODS.items():
        selected = [rows[(method, isotope)] for isotope in ISOTOPES]
        available = np.array([row["n_valid"] > 0 for row in selected], dtype=bool)
        centers = np.array([row["frequency"] * 100 for row in selected])
        lows = np.array([row["ci95_low"] * 100 for row in selected])
        highs = np.array([row["ci95_high"] * 100 for row in selected])
        plotted_bounds.extend(lows[available].tolist())
        plotted_bounds.extend(highs[available].tolist())
        if np.any(available):
            ax.errorbar(x[available] + style["offset"], centers[available],
                        yerr=np.stack((centers[available] - lows[available],
                                       highs[available] - centers[available])),
                        fmt=style["marker"], linestyle="none", color=style["color"],
                        markerfacecolor=style["color"], markeredgecolor="white",
                        markeredgewidth=.65, markersize=6.7, elinewidth=1.3,
                        capsize=4, capthick=1.1, zorder=4, label=method)
        missing_labels.extend(f"{method}: {ISOTOPES[index]}" for index in np.flatnonzero(~available))

    lower, upper = min(plotted_bounds), max(plotted_bounds)
    padding = max(.25, .16 * (upper - lower))
    ax.set_ylim(max(0, lower - padding), min(100, upper + padding))
    ax.set_xlim(-.5, 3.5)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("False-detection frequency (%)", labelpad=9)
    ax.set_xlabel("Isotope", labelpad=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    handles = [Line2D([], [], color=style["color"], marker=style["marker"],
                      linestyle="none", markersize=6.7, markeredgecolor="white",
                      markeredgewidth=.65, label=method)
               for method, style in METHODS.items()]
    handles.append(Line2D([], [], color="#555555", linestyle="--", linewidth=1.1,
                          label="Target: 5%"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .875),
               frameon=False, ncol=3, handlelength=2, columnspacing=1.7)
    fig.suptitle(title, x=.55, y=.966, fontsize=font_size + 1.5)

    generated = sorted({row["n_generated"] for row in rows.values()})
    denominators = sorted({row["n_valid"] for row in rows.values()})
    if len(generated) == len(denominators) == 1 and generated == denominators:
        count_note = f"{generated[0]:,} realizations per isotope and method; all decisions available."
    else:
        parts = []
        for method in METHODS:
            values = [rows[(method, isotope)]["n_valid"] for isotope in ISOTOPES]
            amount = (f"{values[0]:,}" if len(set(values)) == 1
                      else f"{min(values):,}–{max(values):,}")
            parts.append(f"{method} {amount}")
        count_note = "Available decisions per isotope: " + "; ".join(parts) + "."
    fig.text(.55, .073, "Points: empirical frequencies; bars: exact 95% Clopper–Pearson intervals.",
             ha="center", va="center", fontsize=font_size - 1.5)
    fig.text(.55, .032, count_note, ha="center", va="center", fontsize=font_size - 1.5)
    if missing_labels:
        ax.text(.02, .98, "No available decisions for " + "; ".join(missing_labels),
                transform=ax.transAxes, ha="left", va="top", fontsize=font_size - 2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--font-size", type=float, default=12)
    parser.add_argument("--width", type=float, default=8)
    parser.add_argument("--height", type=float, default=5.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default="False detections with strong gas memory")
    args = parser.parse_args()
    if args.width < 5 or args.height < 3.5 or args.font_size <= 0 or args.dpi < 72:
        parser.error("Choose width >= 5, height >= 3.5, positive font size, and dpi >= 72.")
    rows = read_summary(args.input.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows, font_size=args.font_size, width=args.width,
                      height=args.height, title=args.title)
    fig.savefig(output / "false_detections.png", dpi=args.dpi)
    fig.savefig(output / "false_detections.pdf", dpi=args.dpi,
                metadata={"Title": args.title, "Author": "XeMat",
                          "Subject": "Empirical false-detection frequencies and exact 95% binomial intervals",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Saved false_detections.pdf and false_detections.png ({args.dpi} dpi) in {output}")


if __name__ == "__main__":
    main()
