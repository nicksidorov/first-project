"""Draw the saved LC false-positive results; no fitting or Monte Carlo.

Usage: python plot_false_detections.py [--summary summary.csv] [--output-dir .]
Only summary.csv is read. Frequencies/intervals are saved proportions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary: Path, output_dir: Path) -> None:
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_method = {row["method_key"]: row for row in rows}
    if len(rows) != 2 or set(by_method) != {"wls", "poisson"}:
        raise ValueError("Expected exactly one LS and one Poisson ML row")
    if any(row["isotope"] != "Xe133m" for row in rows):
        raise ValueError("This figure is for the fresh Xe133m null experiment")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(6.8, 3.75), constrained_layout=True)
    colors = {"wls": "#2474B7", "poisson": "#D8751F"}
    for y, family in enumerate(("poisson", "wls")):
        row = by_method[family]
        rate, lower, upper = [100*float(row[key]) for key in
                              ("frequency", "ci95_low", "ci95_high")]
        ax.errorbar(rate, y, xerr=[[rate-lower], [upper-rate]],
                    fmt="o" if family == "wls" else "s", markersize=6.5,
                    color=colors[family], capsize=4, elinewidth=1.6,
                    markeredgecolor="white", markeredgewidth=.6, zorder=4)
        ax.annotate(f"{rate:.3f}%  [{lower:.3f}, {upper:.3f}]%",
                    xy=(rate, y), xytext=(0, 15), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=colors[family])
    ax.axvline(5, color="#555555", linestyle="--", linewidth=1.1,
               label="Nominal level: 5%", zorder=2)
    all_lower = [100*float(row["ci95_low"]) for row in rows] + [5]
    all_upper = [100*float(row["ci95_high"]) for row in rows] + [5]
    low, high = min(all_lower), max(all_upper)
    span = max(high-low, .8)
    ax.set_xlim(max(0, low-.35*span), high+.42*span)
    ax.set_ylim(-.55, 1.6)
    ax.set_yticks([0, 1], ["Poisson ML", "LS"])
    ax.set_xlabel("False-positive frequency (%)")
    ax.set_title(r"Fresh $^{133m}\mathrm{Xe}$ absent: May 22, detector 221", pad=12)
    ax.grid(axis="x", color="#E7E7E7", linewidth=.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    n_values = sorted({int(row["n_generated"]) for row in rows})
    n_text = f"{n_values[0]:,}" if len(n_values) == 1 else "/".join(map(str, n_values))
    fail_text = "; ".join(f"{by_method[k]['method']}: {int(by_method[k]['n_fail']):,} unavailable"
                          for k in ("wls", "poisson"))
    fig.supxlabel(f"N = {n_text}; exact 95% Clopper–Pearson intervals\n{fail_text}",
                  fontsize=8.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "false_detections.pdf")
    fig.savefig(output_dir / "false_detections.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=here / "summary.csv")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    plot(args.summary, args.output_dir)
