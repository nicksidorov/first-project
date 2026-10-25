"""Plot saved XeMat covariance ellipses; no fitting or Monte Carlo generation.

Example:
    python plot_ellipse.py --input ellipses.json --quality fit_quality.csv

The two-dimensional joint contour uses chi-square(df=2, p=0.95), not a
one-dimensional 1.96-sigma radius. All coordinates are already in mBq/m3.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

_CACHE = Path(tempfile.gettempdir()) / "xemat-aug17-ellipse-cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ("wls", "poisson")
METHOD_LABELS = {"wls": "LS", "poisson": "Poisson ML"}
COLORS = {"wls": "#1f77b4", "poisson": "#ff7f0e"}
PREFIX = "xe133_xe131m_ellipse"


def read_ellipses(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        key = (row["method_key"], row["uncertainty"])
        if key in result:
            raise ValueError(f"Duplicate ellipse: {key}")
        if row["chi2_df"] != 2 or row["confidence_level"] != 0.95:
            raise ValueError("This figure represents joint 95% regions for two coordinates")
        # For two degrees of freedom chi-square quantile = -2 log(1-p).
        expected_quantile = -2 * np.log1p(-row["confidence_level"])
        np.testing.assert_allclose(row["chi2_quantile"], expected_quantile, rtol=1e-13)
        center = np.asarray(row["center_mBq_m3"], dtype=float)
        covariance = np.asarray(row["covariance_mBq2_m6"], dtype=float)
        if center.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError(f"Wrong dimensions for {key}")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"Non-finite values for {key}")
        np.testing.assert_allclose(covariance, covariance.T, rtol=1e-13, atol=1e-14)
        np.linalg.cholesky(covariance)
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        np.testing.assert_allclose(correlation, row["correlation"], rtol=1e-13)
        result[key] = row
    expected = {(method, kind) for method in METHODS for kind in ("statistical", "total")}
    if set(result) != expected:
        raise ValueError("Expected statistical and configured-total blocks for both free estimators")
    for method in METHODS:
        np.testing.assert_array_equal(result[method, "statistical"]["center_mBq_m3"],
                                      result[method, "total"]["center_mBq_m3"])
    return result


def read_quality(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["method_key"]: row for row in csv.DictReader(stream)}
    for method in METHODS:
        if rows[method]["success"].lower() != "true":
            raise ValueError(f"Unsuccessful free fit: {method}")
    return rows


def ellipse_points(row, count=1001):
    """Map the unit circle through sqrt(q K); return actual concentrations."""
    center = np.asarray(row["center_mBq_m3"], dtype=float)
    covariance = np.asarray(row["covariance_mBq2_m6"], dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    angles = np.linspace(0, 2 * np.pi, count)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    displacement = vectors @ (np.sqrt(row["chi2_quantile"] * values)[:, None] * circle)
    return center[:, None] + displacement


def make_figure(rows, quality, *, width=11.8, height=7.0, font_size=12):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": font_size, "mathtext.fontset": "cm", "text.usetex": False,
        "axes.labelsize": font_size + 1, "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - .5, "ytick.labelsize": font_size - .5,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.225, top=.79, wspace=.12)
    points = {key: ellipse_points(row) for key, row in rows.items()}
    combined = np.concatenate(list(points.values()), axis=1)
    low, high = combined.min(axis=1), combined.max(axis=1)
    padding = .17 * (high - low)
    limits = np.c_[low - padding, high + padding]
    for panel, (ax, method) in enumerate(zip(axes, METHODS)):
        color = COLORS[method]
        ax.set_axisbelow(True)
        ax.grid(color="#e6e6e6", linewidth=.7)
        stat = points[method, "statistical"]
        total = points[method, "total"]
        ax.fill(stat[0], stat[1], color=color, alpha=.10, linewidth=0, zorder=2)
        ax.plot(stat[0], stat[1], color=color, linewidth=1.8, zorder=3)
        ax.plot(total[0], total[1], color=color, linewidth=1.7, linestyle=(0, (5, 3)), zorder=3)
        center = rows[method, "statistical"]["center_mBq_m3"]
        ax.plot(*center, marker="+", color="#202020", markersize=11,
                markeredgewidth=1.7, linestyle="none", zorder=4)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.set_xticks([value for value in ax.get_xticks() if limits[0,0] <= value <= limits[0,1]])
        ax.set_yticks([value for value in ax.get_yticks() if limits[1,0] <= value <= limits[1,1]])
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(r"$a_V(^{133}\mathrm{Xe})\;[\mathrm{mBq}/\mathrm{m}^{3}]$", labelpad=9)
        ax.set_title(f"({'ab'[panel]}) {METHOD_LABELS[method]}", loc="left", pad=12)
        r_stat = rows[method, "statistical"]["correlation"]
        r_total = rows[method, "total"]["correlation"]
        ax.text(.965, .96, rf"$r_{{\mathrm{{stat}}}}={r_stat:.3f}$" + "\n" +
                rf"$r_{{\mathrm{{total}}}}={r_total:.3f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=font_size,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=3), zorder=5)
    axes[0].set_ylabel(r"$a_V(^{131\mathrm{m}}\mathrm{Xe})\;[\mathrm{mBq}/\mathrm{m}^{3}]$", labelpad=9)
    fig.suptitle("Nominal 95% joint confidence regions", y=.95, fontsize=font_size + 4)
    handles = [Line2D([], [], color="#444444", linewidth=1.8, label="Statistical covariance"),
               Line2D([], [], color="#444444", linewidth=1.7, linestyle=(0, (5, 3)),
                      label="Including configured systematics"),
               Line2D([], [], color="#202020", marker="+", markersize=10,
                      markeredgewidth=1.7, linestyle="none", label="Free estimate")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .885),
               frameon=False, ncol=3, fontsize=font_size - .3, columnspacing=1.6, handlelength=2.5)
    fig.text(.53, .155, r"Joint contour: $\Delta\mathbf{a}^{T}K^{-1}\Delta\mathbf{a}"
             r"=\chi^{2}_{2,\,0.95}=5.991$; equal axis scales.", ha="center", fontsize=font_size - .5)
    reduced_ls = float(quality["wls"]["reduced_statistic"])
    reduced_ml = float(quality["poisson"]["reduced_statistic"])
    fig.text(.53, .106, "Model-based approximation; fit diagnostics: " +
             rf"LS $\chi^2/\nu={reduced_ls:.2f}$; Poisson ML $D/\nu={reduced_ml:.2f}$ ($\nu=4$).",
             ha="center", fontsize=font_size - .5)
    fig.text(.53, .061, "Configured systematics: 0.1% efficiencies and 0.1% volume; covariance not inflated by fit statistic.",
             ha="center", fontsize=font_size - 1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ellipses.json")
    parser.add_argument("--quality", type=Path, default=ROOT / "fit_quality.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--font-size", type=float, default=12)
    parser.add_argument("--width", type=float, default=11.8)
    parser.add_argument("--height", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    rows, quality = read_ellipses(args.input), read_quality(args.quality)
    fig = make_figure(rows, quality, width=args.width, height=args.height, font_size=args.font_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / (PREFIX + ".pdf"))
    fig.savefig(args.output_dir / (PREFIX + ".png"), dpi=args.dpi)
    plt.close(fig)
    print("Saved", args.output_dir / (PREFIX + ".pdf"), "and PNG")


if __name__ == "__main__":
    main()
