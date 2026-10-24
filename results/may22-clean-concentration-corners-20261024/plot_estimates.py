"""Draw the two corner plots from saved free estimates; no XeMat calculations.

Only NumPy and Matplotlib are required. This script never generates counts,
fits spectra, or computes detection thresholds. Changing its style options
does not require repeating Monte Carlo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xemat-corner-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
ORDER = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
LABELS = [r"$^{135}\mathrm{Xe}$", r"$^{133}\mathrm{Xe}$",
          r"$^{131\mathrm{m}}\mathrm{Xe}$", r"$^{133\mathrm{m}}\mathrm{Xe}$"]
METHODS = {"ls": ("LS", "#1f77b4"), "poisson_ml": ("Poisson ML", "#ff7f0e")}


def load_arrays(path: Path):
    with np.load(path, allow_pickle=False) as stored:
        isotopes = stored["isotopes"].astype(str).tolist()
        if isotopes != ORDER:
            raise ValueError(f"Expected isotope order {ORDER}, got {isotopes}")
        units = str(stored["units"].item())
        if units != "mBq/m^3":
            raise ValueError(f"Expected mBq/m^3, got {units}")
        estimates = {method: np.array(stored[method], dtype=float) for method in METHODS}
        success = {method: np.array(stored[method + "_success"], dtype=bool) for method in METHODS}
        indices = np.array(stored["scatter_indices"], dtype=np.int64)
    n = len(estimates["ls"])
    for method in METHODS:
        if estimates[method].shape != (n, 4) or success[method].shape != (n,):
            raise ValueError(f"Invalid estimate/status dimensions for {method}")
    # A complete, successful free estimate is used regardless of its sign,
    # distance from the mean, or availability of a fitted covariance matrix.
    valid = {method: success[method] & np.all(np.isfinite(estimates[method]), axis=1)
             for method in METHODS}
    if any(np.count_nonzero(mask) < 2 for mask in valid.values()):
        raise ValueError("Each method needs at least two valid complete estimates")
    common = valid["ls"] & valid["poisson_ml"]
    expected_scatter_n = min(5000, int(common.sum()))
    if indices.ndim != 1 or len(indices) != expected_scatter_n:
        raise ValueError("The saved common scatter selection must contain min(5000, N_common) indices")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("Invalid or duplicate scatter indices")
    if not np.all(common[indices]):
        raise ValueError("Scatter indices must be valid for both methods")
    return estimates, valid, indices, units


def statistics_and_axes(estimates, valid, bins):
    complete = {method: estimates[method][valid[method]] for method in METHODS}
    stats = {}
    for method, values in complete.items():
        covariance = np.cov(values, rowvar=False, ddof=1)
        sd = np.sqrt(np.diag(covariance))
        if np.any(sd == 0):
            raise ValueError(f"A zero empirical standard deviation prevents correlation for {method}")
        stats[method] = {"n_valid": len(values), "mean": values.mean(axis=0),
                         "sd": sd, "correlation": covariance / np.outer(sd, sd)}
    # Pool only to set common plotting bounds and bins. Statistics remain
    # separate for the two methods, and include every valid row.
    pooled = np.vstack(list(complete.values()))
    low, high = pooled.min(axis=0), pooled.max(axis=0)
    low = np.minimum(low, 0.0)
    high = np.maximum(high, 0.0)
    span = high - low
    padding = np.where(span > 0, .035 * span, 1.0)
    limits = np.column_stack([low - padding, high + padding])
    edges = [np.linspace(left, right, bins + 1) for left, right in limits]
    density_limits = []
    for col in range(4):
        peak = max(float(np.histogram(values[:, col], bins=edges[col], density=True)[0].max())
                   for values in complete.values())
        density_limits.append([0.0, 1.16 * peak])
    return complete, stats, limits, edges, density_limits


def format_axis(axis):
    axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_powerlimits((-3, 4))
    axis.set_major_formatter(formatter)


def draw(method, estimates, complete, stats, indices, limits, edges,
         density_limits, output, size, font_size, point_size, alpha, dpi):
    title, color = METHODS[method]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": font_size,
        "axes.labelsize": font_size, "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - .5, "ytick.labelsize": font_size - .5,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(4, 4, figsize=(size, size))
    fig.subplots_adjust(left=.10, right=.98, bottom=.145, top=.88, wspace=.30, hspace=.25)
    shown = estimates[method][indices]
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(limits[col])
            format_axis(ax.xaxis)
            format_axis(ax.yaxis)
            if row == col:
                ax.hist(complete[method][:, col], bins=edges[col], density=True,
                        color=color, alpha=.58, edgecolor="white", linewidth=.22)
                # Put dashed zero over the mean line when the two nearly
                # coincide; the solid coloured line remains visible in gaps.
                ax.axvline(0, color="#222222", ls="--", lw=1.35, zorder=6)
                ax.axvline(stats[method]["mean"][col], color=color, lw=2.1, zorder=5)
                ax.set_ylim(density_limits[col])
                ax.set_title(LABELS[col], pad=7, fontweight="bold")
                ax.set_ylabel("Density", fontsize=font_size - .3)
            else:
                ax.set_ylim(limits[row])
                ax.axvline(0, color="#dadada", lw=.7, zorder=0)
                ax.axhline(0, color="#dadada", lw=.7, zorder=0)
                ax.scatter(shown[:, col], shown[:, row], s=point_size, c=color,
                           alpha=alpha, edgecolors="none", rasterized=True, zorder=2)
                ax.text(.04, .94, f"$r = {stats[method]['correlation'][row, col]:.3f}$",
                        transform=ax.transAxes, ha="left", va="top", fontsize=font_size + 1,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=2.2), zorder=6)
                if col == 0:
                    ax.set_ylabel(LABELS[row] + r" (mBq/m$^3$)")
                else:
                    ax.tick_params(axis="y", labelleft=False)
            if row == 3:
                ax.set_xlabel(LABELS[col] + r" (mBq/m$^3$)", labelpad=7)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(f"Free xenon concentration estimates — {title}", y=.976,
                 fontsize=font_size + 9, fontweight="bold", color=color)
    fig.text(.5, .936,
             f"Clean sample · {stats[method]['n_valid']:,} valid realizations · true concentrations = 0",
             ha="center", va="center", fontsize=font_size + .6)
    handles = [Line2D([], [], color="#222222", ls="--", lw=1.35, label="True zero"),
               Line2D([], [], color=color, lw=2.1, label="Empirical mean")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .067),
               ncol=2, frameon=False, fontsize=font_size + .3, handlelength=2.8, columnspacing=2.5)
    fig.text(.5, .039,
             f"Scatter panels: the same {len(indices):,} randomly selected realizations for both methods.",
             ha="center", va="center", fontsize=font_size - .2)
    fig.text(.5, .018,
             "Histograms, means and Pearson r use all valid estimates; no outliers removed.",
             ha="center", va="center", fontsize=font_size - .2)
    stem = "corner_" + method
    fig.savefig(output / (stem + ".png"), dpi=dpi)
    fig.savefig(output / (stem + ".pdf"), dpi=dpi,
                metadata={"Title": f"Free xenon concentration estimates — {title}",
                          "Author": "XeMat", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "estimates.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--bins", type=int, default=80, help="Common histogram bins per isotope")
    parser.add_argument("--size", type=float, default=12.4, help="Square figure size in inches")
    parser.add_argument("--font-size", type=float, default=13.0)
    parser.add_argument("--point-size", type=float, default=3.4)
    parser.add_argument("--alpha", type=float, default=.22)
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution and rasterized PDF scatter resolution")
    args = parser.parse_args()
    if args.bins < 2 or args.size < 6 or args.font_size <= 0 or args.point_size <= 0:
        parser.error("Invalid bins, figure size, font size, or point size")
    if not 0 < args.alpha <= 1 or args.dpi < 72:
        parser.error("alpha must be in (0,1] and dpi >= 72")
    source = args.input.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    estimates, valid, indices, units = load_arrays(source)
    complete, stats, limits, edges, density_limits = statistics_and_axes(estimates, valid, args.bins)
    for method in METHODS:
        draw(method, estimates, complete, stats, indices, limits, edges, density_limits,
             output, args.size, args.font_size, args.point_size, args.alpha, args.dpi)
    metadata = {
        "source_npz": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "isotope_order": ORDER, "units": units,
        "n_attempted": len(estimates["ls"]),
        "validity_rule": "Successful fit and finite full four-concentration vector; covariance availability is not a filter",
        "negative_estimates_and_outliers_retained": True,
        "scatter_n": len(indices), "scatter_indices_shared": True,
        "scatter_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "concentration_axis_limits": limits.tolist(),
        "histogram_bin_edges": [edge.tolist() for edge in edges],
        "histogram_density_limits": density_limits,
        "methods": {method: {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in details.items()}
                    for method, details in stats.items()},
        "figure_size_inches": [args.size, args.size], "png_dpi": args.dpi,
        "pdf_scatter_rasterized": True, "point_size": args.point_size,
        "point_alpha": args.alpha, "font_size": args.font_size,
        "colors": {method: value[1] for method, value in METHODS.items()},
    }
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved corner_ls and corner_poisson_ml in PDF and PNG, plus plot_metadata.json")
    for method in METHODS:
        print(method, "N_valid=", stats[method]["n_valid"], "scatter=", len(indices))


if __name__ == "__main__":
    main()
