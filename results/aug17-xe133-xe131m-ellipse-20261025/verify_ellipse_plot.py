"""Check the saved figure and LaTeX fragment without fitting or simulation."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone

import plot_ellipse as plot
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = plot.read_ellipses(ROOT / "ellipses.json")
    quality = plot.read_quality(ROOT / "fit_quality.csv")
    fig = plot.make_figure(rows, quality)
    fig.set_dpi(300)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    checks = []
    assert len(fig.axes) == 2
    for ax, method in zip(fig.axes, plot.METHODS):
        assert float(ax.get_aspect()) == 1
        np.testing.assert_array_equal(ax.get_xlim(), fig.axes[0].get_xlim())
        np.testing.assert_array_equal(ax.get_ylim(), fig.axes[0].get_ylim())
        origin, xstep, ystep = ax.transData.transform([[0,0], [1,0], [0,1]])
        np.testing.assert_allclose(np.linalg.norm(xstep-origin), np.linalg.norm(ystep-origin), rtol=1e-11)
        for line, kind in zip(ax.lines[:2], ("statistical", "total")):
            row = rows[method, kind]
            xy = np.asarray(line.get_data())
            delta = xy - np.asarray(row["center_mBq_m3"])[:,None]
            covariance = np.asarray(row["covariance_mBq2_m6"])
            squared_distance = np.einsum("ij,ij->j", delta, np.linalg.solve(covariance, delta))
            expected = -2*np.log(0.05)
            np.testing.assert_allclose(squared_distance, expected, rtol=5e-13, atol=5e-13)
            assert xy[0].min() > ax.get_xlim()[0] and xy[0].max() < ax.get_xlim()[1]
            assert xy[1].min() > ax.get_ylim()[0] and xy[1].max() < ax.get_ylim()[1]
            checks.append(dict(method=method, uncertainty=kind, n_contour_points=xy.shape[1],
                max_absolute_quadratic_form_error=float(np.max(np.abs(squared_distance-expected)))))
        np.testing.assert_allclose(np.asarray(ax.lines[2].get_data()).ravel(),
                                   rows[method, "statistical"]["center_mBq_m3"], rtol=0, atol=0)
    clipped = []
    n_text = 0
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.width == 0 or bounds.height == 0:
            continue
        n_text += 1
        if bounds.x0 < canvas.x0-.5 or bounds.x1 > canvas.x1+.5 or bounds.y0 < canvas.y0-.5 or bounds.y1 > canvas.y1+.5:
            clipped.append(text.get_text())
    assert not clipped, clipped
    saved = Image.open(ROOT / (plot.PREFIX + ".png")).convert("RGBA")
    expected_pixels = np.asarray(fig.canvas.buffer_rgba())
    np.testing.assert_array_equal(np.asarray(saved), expected_pixels)
    metadata = Image.open(ROOT / (plot.PREFIX + ".png")).info
    dpi = metadata["dpi"]
    np.testing.assert_allclose(dpi, [300,300], atol=.01, rtol=0)
    assert saved.size == (3540, 2100)
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["mathtext.fontset"] == "cm"
    pdf = ROOT / (plot.PREFIX + ".pdf")
    assert pdf.read_bytes().startswith(b"%PDF-")
    pdf_info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert "Pages:           1" in pdf_info
    latex_dir = Path(tempfile.mkdtemp(prefix="xemat-ellipse-latex-", dir="/tmp"))
    for name in ("figures.tex", plot.PREFIX + ".pdf"):
        shutil.copy2(ROOT/name, latex_dir/name)
    (latex_dir/"check.tex").write_text(
        r"\documentclass[twocolumn]{article}" + "\n" + r"\usepackage{graphicx}" + "\n" +
        r"\begin{document}\input{figures.tex}\end{document}" + "\n")
    completed = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"],
        cwd=latex_dir, capture_output=True, text=True)
    (latex_dir/"compile_stdout.txt").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout[-3000:]
    latex_log = (latex_dir/"check.log").read_text()
    assert "Overfull" not in latex_log
    plt.close(fig)
    files = ["plot_ellipse.py", "verify_ellipse_plot.py", "ellipses.json", "fit_quality.csv", "figures.tex",
             plot.PREFIX + ".pdf", plot.PREFIX + ".png"]
    result = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
        no_fits=True, no_monte_carlo=True, n_axes=2, shared_limits=True, equal_axis_scale=True,
        contour_checks=checks, centers_checked=True, visible_text_elements_checked=n_text,
        clipped_text=clipped, png_pixels=list(saved.size), png_dpi=list(dpi),
        png_matches_current_script_pixelwise=True, pdf_pages=1, serif_font=True,
        mathtext_fontset="cm", latex_compiled=True, latex_overfull_boxes=False,
        latex_check_directory=str(latex_dir),
        files_sha256={name:sha(ROOT/name) for name in files})
    (ROOT/"figure_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
