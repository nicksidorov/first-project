"""Проверки общей выборки, пар координат и эмпирического Pearson r."""

import io
import warnings

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from xemat.distribution_pairs import render_activity_pair_figure


def sample_result():
    return dict(
        component_names=["Xe135", "Xe133", "Xe131m", "Xe133m"],
        activity_samples=np.array([
            [-.001, -.003, .001, .002],
            [ .002,  .003, .004, .002],
            [ .004,  .007, -.002, .002],
            [ .003,  .005, .002, .002],
        ]),
        original_activity=np.array([-.002, .002, .001, 0]),
        generation_activity=np.array([.001, .003, .002, 0]),
        method="wls", generation="fitted", n_requested=4,
        n_attempted=4, n_valid=4, n_failed=0, seed=12345, cancelled=False,
    )


def correlation_text(axis):
    return next(text.get_text() for text in axis.texts if text.get_text().startswith("r ="))


def test_scatter_preserves_same_trial_pairs_and_converts_units_once():
    result = sample_result()
    source = result["activity_samples"].copy()
    figure = render_activity_pair_figure(result, bins=4)
    assert len(figure.axes) == 16
    # Панель [2,0]: x — Xe135, y — Xe131m, обе координаты той же строки.
    offsets = figure.axes[8].collections[0].get_offsets()
    np.testing.assert_array_equal(offsets, source[:, [0, 2]] * 1000)
    assert np.any(offsets < 0)
    assert sum(patch.get_height() for patch in figure.axes[0].patches) == 4
    assert min(patch.get_x() for patch in figure.axes[0].patches) == pytest.approx(-1.)
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [-2., -2.])
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_xdata(), [1., 1.])
    np.testing.assert_array_equal(result["activity_samples"], source)
    assert "мБк/м³" in figure.axes[12].get_xlabel()
    assert "мБк/м³" in figure.axes[12].get_ylabel()


def test_pearson_matches_independent_sum_formula_and_constant_is_undefined():
    result = sample_result()
    result["covariance"] = np.eye(4)  # Аналитическая ковариация здесь не источник r.
    figure = render_activity_pair_figure(result)
    first, third = result["activity_samples"][:, [0, 2]].T
    a, b = first - first.mean(), third - third.mean()
    expected = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))
    assert correlation_text(figure.axes[1]) == "r = +1.000"
    assert correlation_text(figure.axes[2]) == f"r = {expected:+.3f}"
    assert correlation_text(figure.axes[3]) == "r = —"
    assert figure.axes[1].get_facecolor() != figure.axes[2].get_facecolor()


def test_nonfinite_rows_are_excluded_from_every_panel_with_disclosed_count():
    result = sample_result()
    result["activity_samples"][1, 2] = np.nan
    result["activity_samples"][3, 3] = np.inf
    figure = render_activity_pair_figure(result)
    expected = result["activity_samples"][[0, 2], :]
    for row in range(4):
        assert sum(patch.get_height() for patch in figure.axes[5 * row].patches) == 2
        for column in range(row):
            offsets = figure.axes[4 * row + column].collections[0].get_offsets()
            np.testing.assert_array_equal(offsets, expected[:, [column, row]] * 1000)
    assert "Показано: 2" in figure._suptitle.get_text()
    assert "исключено строк с не конечными оценками: 2" in figure._suptitle.get_text()
    assert correlation_text(figure.axes[2]) == "r = -1.000"


@pytest.mark.parametrize("size", [0, 1, 3])
def test_empty_singleton_and_constant_samples_render_without_fake_zero_correlation(size):
    result = sample_result()
    result.update(activity_samples=np.zeros((size, 4)), n_attempted=size,
                  n_valid=size, cancelled=True, n_requested=100)
    figure = render_activity_pair_figure(result)
    assert correlation_text(figure.axes[1]) == "r = —"
    assert "Остановлено" in figure._suptitle.get_text()
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=60)
    assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_observed_mode_has_no_generation_reference_and_reuse_preserves_canvas():
    result = sample_result()
    result["generation"] = "observed"
    figure = Figure(figsize=(10, 9))
    canvas = FigureCanvasAgg(figure)
    assert render_activity_pair_figure(result, figure=figure, bins=3) is figure
    assert render_activity_pair_figure(result, figure=figure, bins=9) is figure
    assert figure.canvas is canvas
    assert len(figure.axes) == 16
    assert len(figure.legends) == 1
    assert len(figure.axes[0].patches) == 9
    assert len(figure.axes[0].lines) == 1
    assert "Poisson(x)" in figure._suptitle.get_text()
    assert [text.get_text() for text in figure.legends[0].texts] == ["Исходная свободная оценка"]


def test_common_activity_ranges_and_independent_histogram_count_ranges():
    figure = render_activity_pair_figure(sample_result())
    for row in range(4):
        for column in range(4):
            axis = figure.axes[4 * row + column]
            assert axis.get_xlim() == figure.axes[column].get_xlim()
            if row > column:
                assert axis.get_ylim() == figure.axes[row].get_xlim()
    assert figure.axes[0].get_ylim() != figure.axes[4].get_ylim()


def test_one_component_has_histogram_without_correlation_panel():
    result = sample_result()
    result["component_names"] = ["Xe133"]
    result["activity_samples"] = np.array([-.001, .002, .003])
    figure = render_activity_pair_figure(result)
    assert len(figure.axes) == 1
    assert sum(patch.get_height() for patch in figure.axes[0].patches) == 3


def test_identical_nonexact_floats_have_undefined_correlation_without_warnings():
    result = sample_result()
    result["activity_samples"] = np.full((7, 4), .1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        figure = render_activity_pair_figure(result)
    assert correlation_text(figure.axes[1]) == "r = —"
    assert any(text.get_text() == "Нет разброса компоненты" for text in figure.axes[1].texts)


def test_pair_header_and_bottom_legend_fit_at_700_pixels():
    figure = render_activity_pair_figure(sample_result(), Figure(figsize=(7, 7)))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    assert figure._suptitle.get_window_extent(renderer).y0 > figure.axes[0].title.get_window_extent(renderer).y1
    legend = figure.legends[0].get_window_extent(renderer)
    assert legend.y0 >= 0
    assert figure.axes[12].xaxis.label.get_window_extent(renderer).y0 > legend.y1


@pytest.mark.parametrize("bins", [0, -2, 1.5, True])
def test_invalid_bins_are_rejected(bins):
    with pytest.raises(ValueError, match="положительным целым"):
        render_activity_pair_figure(sample_result(), bins=bins)


def test_malformed_sample_columns_are_rejected():
    result = sample_result()
    result["activity_samples"] = np.empty((0, 3))
    with pytest.raises(ValueError, match="соответствовать"):
        render_activity_pair_figure(result)


def test_rendering_never_calls_fit_or_random_generator(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("График не должен пересчитывать Монте-Карло.")
    import xemat.solvers
    monkeypatch.setattr(xemat.solvers, "fit", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    render_activity_pair_figure(sample_result())
