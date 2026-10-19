"""Проверка единиц, отрицательных оценок и подписей графического отчёта."""

import io

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from xemat.distribution_plot import render_distribution_figure


def sample_result(n_components=4):
    samples = np.array([[-.001, .002, 0, -.003], [.002, .001, 0, -.002],
                        [.004, .003, 0, -.001], [.003, .005, 0, 0]])[:, :n_components]
    return {
        "generation": "fitted", "method": "wls",
        "component_names": ["Xe135", "Xe133", "Xe131m", "Xe133m"][:n_components],
        "component_indices": np.arange(n_components),
        "activity_samples": samples,
        "original_activity": np.array([-.002, .002, 0, -.001])[:n_components],
        "generation_activity": np.array([.001, .003, 0, .0005])[:n_components],
        "n_attempted": 4, "n_requested": 4, "n_valid": 4, "n_failed": 0,
        "seed": 42, "cancelled": False,
    }


def draw_and_export(figure):
    FigureCanvasAgg(figure).draw()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=60)
    assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_four_panels_preserve_negative_samples_and_convert_units_once():
    result = sample_result()
    original_samples = result["activity_samples"].copy()
    figure = render_distribution_figure(result, bins=5)
    assert len(figure.axes) == 4
    first = figure.axes[0]
    assert sum(patch.get_height() for patch in first.patches) == 4
    assert min(patch.get_x() for patch in first.patches) == pytest.approx(-1.)
    assert first.lines[0].get_xdata() == pytest.approx([-2., -2.])
    assert first.lines[1].get_xdata() == pytest.approx([1., 1.])
    # Свободная исходная оценка за пределами выборки тоже видна.
    assert first.get_xlim()[0] < -2 < first.get_xlim()[1]
    assert "мБк/м³" in figure.axes[2].get_xlabel()
    assert first.get_ylabel() == "Число реализаций"
    assert "Среднее: 2" in first.get_title()
    np.testing.assert_array_equal(result["activity_samples"], original_samples)
    draw_and_export(figure)


def test_observed_generator_has_no_claimed_truth_line():
    result = sample_result(1)
    result["generation"] = "observed"
    # Даже при случайно оставленном поле generation_activity нельзя рисовать
    # его как истинную активность сценария Poisson(x).
    figure = render_distribution_figure(result)
    assert len(figure.axes[0].lines) == 1
    assert "Poisson(x)" in figure._suptitle.get_text()
    assert [text.get_text() for text in figure.legends[0].texts] == ["Исходная свободная оценка"]
    draw_and_export(figure)


@pytest.mark.parametrize("size", [0, 1, 5])
def test_zero_empty_and_constant_samples_render(size):
    result = sample_result(2)
    result.update(activity_samples=np.zeros((size, 2)), n_valid=size,
                  n_attempted=size, n_requested=100, cancelled=True)
    figure = render_distribution_figure(result)
    assert len(figure.axes) == 2
    assert "остановлено" in figure._suptitle.get_text()
    assert "запрошено: 100" in figure._suptitle.get_text()
    if size:
        assert sum(patch.get_height() for patch in figure.axes[0].patches) == size
    else:
        assert any(text.get_text() == "Нет успешных реализаций" for text in figure.axes[0].texts)
    if size < 2:
        assert "СКО: —" in figure.axes[0].get_title()
    else:
        assert "СКО: 0" in figure.axes[0].get_title()
    draw_and_export(figure)


def test_existing_figure_is_reused_and_bin_change_does_not_accumulate_artists():
    figure = Figure(figsize=(11, 8))
    FigureCanvasAgg(figure)
    canvas = figure.canvas
    result = sample_result()
    first = render_distribution_figure(result, figure=figure, bins=4)
    second = render_distribution_figure(result, figure=figure, bins=10)
    assert first is second is figure
    assert figure.canvas is canvas
    assert len(figure.axes) == 4
    assert len(figure.legends) == 1
    assert len(figure.axes[0].patches) == 10
    assert sum(patch.get_height() for patch in figure.axes[0].patches) == 4
    np.testing.assert_array_equal(figure.get_size_inches(), [11, 8])


def test_title_discloses_failed_and_partial_runs():
    result = sample_result(1)
    result.update(n_requested=100, n_attempted=6, n_valid=4, n_failed=2,
                  method="poisson")
    figure = render_distribution_figure(result)
    text = figure._suptitle.get_text()
    assert "свободный ММП Пуассона" in text
    assert "4/6" in text
    assert "частичный результат" in text
    assert "ошибок: 2" in text
    assert "seed: 42" in text


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_rejects_invalid_bin_count(bins):
    with pytest.raises(ValueError, match="положительным целым"):
        render_distribution_figure(sample_result(), bins=bins)


def test_three_components_hide_unused_panel():
    figure = render_distribution_figure(sample_result(3))
    assert len(figure.axes) == 4
    assert [axis.get_visible() for axis in figure.axes] == [True, True, True, False]


def test_small_embedded_canvas_keeps_headers_legends_and_rows_separate():
    """Доступные ~400 px в окне 1024×768 не должны смешивать подписи."""
    figure = render_distribution_figure(sample_result(), Figure(figsize=(10, 4)))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    header = figure._suptitle.get_window_extent(renderer)
    first_title = figure.axes[0].title.get_window_extent(renderer)
    bottom_title = figure.axes[2].title.get_window_extent(renderer)
    legend = figure.legends[0].get_window_extent(renderer)
    xlabel = figure.axes[2].xaxis.label.get_window_extent(renderer)
    assert header.y0 > first_title.y1
    assert figure.axes[0].get_window_extent(renderer).y0 > bottom_title.y1
    assert xlabel.y0 > legend.y1
    assert legend.y0 >= 0
    assert header.y1 <= figure.bbox.y1
    # Constrained layout повторно учитывает размер после изменения окна.
    figure.set_size_inches(10, 3.3)
    canvas.draw()
    assert figure._suptitle.get_window_extent(renderer).y0 > figure.axes[0].title.get_window_extent(renderer).y1
