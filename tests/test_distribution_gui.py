"""GUI эксперимента: поток, отмена, снимок исходного расчёта и экспорт."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from threading import Event
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog

from process_measurement import process_prepared_measurement
from project import MainWindow
from xemat import distribution_dialog as gui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def results(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(
        path, M=np.eye(4), x=[10, 0, 3, 0], b=[2, 2, 2, 2],
        n_sample_parameters=2, coordinates="beta_gamma",
        T_c=[[.1, -.05], [0, .2]], H=np.diag([.1, .2]),
        isotope_names=["Xe133", "Xe133m"], mdc_component_indices=[1, 0],
    )
    result = process_prepared_measurement(path, method="wls")
    # Эксперимент с распределениями не зависит от успешного расчёта порогов.
    result["detection"] = None
    return result


@pytest.fixture
def experiment():
    return dict(
        experiment="activity_distributions", experiment_version=1,
        generation="fitted", method="wls", component_indices=np.array([0, 1]),
        component_names=["Xe133", "Xe133m"],
        activity_samples=np.array([[-.1, .2], [.1, .3], [.2, -.1]]),
        original_activity=np.array([.08, .19]),
        original_stat_sigma=np.array([.02, .01]),
        generation_activity=np.array([.1, .2]), generation_mu=np.array([10., 2., 5., 4.]),
        generator_fit_method="poisson_nonnegative", generator_physical_parameters=np.array([.1, .2, .3, .4]),
        summary=[
            dict(component_index=i, name=name, mean=.01, std=.03, median=.02,
                 q025=-.04, q975=.06, mean_stat_sigma=.025, sigma_count=3)
            for i, name in enumerate(["Xe133", "Xe133m"])
        ],
        n_requested=4, n_attempted=4, n_valid=3, n_failed=1,
        failure_reasons={"solver <failed>": 1}, seed=12345, cancelled=False,
        warnings=["Test <warning>"], detector_background_estimated=False,
        physical_coordinates="[a_c; gamma]", source={},
    )


def wait_until(predicate, app, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("GUI condition was not reached")


def test_defaults_only_free_methods_and_no_detection_requirement(app, results):
    results["fit"] = replace(results["fit"], method="poisson_nonnegative")
    dialog = gui.DistributionDialog(results)
    assert dialog.generation_combo.currentData() == "fitted"
    assert [dialog.method_combo.itemData(i) for i in range(dialog.method_combo.count())] == ["wls", "poisson"]
    assert dialog.method_combo.currentData() == "poisson"
    assert dialog.start_button.isEnabled()
    assert dialog.trials_spin.value() == 1000
    assert dialog.seed_spin.value() == 12345
    assert dialog.result_tabs.tabText(0) == "Гистограммы"
    assert dialog.result_tabs.tabText(1) == "Корреляции активностей"
    assert dialog.result_tabs.tabText(2) == "Статистика и пояснения"
    assert dialog.result_tabs.widget(0) is dialog.canvas
    assert dialog.result_tabs.widget(1) is dialog.pairs_scroll
    assert dialog.result_tabs.widget(2) is dialog.results_output
    assert not dialog.save_button.isEnabled()
    assert "неотрицательным ММП" in dialog.assumption_label.text()
    dialog.generation_combo.setCurrentIndex(dialog.generation_combo.findData("observed"))
    assert "во всех повторах там будет ноль" in dialog.assumption_label.text()
    assert "Единой заданной активности" in dialog.assumption_label.text()
    dialog.close()


def test_compute_off_main_thread_plot_on_main_thread_and_rebin_without_rerun(app, results, experiment, monkeypatch):
    captured = {}
    release = Event()
    entered = Event()
    calls = []
    render = gui.render_distribution_figure

    def runner(source, **options):
        captured.update(options)
        captured["source"] = source
        captured["thread"] = gui.QThread.currentThread()
        calls.append("run")
        entered.set()
        assert release.wait(5)
        options["progress"](4, 4)
        return experiment

    def renderer(*args, **kwargs):
        captured["plot_thread"] = gui.QThread.currentThread()
        captured["bins"] = kwargs["bins"]
        return render(*args, **kwargs)

    monkeypatch.setattr(gui, "run_from_results", runner)
    monkeypatch.setattr(gui, "render_distribution_figure", renderer)
    dialog = gui.DistributionDialog(results)
    dialog.generation_combo.setCurrentIndex(dialog.generation_combo.findData("observed"))
    dialog.method_combo.setCurrentIndex(dialog.method_combo.findData("poisson"))
    dialog.trials_spin.setValue(4)
    dialog.seed_spin.setValue(678)
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        assert dialog.is_running and not dialog.settings_group.isEnabled()
        assert dialog.cancel_button.isEnabled() and not dialog.save_button.isEnabled()
        assert captured["thread"] is not app.thread()
        assert captured["source"] is results
        assert captured["n_trials"] == 4 and captured["seed"] == 678
        assert captured["generation"] == "observed" and captured["method"] == "poisson"
        release.set()
        wait_until(lambda: not dialog.is_running, app)
        assert dialog.experiment_result is experiment
        assert captured["plot_thread"] is app.thread()
        assert dialog.save_button.isEnabled() and dialog.save_plot_button.isEnabled()
        assert dialog.progress_bar.value() == 4
        text = dialog.results_output.toPlainText()
        assert "мБк/м³" in text and "СКО Монте-Карло" in text
        assert "σ исходная" in text and "Центральные 95%" in text
        assert "неуспешно: 1" in text and "solver <failed>" in text
        assert "Test <warning>" in text
        dialog.bins_spin.setValue(47)
        assert captured["bins"] == 47
        assert calls == ["run"]
        dialog.seed_spin.setValue(679)
        assert "Показан предыдущий результат" in dialog.status_label.text()
        assert dialog.experiment_result is experiment
    finally:
        release.set()
        dialog.worker.requestInterruption()
        dialog.worker.wait(5000)
        app.processEvents()
        dialog.close()


@pytest.mark.parametrize("close", [False, True])
def test_cancel_preserves_partial_results_and_defers_close(app, results, experiment, monkeypatch, close):
    entered = Event()
    release = Event()
    partial = deepcopy(experiment)
    partial["cancelled"] = True

    def runner(source, **options):
        entered.set()
        assert release.wait(5)
        assert options["should_cancel"]()
        return partial

    monkeypatch.setattr(gui, "run_from_results", runner)
    dialog = gui.DistributionDialog(results)
    dialog.show()
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        if close:
            dialog.close()
        else:
            dialog.request_cancel()
        assert dialog.is_running and dialog.isVisible()
        assert not dialog.cancel_button.isEnabled()
        release.set()
        wait_until(lambda: not dialog.is_running, app)
        assert dialog.experiment_result is partial
        assert dialog.save_button.isEnabled()
        assert "Частичные результаты" in dialog.status_label.text()
        assert "результаты частичные" in dialog.results_output.toPlainText()
        assert dialog.isVisible() is not close
    finally:
        release.set()
        dialog.worker.requestInterruption()
        dialog.worker.wait(5000)
        app.processEvents()
        dialog.close()


def test_failed_generation_restores_controls_without_fake_histogram(app, results, monkeypatch):
    def runner(*args, **kwargs):
        raise ValueError("Опорная ММП-подгонка не сошлась")

    monkeypatch.setattr(gui, "run_from_results", runner)
    dialog = gui.DistributionDialog(results)
    dialog.start_experiment()
    wait_until(lambda: not dialog.is_running, app)
    assert dialog.experiment_result is None
    assert not dialog.save_button.isEnabled() and not dialog.save_plot_button.isEnabled()
    assert dialog.settings_group.isEnabled() and dialog.start_button.isEnabled()
    assert "Опорная ММП-подгонка не сошлась" in dialog.status_label.text()
    dialog.close()


def test_export_json_text_and_rendered_png(app, results, experiment, tmp_path, monkeypatch):
    dialog = gui.DistributionDialog(results)
    dialog._on_result(experiment)
    dialog._set_running(False)
    path = tmp_path / "distribution"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "Данные эксперимента JSON (*.json)"))
    dialog.save_results()
    written = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["activity_samples"] == experiment["activity_samples"].tolist()
    assert written["activity_samples"][0][0] < 0
    assert written["original_stat_sigma"] == [.02, .01]
    saves = []
    monkeypatch.setattr(gui, "save_distribution_report", lambda result, target: saves.append((result, target)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "Текстовый отчёт (*.txt)"))
    dialog.save_results()
    assert saves == [(experiment, path.with_suffix(".txt"))]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "Изображение PNG (*.png)"))
    dialog.save_plot()
    assert path.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    dialog.close()


def test_stats_convert_bq_to_mbq_once_and_do_not_claim_confidence_interval(experiment):
    text = gui.DistributionDialog._result_html(experiment)
    assert "<td>10</td><td>30</td><td>20</td><td>25</td><td>20</td>" in text
    assert "<td>-40 … 60</td>" in text
    assert "не доверительный интервал истинной активности" in text


def test_pair_tab_uses_saved_rows_lazily_and_invalidates_on_bins_or_new_result(app, results, experiment, monkeypatch):
    rendered = []

    def pairs(result, figure, *, bins):
        rendered.append((result, bins))
        return figure

    def unexpected_run(*args, **kwargs):
        pytest.fail("Переключение графиков не должно запускать Монте-Карло")

    monkeypatch.setattr(gui, "render_activity_pair_figure", pairs)
    monkeypatch.setattr(gui, "run_from_results", unexpected_run)
    dialog = gui.DistributionDialog(results)
    dialog._on_result(experiment)
    dialog._set_running(False)
    assert rendered == []
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert len(rendered) == 1 and rendered[-1][0] is experiment
    dialog.result_tabs.setCurrentWidget(dialog.results_output)
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert len(rendered) == 1
    dialog.result_tabs.setCurrentWidget(dialog.canvas)
    dialog.bins_spin.setValue(41)
    assert len(rendered) == 1
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert rendered[-1][1] == 41 and len(rendered) == 2
    dialog.seed_spin.setValue(54321)
    dialog.generation_combo.setCurrentIndex(dialog.generation_combo.findData("observed"))
    assert dialog.experiment_result is experiment and len(rendered) == 2
    replacement = deepcopy(experiment)
    replacement["seed"] = 54321
    replacement["activity_samples"] = replacement["activity_samples"] + .5
    dialog._on_result(replacement)
    assert len(rendered) == 3 and rendered[-1][0] is replacement
    dialog.result_tabs.setCurrentWidget(dialog.canvas)
    final = deepcopy(replacement)
    final["method"] = "poisson"
    dialog._on_result(final)
    assert len(rendered) == 3
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert len(rendered) == 4 and rendered[-1][0] is final
    dialog.close()


def test_png_exports_active_figure_and_is_disabled_for_statistics(app, results, experiment, tmp_path, monkeypatch):
    dialog = gui.DistributionDialog(results)
    dialog._on_result(experiment)
    dialog._set_running(False)
    saves, paths = [], []
    monkeypatch.setattr(dialog.figure, "savefig", lambda *args, **kwargs: saves.append("histograms"))
    monkeypatch.setattr(dialog.pairs_figure, "savefig", lambda *args, **kwargs: saves.append("pairs"))

    def choose_file(parent, title, filename, selected_filter):
        paths.append(filename)
        return str(tmp_path / filename), selected_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_file)
    assert "гистограммы" in dialog.save_plot_button.text()
    dialog.save_plot()
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert "корреляции" in dialog.save_plot_button.text()
    assert dialog.save_plot_button.isEnabled()
    dialog.save_plot()
    assert saves == ["histograms", "pairs"]
    assert paths == ["activity_distributions.png", "activity_correlations.png"]
    dialog.result_tabs.setCurrentWidget(dialog.results_output)
    assert not dialog.save_plot_button.isEnabled()
    dialog.save_plot()
    assert saves == ["histograms", "pairs"]
    assert len(paths) == 2
    dialog.close()


def test_pair_tab_keeps_large_canvas_scrollable_on_small_window(app, results, experiment):
    dialog = gui.DistributionDialog(results)
    dialog._on_result(experiment)
    dialog._set_running(False)
    dialog.resize(1024, 768)
    dialog.show()
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    app.processEvents()
    assert dialog.height() == 768
    assert dialog.pairs_canvas.width() >= 800 and dialog.pairs_canvas.height() >= 800
    scrollbar = dialog.pairs_scroll.verticalScrollBar()
    assert scrollbar.maximum() > 0
    position = QPoint(100, 100)
    event = QWheelEvent(
        QPointF(position), QPointF(dialog.pairs_canvas.mapToGlobal(position)),
        QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )
    app.sendEvent(dialog.pairs_canvas, event)
    assert scrollbar.value() > 0
    scrollbar.setValue(scrollbar.maximum())
    assert scrollbar.value() == scrollbar.maximum()
    dialog.close()


def test_empty_cancelled_result_can_show_and_export_pairs(app, results, experiment):
    empty = deepcopy(experiment)
    empty.update(activity_samples=np.empty((0, 2)), n_attempted=0, n_valid=0, n_failed=0,
                 failure_reasons={}, cancelled=True)
    for row in empty["summary"]:
        row.update(mean=None, std=None, median=None, q025=None, q975=None, mean_stat_sigma=None, sigma_count=0)
    dialog = gui.DistributionDialog(results)
    dialog._on_result(empty)
    dialog._on_finished()
    dialog.result_tabs.setCurrentWidget(dialog.pairs_scroll)
    assert dialog.experiment_result is empty
    assert dialog._pairs_plot_error is None and not dialog._pairs_dirty
    assert dialog.save_plot_button.isEnabled()
    assert "Частичные результаты" in dialog.status_label.text()
    assert "результаты частичные" in dialog.results_output.toPlainText()
    dialog.close()


def test_main_uses_stored_snapshot_retains_window_and_resets_on_input_change(app, results, monkeypatch):
    monkeypatch.setattr(gui.DistributionDialog, "exec_", lambda self: 0)
    window = MainWindow()
    assert not window.distribution_button.isEnabled()
    window.on_processing_finished(results)
    assert window.distribution_button.isEnabled()
    assert not window.monte_carlo_button.isEnabled()
    window.method_combo.setCurrentIndex(window.method_combo.findData("poisson"))
    window.checkbox_fixed_roi.setChecked(not window.checkbox_fixed_roi.isChecked())
    assert window.results is results and window.distribution_button.isEnabled()
    window.open_distributions()
    dialog = window.distribution_dialog
    assert dialog.source_results is results
    assert dialog.method_combo.currentData() == "wls"
    window.open_distributions()
    assert window.distribution_dialog is dialog
    window._set_busy(True)
    assert not window.distribution_button.isEnabled()
    window._set_busy(False)
    assert window.distribution_button.isEnabled()
    replacement = dict(results)
    window.on_processing_finished(replacement)
    assert window.distribution_dialog is None
    window.open_distributions()
    assert window.distribution_dialog is not dialog
    assert window.distribution_dialog.source_results is replacement
    window.invalidate_results()
    assert not window.distribution_button.isEnabled()
    assert window.distribution_dialog is None
    window.close()


def test_failed_fit_disables_distribution_button(app, results):
    results["fit"] = replace(results["fit"], success=False)
    window = MainWindow()
    window.on_processing_finished(results)
    assert not window.distribution_button.isEnabled()
    window.close()


def test_main_close_does_not_destroy_running_worker(app, results, experiment, monkeypatch):
    monkeypatch.setattr(gui.DistributionDialog, "exec_", lambda self: 0)
    release = Event()
    entered = Event()

    def runner(source, **options):
        entered.set()
        assert release.wait(5)
        assert options["should_cancel"]()
        return experiment

    monkeypatch.setattr(gui, "run_from_results", runner)
    window = MainWindow()
    window.show()
    window.on_processing_finished(results)
    window.open_distributions()
    dialog = window.distribution_dialog
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        assert not window.close()
        assert window.isVisible() and dialog.is_running
        release.set()
        wait_until(lambda: not dialog.is_running, app)
        assert window.close()
    finally:
        release.set()
        dialog.worker.requestInterruption()
        dialog.worker.wait(5000)
        app.processEvents()
        window.close()
