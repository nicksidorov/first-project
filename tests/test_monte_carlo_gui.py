"""Состояния окна Монте-Карло без длительных статистических экспериментов."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from threading import Event
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog

from process_measurement import process_prepared_measurement
from project import MainWindow
from xemat import monte_carlo_dialog as gui


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
    return process_prepared_measurement(path, method="wls")


@pytest.fixture
def experiment():
    scenarios = []
    for label, activity, count in (("A=0", 0., 1), ("A=MDC", .2, 3)):
        scenarios.append(dict(
            label=label, activity=activity, n_requested=4, n_attempted=4,
            n_valid=3, n_failed=1, n_detected=count, detection_rate=count / 3,
            confidence_interval=(.01, .99), failure_probability_bounds=(count / 4, (count + 1) / 4),
            mean_L_C=.04, mean_free_activity=.03, std_free_activity=.02,
            failure_reasons={"solver <failed>": 1},
        ))
    return dict(
        component_name="Xe133m", component_index=1, method="wls",
        requested_trials_per_scenario=4, seed=12345, alpha=.05, beta_ii=.05,
        reference_L_C=.04, reference_MDC=.2, scenarios=scenarios,
        cancelled=False, warnings=["Test <warning>"],
    )


def wait_until(predicate, app, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    raise AssertionError("GUI condition was not reached")


def test_dialog_defaults_and_component_mapping(app, results):
    results["detection"] = replace(results["detection"], converged=np.array([True, False]))
    results["fit"] = replace(results["fit"], method="poisson_nonnegative")
    dialog = gui.MonteCarloDialog(results)
    assert dialog.isotope_combo.count() == 1
    assert dialog.isotope_combo.currentText() == "Xe133m"
    assert dialog.isotope_combo.currentData() == 1
    assert dialog.method_combo.currentData() == "poisson"
    assert dialog.trials_spin.value() == 1000
    assert dialog.seed_spin.value() == 12345
    assert dialog.start_button.isEnabled()
    assert not dialog.cancel_button.isEnabled()
    assert not dialog.save_button.isEnabled()
    dialog.close()


def test_nonfinite_or_unconverged_limits_not_selectable(app, results):
    results["detection"] = replace(
        results["detection"], L_C=np.array([np.nan, .1]), MDC=np.array([.2, np.inf]),
    )
    assert gui.eligible_components(results) == []
    dialog = gui.MonteCarloDialog(results)
    assert not dialog.start_button.isEnabled()
    assert "Нет изотопов" in dialog.status_label.text()
    dialog.close()


def test_radon_remains_nuisance_even_if_listed_in_prepared_detection(app, results):
    results["isotope_names"] = ["Rn 222", "Xe133m"]
    assert gui.eligible_components(results) == [1]


def test_dialog_runs_core_off_main_thread_and_displays_rates(app, results, experiment, monkeypatch):
    captured = {}
    release = Event()
    entered = Event()

    def runner(source, **options):
        captured.update(options)
        captured["source"] = source
        captured["thread"] = gui.QThread.currentThread()
        entered.set()
        assert release.wait(5)
        options["progress"](8, 8)
        return experiment

    monkeypatch.setattr(gui, "run_from_results", runner)
    dialog = gui.MonteCarloDialog(results)
    dialog.trials_spin.setValue(4)
    dialog.seed_spin.setValue(678)
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        assert dialog.is_running
        assert not dialog.settings_group.isEnabled()
        assert not dialog.start_button.isEnabled()
        assert dialog.cancel_button.isEnabled()
        assert not dialog.save_button.isEnabled()
        assert captured["thread"] is not app.thread()
        assert captured["source"] is results
        assert captured["component_index"] == 1
        assert captured["n_trials"] == 4
        assert captured["seed"] == 678
        assert captured["method"] == "wls"
        release.set()
        wait_until(lambda: not dialog.is_running, app)
        assert dialog.experiment_result is experiment
        assert dialog.settings_group.isEnabled()
        assert dialog.save_button.isEnabled()
        assert dialog.progress_bar.value() == 8
        text = dialog.results_output.toPlainText()
        assert "33.33%" in text
        assert "100.00%" in text
        assert "5.00%" in text and "95.00%" in text
        assert "неуспешно: 1" in text
        assert "25.00% … 50.00%" in text
        assert "solver <failed>" in text
        assert "Test <warning>" in text
    finally:
        release.set()
        dialog.worker.requestInterruption()
        dialog.worker.wait(5000)
        app.processEvents()
        dialog.close()


@pytest.mark.parametrize("close", [False, True])
def test_cancel_preserves_partial_result_and_defers_close(app, results, experiment, monkeypatch, close):
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
    dialog = gui.MonteCarloDialog(results)
    dialog.show()
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        if close:
            dialog.close()
        else:
            dialog.request_cancel()
        assert dialog.is_running
        assert dialog.isVisible()
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


def test_worker_error_restores_controls_without_false_result(app, results, monkeypatch):
    def runner(*args, **kwargs):
        raise ValueError("Нулевая модель недопустима")

    monkeypatch.setattr(gui, "run_from_results", runner)
    dialog = gui.MonteCarloDialog(results)
    dialog.start_experiment()
    wait_until(lambda: not dialog.is_running, app)
    assert dialog.experiment_result is None
    assert not dialog.save_button.isEnabled()
    assert dialog.settings_group.isEnabled()
    assert dialog.start_button.isEnabled()
    assert "Нулевая модель недопустима" in dialog.status_label.text()
    dialog.close()


def test_save_json_and_text_export_exact_experiment(app, results, experiment, tmp_path, monkeypatch):
    dialog = gui.MonteCarloDialog(results)
    dialog.experiment_result = experiment
    dialog._set_running(False)
    path = tmp_path / "experiment"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "Данные эксперимента JSON (*.json)"))
    dialog.save_results()
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) == json.loads(json.dumps(experiment))
    written = []
    monkeypatch.setattr(gui, "save_monte_carlo_report", lambda result, target: written.append((result, target)))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "Текстовый отчёт (*.txt)"))
    dialog.save_results()
    assert written == [(experiment, path.with_suffix(".txt"))]
    dialog.close()


def test_main_button_follows_stored_result_and_retains_dialog(app, results, monkeypatch):
    monkeypatch.setattr(gui.MonteCarloDialog, "exec_", lambda self: 0)
    window = MainWindow()
    assert not window.monte_carlo_button.isEnabled()
    window.on_processing_finished(results)
    assert window.monte_carlo_button.isEnabled()
    window.method_combo.setCurrentIndex(window.method_combo.findData("poisson"))
    window.checkbox_fixed_roi.setChecked(not window.checkbox_fixed_roi.isChecked())
    assert window.results is results
    assert window.monte_carlo_button.isEnabled()
    window.open_monte_carlo()
    dialog = window.monte_carlo_dialog
    assert dialog.source_results is results
    assert dialog.method_combo.currentData() == "wls"
    window.open_monte_carlo()
    assert window.monte_carlo_dialog is dialog
    window._set_busy(True)
    assert not window.monte_carlo_button.isEnabled()
    window._set_busy(False)
    window.invalidate_results()
    assert not window.monte_carlo_button.isEnabled()
    window.close()


def test_failed_fit_disables_experiment(app, results):
    results["fit"] = replace(results["fit"], success=False)
    window = MainWindow()
    window.on_processing_finished(results)
    assert not window.monte_carlo_button.isEnabled()
    window.close()


def test_main_close_cannot_destroy_running_experiment(app, results, experiment, monkeypatch):
    monkeypatch.setattr(gui.MonteCarloDialog, "exec_", lambda self: 0)
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
    window.open_monte_carlo()
    dialog = window.monte_carlo_dialog
    dialog.start_experiment()
    try:
        wait_until(entered.is_set, app)
        assert not window.close()
        assert window.isVisible()
        assert dialog.is_running
        release.set()
        wait_until(lambda: not dialog.is_running, app)
        assert window.close()
    finally:
        release.set()
        dialog.worker.requestInterruption()
        dialog.worker.wait(5000)
        app.processEvents()
        window.close()
