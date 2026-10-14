"""Маршрут трёх спектров в GUI и явный выбор радоновой калибровки."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

import project
from project import MainWindow, RadonCalibrationDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def select_three_files(window, tmp_path):
    for attribute in ("sample_file", "background_file", "detector_file"):
        path = tmp_path / (attribute + ".ims")
        path.touch()
        setattr(window, attribute, str(path))
    window.invalidate_results()


def test_constants_default_requires_all_three_and_ignores_unused_npz(app, tmp_path):
    window = MainWindow()
    assert window.source_combo.currentData() == "constants"
    assert window.model_file_row.isHidden()
    assert window.checkbox_fixed_roi.isChecked()
    assert not window.calculate_button.isEnabled()
    select_three_files(window, tmp_path)
    window.model_file = str(tmp_path / "removed_model.npz")
    window.check_files_selected()
    assert window.calculate_button.isEnabled()
    window.detector_file = None
    window.invalidate_results()
    assert not window.calculate_button.isEnabled()
    window.close()


def test_source_change_invalidates_results_and_updates_roi_default(app, tmp_path):
    window = MainWindow()
    select_three_files(window, tmp_path)
    window.results = {"previous": True}
    window.save_button.setEnabled(True)
    window.source_combo.setCurrentIndex(window.source_combo.findData("prepared"))
    assert not window.model_file_row.isHidden()
    assert window.results is None
    assert not window.save_button.isEnabled()
    assert not window.calculate_button.isEnabled()
    assert not window.checkbox_fixed_roi.isChecked()
    window.source_combo.setCurrentIndex(window.source_combo.findData("constants"))
    assert window.model_file_row.isHidden()
    assert window.calculate_button.isEnabled()
    assert window.checkbox_fixed_roi.isChecked()
    window.close()


def test_constants_worker_receives_physical_options_and_locks_settings(app, tmp_path, monkeypatch):
    class Worker:
        def __init__(self, model_file, **options):
            self.model_file, self.options = model_file, options
            self.progress = self.message = self.result_ready = self.error = SimpleNamespace(connect=lambda _: None)

        def start(self):
            self.started = True

        def isRunning(self):
            return False

    monkeypatch.setattr(project, "ProcessingThread", Worker)
    window = MainWindow()
    select_three_files(window, tmp_path)
    window.model_file = "unused.npz"
    window.calculate()
    assert window.thread.started
    assert window.thread.model_file is None
    assert window.thread.options["radon_mode"] == "legacy"
    assert window.thread.options["radon_h"] is None
    assert window.thread.options["use_fixed_roi"] is True
    assert not window.source_combo.isEnabled()
    assert not window.radon_button.isEnabled()
    assert not window.checkbox_fixed_roi.isEnabled()
    window.close()


def test_radon_dialog_requires_all_calibrated_coefficients(app):
    dialog = RadonCalibrationDialog("legacy", None)
    assert dialog.values() == ("legacy", None)
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("calibrated"))
    with pytest.raises(ValueError, match="все три"):
        dialog.values()
    for key, text in (("sample", "1,2e-4"), ("gasbk", "2e-4"), ("sample_memory", "3e-4")):
        dialog.fields[key].setText(text)
    assert dialog.values() == ("calibrated", {"sample": 1.2e-4, "gasbk": 2e-4, "sample_memory": 3e-4})
    dialog.fields["gasbk"].setText("nan")
    with pytest.raises(ValueError, match="больше нуля"):
        dialog.values()
    dialog.close()
