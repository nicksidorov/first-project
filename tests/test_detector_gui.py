"""Несогласованные детекторы останавливают GUI и не оставляют старый результат."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

import project
from project import MainWindow, ProcessingThread
from xemat.detectors import DetectorSelectionError, validate_detector_ids


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("detector_id", [221, 222])
def test_detector_id_normalizes_integer_and_string_metadata(detector_id):
    records = {
        "SAMPLE": {"detector_id": np.int64(detector_id)},
        "GASBK": {"detector_id": str(detector_id)},
        "DETBK": {"detector_id": f" {detector_id} "},
    }
    assert validate_detector_ids(records) == detector_id


@pytest.mark.parametrize("detector_id", [None, True, 221.5, "221x", 999])
def test_no_common_invalid_detector_is_silently_assigned_a_calibration(detector_id):
    records = {role: {"detector_id": detector_id} for role in ("SAMPLE", "GASBK", "DETBK")}
    with pytest.raises(DetectorSelectionError):
        validate_detector_ids(records)


def test_worker_emits_detector_warning_without_result_or_generic_error(app, monkeypatch):
    message = "Номера детекторов не совпадают: SAMPLE: 221; GASBK: 222; DETBK: 221."

    def fail(*args, **kwargs):
        raise DetectorSelectionError(message)

    monkeypatch.setattr(project, "process_prepared_measurement", fail)
    worker = ProcessingThread(None)
    warnings, results, errors = [], [], []
    worker.warning.connect(warnings.append)
    worker.result_ready.connect(results.append)
    worker.error.connect(errors.append)
    worker.start()
    assert worker.wait(5000)
    app.processEvents()
    assert warnings == [message]
    assert not results
    assert not errors


def test_detector_warning_clears_stale_results_and_restores_file_selection(app, tmp_path, monkeypatch):
    warnings, errors = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[2]))
    window = MainWindow()
    for attribute in ("sample_file", "background_file", "detector_file"):
        file = tmp_path / (attribute + ".phd")
        file.touch()
        setattr(window, attribute, str(file))
    window.results = {"previous": True}
    window.results_output.setHtml("Старый результат")
    window.results_output.setVisible(True)
    window.text_output.setText("Старый протокол")
    window.save_button.setEnabled(True)
    window._set_busy(True)
    message = "SAMPLE: 221; GASBK: 222; DETBK: 221. Выберите согласованный комплект."
    window.on_processing_warning(message)
    assert warnings == [message]
    assert not errors
    assert window.results is None
    assert window.results_output.isHidden()
    assert not window.text_output.toPlainText()
    assert not window.save_button.isEnabled()
    assert window.file_group.isEnabled()
    assert window.calculate_button.isEnabled()
    assert window.progress_bar.isHidden()
    assert "детектор" in window.message_label.text()
    window.close()
