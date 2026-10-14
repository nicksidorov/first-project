"""Проверки пользовательского пути: NPZ / три IMS → ядро → окно / экспорт."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

from ims_parser import parse_ims_spectral_file
from process_measurement import METHOD_NAMES, process_prepared_measurement
from project import MainWindow, ProcessingThread
from results_saver import format_results, save_results_to_file
from xemat.io import save_json


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def prepared(tmp_path):
    path = tmp_path / "model.npz"
    np.savez(
        path, M=np.eye(4), x=[10, 0, 3, 0], b=[2, 2, 2, 2],
        n_sample_parameters=2, coordinates="beta_gamma",
        T_c=[[.1, -.05], [0, .2]], H=np.diag([.1, .2]),
        isotope_names=["Xe133", "Xe133m"], mdc_component_indices=[0, 1],
    )
    return path


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_all_four_methods_npz_to_exports_and_html(prepared, tmp_path, app, method):
    result = process_prepared_measurement(prepared, method=method)
    solution = result["fit"]
    assert solution.success
    assert solution.method == method
    np.testing.assert_array_equal(result["model"].x, [10, 0, 3, 0])
    np.testing.assert_allclose(solution.mu, result["model"].M @ solution.theta + result["model"].b)
    assert result["activities"].a_c is not None
    assert result["detection"] is not None
    assert np.all(result["detection"].converged)
    if method in ("nnls", "poisson_nonnegative"):
        assert np.min(solution.physical_parameters) >= -1e-10
        assert solution.covariance is None
        assert result["activities"].std_A_c is None
    if method.startswith("poisson"):
        assert np.all(solution.mu >= -1e-10)
        assert np.all(solution.mu[result["model"].x > 0] > 0)
    text = format_results(result)
    assert "mu = M @ theta + b" in text
    assert "ФИЗИЧЕСКИЕ КООРДИНАТЫ z=[a_c;gamma]" in text
    save_results_to_file(result, tmp_path / "report.txt")
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == text
    save_json(result, tmp_path / "report.json")
    raw_json = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    assert json.loads(raw_json)["fit"]["method"] == method
    window = MainWindow()
    html = window.generate_activities_html(result)
    assert METHOD_NAMES[method] in html.replace("&gt;", ">").replace("&lt;", "<")
    window.on_processing_finished(result)
    assert window.save_button.isEnabled()
    assert "Xe133" in window.results_output.toPlainText()
    window.close()


def _ims_file(path, counts, live_time, hour):
    """Минимальный настоящий IMS с двумерной 256×256 гистограммой."""
    histogram = np.zeros((256, 256), dtype=int)
    histogram[1, 1] = counts[0]
    histogram[3, 3] = counts[1]
    header = (
        "#Header 3\nIMS MIX58_221\n"
        f"#Acquisition\n2026/10/14 {hour:02}:00:00 {live_time} {live_time}\n"
        "#g_Energy\n0 0 0\n127 127 0\n255 255 0\n"
        "#b_Energy\n0 C 0 0\n127 C 127 0\n255 C 255 0\n"
        "#ROI_Limits\n1 1 1 1 1\n2 3 3 3 3\n"
        "#Histogram\n256 256\n"
    )
    path.write_text(header + "\n".join(" ".join(str(value) for value in row) for row in histogram) + "\nSTOP\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def spectra(tmp_path):
    sample = _ims_file(tmp_path / "sample.phd", [0, 8], 10, 12)
    gasbk = _ims_file(tmp_path / "gasbk.phd", [3, 1], 20, 10)
    detbk = _ims_file(tmp_path / "detbk.phd", [2, 4], 100, 9)
    return {"sample_file": sample, "background_file": gasbk, "detector_file": detbk}


@pytest.fixture
def spectral_model(tmp_path):
    path = tmp_path / "spectral_model.npz"
    # Специально другие x,b: проверяется замена данными из выбранных IMS.
    np.savez(path, M=np.eye(4), x=np.full(4, 99), b=np.full(4, 99),
             n_sample_parameters=2, coordinates="a_c_gamma", roi_numbers=[1, 2])
    return path


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_real_parser_three_spectra_raw_counts_fixed_background(spectral_model, spectra, method):
    result = process_prepared_measurement(spectral_model, method=method, **spectra)
    np.testing.assert_array_equal(result["model"].x, [0, 8, 3, 1])
    np.testing.assert_allclose(result["model"].b, [.2, .4, .4, .8])
    assert result["metadata"]["sample_roi_counts"] == {1: 0, 2: 8}
    assert result["metadata"]["detector_info"]["sample"]["id"] == 221
    assert "фиксированный фон" in result["metadata"]["background_source"]


def test_partial_spectra_and_ambiguous_roi_order_rejected(prepared, spectra):
    with pytest.raises(ValueError, match="все три"):
        process_prepared_measurement(prepared, sample_file=spectra["sample_file"])
    with pytest.raises(ValueError, match="sample_roi_numbers"):
        process_prepared_measurement(prepared, **spectra)


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_overlapping_inclusive_roi_endpoint_rejected(spectral_model, spectra, method):
    for path in spectra.values():
        from pathlib import Path
        file = Path(path)
        # ROI имеют общий канал (2,2), даже если в нём ноль наблюдений.
        source = file.read_text(encoding="utf-8")
        file.write_text(source.replace("1 1 1 1 1\n2 3 3 3 3", "1 1 2 1 2\n2 2 3 2 3"), encoding="utf-8")
    with pytest.raises(ValueError, match="пересекаются"):
        process_prepared_measurement(spectral_model, method=method, **spectra)


def test_parser_metadata_and_integer_histogram_preserved(spectra):
    data = parse_ims_spectral_file(spectra["sample_file"], is_sample=True)
    assert data["histogram"].shape == (256, 256)
    assert data["histogram"].sum() == 8
    assert data["live_time"] == 10
    assert data["detector_name"] == "MIX58_221"
    assert data["roi_numbers"] == [1, 2]


def test_weight_metadata_is_not_applied_to_poisson(prepared, tmp_path):
    with np.load(prepared) as archive:
        data = {key: archive[key] for key in archive.files}
    path = tmp_path / "weighted.npz"
    np.savez(path, **data, weights=np.array([2, 3, 4, 5]))
    result = process_prepared_measurement(path, method="poisson")
    assert result["fit"].success
    assert any("не применяются" in message for message in result["warnings"])


def test_t_c_only_computes_concentration_without_detector_activity(prepared, tmp_path, app):
    with np.load(prepared) as archive:
        data = {key: archive[key] for key in archive.files if key != "H"}
    path = tmp_path / "concentration_only.npz"
    np.savez(path, **data)
    result = process_prepared_measurement(path, method="wls")
    assert result["activities"].a is None
    assert result["activities"].a_c is not None
    assert "a=не вычислена" in format_results(result)
    window = MainWindow()
    assert "Xe133" in window.generate_activities_html(result)
    window.close()


def test_invalid_optional_activity_map_keeps_valid_fit_and_diagnostic(prepared, tmp_path):
    with np.load(prepared) as archive:
        data = {key: archive[key] for key in archive.files}
    data["H"] = np.ones((1, 1))
    path = tmp_path / "invalid_activity_map.npz"
    np.savez(path, **data)
    result = process_prepared_measurement(path, method="wls")
    assert result["fit"].success
    assert result["activities"] is None
    assert result["detection"] is not None
    assert any("Активности недоступны" in message for message in result["warnings"])
    assert "Активности недоступны" in format_results(result)


def test_radon_note_and_mdc_exclusion(prepared, tmp_path, app):
    with np.load(prepared) as archive:
        data = {key: archive[key] for key in archive.files}
    data["isotope_names"] = np.array(["Rn222", "Xe133"])
    path = tmp_path / "radon.npz"
    np.savez(path, **data)
    result = process_prepared_measurement(path, method="wls")
    assert result["detection"] is None
    assert any("Rn" in message for message in result["warnings"])
    assert "не объёмная активность радона" in format_results(result)
    window = MainWindow()
    assert "вспомогательный параметр" in window.generate_activities_html(result)
    window.close()


def test_gui_model_selection_and_stale_results(prepared, app, monkeypatch):
    window = MainWindow()
    assert not window.calculate_button.isEnabled()
    assert window.method_combo.count() == 4
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(prepared), ""))
    window.select_model_file()
    assert window.calculate_button.isEnabled()
    result = process_prepared_measurement(prepared, method="nnls")
    window.on_processing_finished(result)
    assert window.results is result
    assert window.save_button.isEnabled()
    window.method_combo.setCurrentIndex(1)
    assert window.results is None
    assert not window.save_button.isEnabled()
    window.sample_file = str(prepared)
    window.invalidate_results()
    assert not window.calculate_button.isEnabled()
    window.clear_spectra()
    assert window.calculate_button.isEnabled()
    window.close()


def test_background_thread_returns_result(prepared, app):
    worker = ProcessingThread(str(prepared), method="nnls")
    results, errors = [], []
    worker.result_ready.connect(results.append)
    worker.error.connect(errors.append)
    worker.start()
    assert worker.wait(5000)
    app.processEvents()
    assert not errors
    assert len(results) == 1 and results[0]["fit"].success
