"""Проверки пользовательского пути: NPZ / три IMS → ядро → окно / экспорт."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtGui import QTextCursor, QTextDocument
from PyQt5.QtWidgets import QApplication, QFileDialog

from ims_parser import parse_ims_spectral_file
from process_measurement import METHOD_NAMES, process_prepared_measurement
from project import MainWindow, ProcessingThread
from results_saver import format_results, save_results_to_file
from xemat.io import save_json
from xemat.detectors import DetectorSelectionError


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
    assert result["fit_quality"].label in text
    assert "ν≤0" in text
    save_results_to_file(result, tmp_path / "report.txt")
    assert (tmp_path / "report.txt").read_text(encoding="utf-8") == text
    save_json(result, tmp_path / "report.json")
    raw_json = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    assert json.loads(raw_json)["fit"]["method"] == method
    assert json.loads(raw_json)["fit_quality"]["reduced_statistic"] is None
    window = MainWindow()
    html = window.generate_activities_html(result)
    assert METHOD_NAMES[method] in html.replace("&gt;", ">").replace("&lt;", "<")
    assert result["fit_quality"].label in html
    assert "ν≤0" in html
    window.on_processing_finished(result)
    assert window.save_button.isEnabled()
    assert "Xe133" in window.results_output.toPlainText()
    window.close()


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_reduced_statistic_in_report_window_and_json(tmp_path, app, method):
    path = tmp_path / "overdetermined.npz"
    np.savez(path, M=np.ones((3, 1)), x=[0, 4, 8], b=[1, 1, 1],
             n_sample_parameters=1, coordinates="a_c_gamma")
    result = process_prepared_measurement(path, method=method)
    if method.startswith("poisson"):
        expected = 8 * np.log(2)
        label = "Приведённый девианс D/ν"
    else:
        mean = 16 / 11  # Взвешенная средняя x при весах [1, 1/4, 1/8].
        expected = np.sum((np.array([0, 4, 8]) - mean)**2 / [1, 4, 8]) / 2
        label = "Приведённый χ²/ν"
    assert result["fit_quality"].reduced_statistic == pytest.approx(expected)
    text = format_results(result)
    value_line = next(line for line in text.splitlines() if label in line)
    assert float(value_line.split(":")[-1]) == pytest.approx(expected, rel=1e-6)
    window = MainWindow()
    window.on_processing_finished(result)
    assert f"{label}: {expected:.5g}" in window.results_output.toPlainText()
    if method in ("nnls", "poisson_nonnegative"):
        assert "номинальное" in window.results_output.toPlainText()
    save_json(result, tmp_path / "quality.json")
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["fit_quality"]["reduced_statistic"] == pytest.approx(expected)
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


@pytest.mark.parametrize("file_key", ["sample_file", "background_file", "detector_file"])
@pytest.mark.parametrize("replacement", ["MIX58_222", "MIX58_999", "UNKNOWN"],
                         ids=["mismatch", "unknown", "missing"])
def test_prepared_model_still_requires_consistent_spectral_detectors(
    spectral_model, spectra, file_key, replacement,
):
    file = Path(spectra[file_key])
    file.write_text(file.read_text(encoding="utf-8").replace("MIX58_221", replacement), encoding="utf-8")
    with pytest.raises(DetectorSelectionError) as error:
        process_prepared_measurement(spectral_model, **spectra)
    for role in ("SAMPLE", "GASBK", "DETBK"):
        assert role in str(error.value)


def test_prepared_model_accepts_matching_222_spectra_without_replacing_npz_matrix(spectral_model, spectra):
    for filename in spectra.values():
        file = Path(filename)
        file.write_text(file.read_text(encoding="utf-8").replace("MIX58_221", "MIX58_222"), encoding="utf-8")
    result = process_prepared_measurement(spectral_model, **spectra)
    assert result["fit"].success
    np.testing.assert_array_equal(result["model"].M, np.eye(4))
    assert result["metadata"]["detector_info"]["sample"]["id"] == 222


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
    assert "Активность в детекторе не рассчитана" in format_results(result)
    window = MainWindow()
    assert "Xe133" in window.generate_activities_html(result)
    window.close()


@pytest.mark.parametrize("shape", [(1, 1), (1, 2, 2)])
def test_invalid_optional_activity_map_keeps_valid_fit_and_diagnostic(prepared, tmp_path, shape):
    with np.load(prepared) as archive:
        data = {key: archive[key] for key in archive.files}
    data["H"] = np.ones(shape)
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
    html = window.generate_activities_html(result)
    assert "вспомогательный параметр" in html
    assert html.count(">Rn 222</td>") == 2
    assert "(вспом.)" not in html
    window.close()


@pytest.mark.parametrize(
    "include_detection, converged, detected, highlighted",
    [
        (True, [True, True], [True, False], "Xe133m"),
        (True, [False, True], [True, False], None),
        (True, [True, True], [False, False], None),
        (False, [True, True], [True, False], None),
    ],
    ids=["permuted-components", "nonconverged", "not-detected", "no-detection"],
)
def test_activity_rows_use_core_detection_in_both_rendered_tables(
    prepared, app, include_detection, converged, detected, highlighted,
):
    result = process_prepared_measurement(prepared, method="nnls")
    # Порядок порогов отличается от порядка изотопов в таблицах.
    # Свободная оценка Xe133m находится между L_C и MDC, а отображаемая
    # ограниченная оценка равна нулю. Xe133, наоборот, показан большим.
    # Подсветка должна следовать решению ядра, не величинам в таблице.
    result["activities"] = replace(result["activities"], a=np.array([100., 0.]), a_c=np.array([100., 0.]))
    result["detection"] = replace(
        result["detection"],
        component_indices=np.array([1, 0]),
        free_activity=np.array([2., 0.]),
        L_C=np.array([1., 1.]), MDC=np.array([3., 3.]),
        converged=np.array(converged), detected=np.array(detected),
    ) if include_detection else None
    window = MainWindow()
    window.results_output.setHtml(window.generate_activities_html(result))
    document = window.results_output.document()
    for name in ("Xe133", "Xe133m"):
        cursor, activity_rows = QTextCursor(document), 0
        while True:
            cursor = document.find(name, cursor, QTextDocument.FindWholeWords)
            if cursor.isNull():
                break
            table = cursor.currentTable()
            # У двух таблиц активностей четыре столбца, у таблицы порогов — пять.
            if table is None or table.columns() != 4:
                continue
            activity_rows += 1
            row = table.cellAt(cursor).row()
            for column in range(table.columns()):
                color = table.cellAt(row, column).format().background().color().name()
                assert (color == "#fde2e2") == (name == highlighted)
        assert activity_rows == 2
    window.close()


def test_gui_option_changes_keep_previous_result_until_calculate(prepared, tmp_path, app, monkeypatch):
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
    old_html = window.results_output.toHtml()
    old_text = window.text_output.toPlainText()
    window.method_combo.setCurrentIndex(window.method_combo.findData("wls"))
    for change_roi in (False, True):
        if change_roi:
            window.checkbox_fixed_roi.setChecked(not window.checkbox_fixed_roi.isChecked())
        assert window.results is result
        assert window.save_button.isEnabled()
        assert not window.results_output.isHidden()
        assert window.results_output.toHtml() == old_html
        assert window.text_output.toPlainText() == old_text
        assert "предыдущ" in window.statusBar().currentMessage().lower()

    saved = tmp_path / "previous_result.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(saved), ""))
    window.save_results()
    assert saved.read_text(encoding="utf-8") == format_results(result)
    assert window.results["fit"].method == "nnls"

    # Не запускаем реальный поток: проверяем состояние окна в момент старта.
    started = []
    monkeypatch.setattr(ProcessingThread, "start", lambda self: started.append(self))
    window.calculate()
    assert started == [window.thread]
    assert window.thread.options["method"] == "wls"
    assert window.thread.options["use_fixed_roi"] is True
    assert window.results is None
    assert not window.save_button.isEnabled()
    assert window.results_output.isHidden()
    assert not window.text_output.toPlainText()

    updated = process_prepared_measurement(prepared, method="wls", use_fixed_roi=True)
    window.on_processing_finished(updated)
    assert window.results is updated
    assert not window.results_output.isHidden()
    assert window.save_button.isEnabled()
    assert window.text_output.toPlainText() == format_results(updated)
    window.close()


def test_gui_source_and_file_selection_still_clear_results(prepared, spectra, app, monkeypatch):
    window = MainWindow()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(prepared), ""))
    window.select_model_file()
    result = process_prepared_measurement(prepared, method="nnls")
    window.on_processing_finished(result)
    window.source_combo.setCurrentIndex(window.source_combo.findData("constants"))
    assert window.results is None
    assert window.results_output.isHidden()
    assert not window.save_button.isEnabled()
    assert not window.text_output.toPlainText()

    window.select_model_file()
    window.on_processing_finished(result)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (spectra["sample_file"], ""))
    window.select_sample_file()
    assert window.results is None
    assert window.results_output.isHidden()
    assert not window.save_button.isEnabled()
    assert not window.text_output.toPlainText()
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
