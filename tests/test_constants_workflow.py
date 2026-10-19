"""Путь три настоящих IMS → constants.py → четыре решателя и экспорт."""

from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from constants import DETECTOR_RESPONSE_MATRICES, FIXED_ROI_TABLE, Rs, Rg
from constants import FIXED_EFFICIENCIES_SAMPLE, FIXED_EFFICIENCIES_GASBK
from process_measurement import METHOD_NAMES, process_prepared_measurement
from results_saver import format_results
from ims_parser import parse_ims_spectral_file
from xemat.io import save_json
from xemat.detectors import DetectorSelectionError


ROI_ORDER = [1, 2, 3, 5, 6, 7, 8]


def _write_ims(path, counts, *, start, real_time, live_time, sample=False):
    """Используем физические ROI, но независимый линейный масштаб каналов."""
    histogram = np.zeros((256, 256), dtype=int)
    roi_lines = []
    for number, value in zip(ROI_ORDER, counts):
        beta_min, beta_max, gamma_min, gamma_max = FIXED_ROI_TABLE[number]
        beta_channel = int(round((beta_min + beta_max) / 8))
        gamma_channel = int(round((gamma_min + gamma_max) / 4))
        histogram[gamma_channel, beta_channel] = value
        roi_lines.append(f"{number} {beta_min} {beta_max} {gamma_min} {gamma_max}")

    header = ["#Header 3", "IMS MIX58_221"]
    if sample:
        collection_start = start - timedelta(hours=20)
        collection_end = start - timedelta(hours=4)
        header += [
            "#Collection",
            f"{collection_start:%Y/%m/%d %H:%M:%S} {collection_end:%Y/%m/%d %H:%M:%S} 9999",
            "#Processing", "0.174 0",
        ]
    header += [
        "#Acquisition", f"{start:%Y/%m/%d %H:%M:%S} {real_time} {live_time}",
        "#b-gEfficiency", "Xe135 2 0.27", "Xe133 3 0.15", "Xe131m 5 0.31", "Xe133m 6 0.40",
        "#g_Energy", "0 0 0", "254 127 0", "510 255 0",
        "#b_Energy", "0 C 0 0", "508 C 127 0", "1020 C 255 0",
        "#ROI_Limits", *roi_lines, "#Histogram", "256 256",
    ]
    body = "\n".join(" ".join(str(value) for value in row) for row in histogram)
    path.write_text("\n".join(header) + "\n" + body + "\nSTOP\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def constant_spectra(tmp_path):
    start = datetime(2026, 10, 14, 12)
    sample_counts = np.rint(Rs @ [4000, 10000, 12000, 13000, 11000]
                            + Rg @ [1200, 1600, 1700, 2000, 2200] + 30).astype(int)
    gas_counts = np.rint(Rg @ [2500, 3000, 3500, 4000, 4500] + 50).astype(int)
    detector_counts = np.array([11, 13, 17, 19, 23, 29, 31])
    paths = {
        "sample_file": _write_ims(tmp_path / "sample.phd", sample_counts,
                                  start=start, real_time=7200, live_time=5400, sample=True),
        "background_file": _write_ims(tmp_path / "gasbk.phd", gas_counts,
                                      start=start - timedelta(hours=6), real_time=10800, live_time=9000),
        "detector_file": _write_ims(tmp_path / "detbk.phd", detector_counts,
                                    start=start - timedelta(days=1), real_time=3600, live_time=3600),
    }
    return paths, np.r_[sample_counts, gas_counts, detector_counts], np.zeros(21)


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_constants_three_spectra_all_methods_without_npz(constant_spectra, method):
    paths, counts, background = constant_spectra
    result = process_prepared_measurement(None, method=method, radon_mode="legacy", **paths)
    model, solution = result["model"], result["fit"]
    assert solution.success
    assert model.M.shape == (21, 17)
    assert model.coordinates == "beta_gamma_zeta"
    assert model.n_detector_parameters == 7
    assert_allclose(model.M[:7, -7:], 1.5 * np.eye(7))
    assert_allclose(model.M[7:14, -7:], 2.5 * np.eye(7))
    assert_array_equal(model.M[14:, -7:], np.eye(7))
    assert result["fit_quality"].degrees_of_freedom == 4
    assert_array_equal(model.x, counts)
    assert_allclose(model.b, background)
    assert_allclose(solution.mu, model.M @ solution.theta + model.b)
    assert result["activities"] is not None
    assert np.all(np.isfinite(result["activities"].a_c))
    assert result["detection"] is not None
    assert np.all(result["detection"].converged)
    assert len(result["detection"].MDC) == 4
    if method in ("nnls", "poisson_nonnegative"):
        assert np.min(solution.physical_parameters) >= -1e-9
    report = format_results(result)
    assert "mu = M @ theta" in report
    assert "DETBK ROI 8" in report
    assert "constants.py" in report
    assert "Rn" in report


def test_constants_requires_all_three_spectra(constant_spectra):
    paths, _, _ = constant_spectra
    with pytest.raises(ValueError):
        process_prepared_measurement(None, sample_file=paths["sample_file"], radon_mode="legacy")


def test_ims_efficiency_spectral_suffix_preserves_counts_and_model(constant_spectra):
    from pathlib import Path

    paths, _, _ = constant_spectra
    baseline = process_prepared_measurement(None, method="wls", radon_mode="legacy", **paths)
    labels = {2: "XE135-10_81:30", 3: "XE133-10_81:30",
              5: "XE131M-10_81:30", 6: "XE133M-10_81:30"}
    names = {2: "Xe135", 3: "Xe133", 5: "Xe131m", 6: "Xe133m"}
    for file_key in ("sample_file", "background_file"):
        path = Path(paths[file_key])
        contents = path.read_text(encoding="utf-8")
        for roi, label in labels.items():
            contents = contents.replace(f"{names[roi]} {roi} ", f"{label} {roi} ")
        path.write_text(contents, encoding="utf-8")
        parsed = parse_ims_spectral_file(str(path), is_sample=file_key == "sample_file")
        # Парсер сохраняет полную исходную метку для диагностики.
        assert parsed["roi_to_isotope"] == labels

    result = process_prepared_measurement(None, method="wls", radon_mode="legacy", **paths)
    assert result["fit"].success
    for field in ("x", "b", "M", "T_c"):
        assert_array_equal(getattr(result["model"], field), getattr(baseline["model"], field))
    assert_allclose(result["fit"].theta, baseline["fit"].theta, rtol=0, atol=0)


def test_constants_rejects_roi_geometry_changed_from_calibration(constant_spectra):
    from pathlib import Path

    paths, _, _ = constant_spectra
    # Все файлы согласованы друг с другом, но не с калибровочными Rs/Rg.
    for path in paths.values():
        file = Path(path)
        file.write_text(file.read_text(encoding="utf-8").replace("319.35", "320.35"), encoding="utf-8")
    with pytest.raises(ValueError, match="ROI|границ|калибров"):
        process_prepared_measurement(None, radon_mode="legacy", **paths)


def test_constants_rejects_missing_collection_metadata(constant_spectra):
    from pathlib import Path

    paths, _, _ = constant_spectra
    file = Path(paths["sample_file"])
    file.write_text(file.read_text(encoding="utf-8").replace("#Collection", "#Unknown"), encoding="utf-8")
    with pytest.raises(ValueError, match="collection|пробоотбор|Collection"):
        process_prepared_measurement(None, radon_mode="legacy", **paths)


def test_fixed_roi_uses_calibrated_bounds_and_preserves_raw_counts(constant_spectra):
    from pathlib import Path

    paths, counts, background = constant_spectra
    # ROI1 не задаёт эффективности четырёх изотопов ксенона: её описание
    # можно заменить фиксированными границами без перепривязки #b-gEfficiency.
    for path in paths.values():
        file = Path(path)
        file.write_text(file.read_text(encoding="utf-8").replace("319.35", "320.35"), encoding="utf-8")
    result = process_prepared_measurement(None, radon_mode="legacy", use_fixed_roi=True, **paths)
    assert result["fit"].success
    assert_array_equal(result["model"].x, counts)
    assert_allclose(result["model"].b, background)


@pytest.mark.parametrize("file_key", ["sample_file", "background_file"])
def test_fixed_roi_rejects_efficiency_tied_to_different_reference_roi(constant_spectra, file_key):
    from pathlib import Path

    paths, _, _ = constant_spectra
    file = Path(paths[file_key])
    # Эффективность Xe135 относится к ROI2 исходного файла. Изменение её
    # границ не разрешает переносить тот же epsilon на фиксированную ROI2.
    file.write_text(file.read_text(encoding="utf-8").replace("224.65", "225.65"), encoding="utf-8")
    with pytest.raises(ValueError, match="ROI|эффектив|границ|калибров"):
        process_prepared_measurement(None, radon_mode="legacy", use_fixed_roi=True, **paths)


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_fixed_efficiencies_work_without_file_efficiencies_and_are_exported(
    constant_spectra, monkeypatch, tmp_path, method,
):
    paths, counts, background = constant_spectra
    # Фиксированная калибровка совпадает с ε синтетических файлов.
    eps = [.27, .15, .31, .40]
    monkeypatch.setitem(FIXED_EFFICIENCIES_SAMPLE, 221, eps.copy())
    monkeypatch.setitem(FIXED_EFFICIENCIES_GASBK, 221, eps.copy())
    expected = process_prepared_measurement(method=method, **paths)
    for key in ("sample_file", "background_file"):
        path = Path(paths[key])
        text = path.read_text(encoding="utf-8")
        text = text[:text.index("#b-gEfficiency")] + text[text.index("#g_Energy"):]
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="эффективност"):
        process_prepared_measurement(method=method, **paths)
    actual = process_prepared_measurement(method=method, use_fixed_efficiencies=True, **paths)
    assert actual["use_fixed_efficiencies"] is True
    assert_array_equal(actual["model"].x, counts)
    assert_allclose(actual["model"].b, background)
    assert_allclose(actual["fit"].theta, expected["fit"].theta, rtol=0, atol=0)
    assert_allclose(actual["activities"].a_c, expected["activities"].a_c, rtol=0, atol=0)
    assert_allclose(actual["detection"].MDC, expected["detection"].MDC, rtol=0, atol=0)
    assert actual["metadata"]["sample_efficiencies"] == eps
    assert actual["metadata"]["gasbk_efficiencies"] == eps
    assert actual["calibration_matrices"]["I_c"][0] == pytest.approx(.3 * .3700)
    report = format_results(actual)
    assert "FIXED_EFFICIENCIES_SAMPLE/GASBK[221]" in report
    assert "ε SAMPLE" in report and "ε GASBK" in report
    save_json(actual, tmp_path / "fixed.json")
    saved = json.loads((tmp_path / "fixed.json").read_text(encoding="utf-8"))
    assert saved["use_fixed_efficiencies"] is True
    assert saved["metadata"]["sample_efficiencies"] == eps


def test_fixed_roi_and_efficiencies_replace_file_reference_geometry(constant_spectra, monkeypatch):
    paths, counts, background = constant_spectra
    monkeypatch.setitem(FIXED_EFFICIENCIES_SAMPLE, 221, [.27, .15, .31, .40])
    monkeypatch.setitem(FIXED_EFFICIENCIES_GASBK, 221, [.27, .15, .31, .40])
    for path in paths.values():
        file = Path(path)
        file.write_text(file.read_text(encoding="utf-8").replace("224.65", "225.65"), encoding="utf-8")
    actual = process_prepared_measurement(
        method="wls", use_fixed_roi=True, use_fixed_efficiencies=True, **paths,
    )
    assert_array_equal(actual["model"].x, counts)
    assert_allclose(actual["model"].b, background)
    with pytest.raises(ValueError, match="границы ROI 2"):
        process_prepared_measurement(method="wls", use_fixed_efficiencies=True, **paths)


def test_gui_fixed_efficiencies_preserves_export_until_calculation(constant_spectra, monkeypatch, tmp_path):
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from project import MainWindow, ProcessingThread

    app = QApplication.instance() or QApplication([])
    paths, _, _ = constant_spectra
    monkeypatch.setitem(FIXED_EFFICIENCIES_SAMPLE, 221, [.21, .36, .29, .19])
    monkeypatch.setitem(FIXED_EFFICIENCIES_GASBK, 221, [.33, .25, .47, .38])
    window = MainWindow()
    for name, path in paths.items():
        setattr(window, name, path)
    window.method_combo.setCurrentIndex(window.method_combo.findData("wls"))
    old = process_prepared_measurement(method="wls", use_fixed_roi=True, **paths)
    window.on_processing_finished(old)
    old_html = window.results_output.toHtml()
    old_text = window.text_output.toPlainText()
    displayed = window.results_output.toPlainText()
    assert displayed.index("2. ОБЪЁМНАЯ АКТИВНОСТЬ") < displayed.index("ROI:") < displayed.index("Координаты theta:")

    window.checkbox_fixed_efficiencies.setChecked(True)
    assert window.results is old
    assert window.save_button.isEnabled()
    assert window.results_output.toHtml() == old_html
    assert window.text_output.toPlainText() == old_text
    assert "предыдущий расчёт" in window.message_label.text()
    assert "фиксированные из constants.py" in window.source_hint.text()
    target = tmp_path / "previous.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    window.save_results()
    assert target.read_text(encoding="utf-8") == format_results(old)
    assert "#b-gEfficiency" in target.read_text(encoding="utf-8")
    window.checkbox_fixed_efficiencies.setChecked(False)
    assert "Расчёт завершён" in window.message_label.text()
    window.checkbox_fixed_efficiencies.setChecked(True)

    monkeypatch.setattr(ProcessingThread, "start", lambda self: None)
    window.calculate()
    assert window.thread.options["use_fixed_efficiencies"] is True
    assert window.results is None
    assert not window.checkbox_fixed_efficiencies.isEnabled()
    window.thread.run()
    assert window.results is not None
    assert window.results["use_fixed_efficiencies"] is True
    assert not np.allclose(window.results["activities"].a_c, old["activities"].a_c)
    assert "Эффективности: фиксированные, constants.py" in window.results_output.toPlainText()
    assert "FIXED_EFFICIENCIES_SAMPLE/GASBK[221]" in window.text_output.toPlainText()
    assert window.checkbox_fixed_efficiencies.isEnabled()
    window.close()


@pytest.mark.parametrize("file_key", ["sample_file", "background_file", "detector_file"])
@pytest.mark.parametrize("replacement", ["MIX58_222", "MIX58_999", "UNKNOWN"],
                         ids=["mismatch", "unknown", "missing"])
def test_constants_rejects_incompatible_detectors_with_file_details(constant_spectra, file_key, replacement):
    paths, _, _ = constant_spectra
    file = Path(paths[file_key])
    file.write_text(file.read_text(encoding="utf-8").replace("MIX58_221", replacement), encoding="utf-8")
    with pytest.raises(DetectorSelectionError, match="детектор") as error:
        process_prepared_measurement(None, radon_mode="legacy", **paths)
    for role in ("SAMPLE", "GASBK", "DETBK"):
        assert role in str(error.value)
    assert "221" in str(error.value)
    if replacement != "UNKNOWN":
        assert replacement.split("_")[1] in str(error.value)


def test_three_spectra_select_detector_222_in_fit_and_report(constant_spectra):
    paths, counts, background = constant_spectra
    result_221 = process_prepared_measurement(None, method="wls", radon_mode="legacy", **paths)
    for file_path in paths.values():
        file = Path(file_path)
        contents = file.read_text(encoding="utf-8").replace(
            "IMS MIX58_221", "MIX58 MIX58_222 B CYLINDER          FULL",
        )
        file.write_text(contents, encoding="utf-8")
    result = process_prepared_measurement(None, method="wls", radon_mode="legacy", **paths)
    assert result["fit"].success
    assert result["metadata"]["calibration_detector_id"] == 222
    for role in ("sample", "background", "detector"):
        assert result["metadata"]["detector_info"][role]["id"] == 222
    assert_array_equal(result["calibration_matrices"]["R_c0"], DETECTOR_RESPONSE_MATRICES[222][0])
    assert_array_equal(result["calibration_matrices"]["R_g0"], DETECTOR_RESPONSE_MATRICES[222][1])
    assert not np.array_equal(result["model"].M, result_221["model"].M)
    # Замена калибровки не меняет наблюдения, фон или временной пересчёт активностей.
    assert_array_equal(result["model"].x, counts)
    assert_allclose(result["model"].b, background)
    assert_array_equal(result["model"].T_c, result_221["model"].T_c)
    report = format_results(result)
    assert "Rs_222" in report and "Rg_222" in report


def test_calibrated_radon_scales_survive_export(constant_spectra, tmp_path):
    paths, _, _ = constant_spectra
    coefficients = {"sample": .013, "gasbk": .017, "sample_memory": .019}
    result = process_prepared_measurement(None, method="wls", radon_mode="calibrated",
                                          radon_h=coefficients, **paths)
    path = tmp_path / "calibrated.json"
    save_json(result, path)
    exported = json.loads(path.read_text(encoding="utf-8"))
    assert exported["metadata"]["radon_mode"] == "calibrated"
    assert exported["metadata"]["radon_h"] == coefficients
    assert exported["calibration_matrices"]["H"][0][0] == coefficients["sample"]
    assert exported["calibration_matrices"]["H_g2"][0][0] == coefficients["gasbk"]
    assert exported["calibration_matrices"]["H_g1"][0][0] == coefficients["sample_memory"]
