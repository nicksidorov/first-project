"""Путь три настоящих IMS → constants.py → четыре решателя и экспорт."""

from datetime import datetime, timedelta
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from constants import FIXED_ROI_TABLE, Rs, Rg
from process_measurement import METHOD_NAMES, process_prepared_measurement
from results_saver import format_results
from xemat.io import save_json


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
    return paths, np.r_[sample_counts, gas_counts], np.r_[1.5 * detector_counts, 2.5 * detector_counts]


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_constants_three_spectra_all_methods_without_npz(constant_spectra, method):
    paths, counts, background = constant_spectra
    result = process_prepared_measurement(None, method=method, radon_mode="legacy", **paths)
    model, solution = result["model"], result["fit"]
    assert solution.success
    assert model.M.shape == (14, 10)
    assert model.coordinates == "beta_gamma"
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
    assert "mu = M @ theta + b" in report
    assert "constants.py" in report
    assert "Rn" in report


def test_constants_requires_all_three_spectra(constant_spectra):
    paths, _, _ = constant_spectra
    with pytest.raises(ValueError):
        process_prepared_measurement(None, sample_file=paths["sample_file"], radon_mode="legacy")


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


def test_constants_rejects_files_from_different_detectors(constant_spectra):
    from pathlib import Path

    paths, _, _ = constant_spectra
    file = Path(paths["detector_file"])
    file.write_text(file.read_text(encoding="utf-8").replace("MIX58_221", "MIX58_222"), encoding="utf-8")
    with pytest.raises(ValueError, match="детектор"):
        process_prepared_measurement(None, radon_mode="legacy", **paths)


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
