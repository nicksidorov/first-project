"""Физическая проверка матриц по интегрированию кинетики Бейтмана.

Эталон не повторяет аналитические формулы реализации: scipy.expm и quad_vec
интегрируют эволюцию активностей при измерении и постоянном пробоотборе.
"""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from scipy.linalg import expm

from constants import DETECTOR_RESPONSE_MATRICES, ROI_INFO, Rs, Rg, XENON_ISOTOPES
from xemat.calibration import build_physical_matrices, decay_correction


ISOTOPES = ("Rn222", "Xe135", "Xe133", "Xe131m", "Xe133m")


@pytest.fixture
def measurement_metadata():
    start = datetime(2026, 10, 14, 12)
    sample = {
        "detector_id": 221,
        "start_time": start,
        "collection_start": start - timedelta(hours=20),
        "collection_end": start - timedelta(hours=4),
        "real_time": 7200.0,
        "live_time": 5400.0,
        "xenon_volume": 0.174,
        # Намеренно отличается от V_Xe/0.087: статья использует именно V_Xe.
        "air_volume": 9999.0,
        "efficiency": {2: .18, 3: .37, 5: .24, 6: .12},
    }
    gasbk = {
        "detector_id": 221,
        "start_time": start - timedelta(hours=6),
        "real_time": 10800.0,
        "live_time": 9000.0,
        "efficiency": {2: .27, 3: .15, 5: .31, 6: .40},
    }
    return sample, gasbk


def _activity_generator():
    decay = np.array([XENON_ISOTOPES[name] for name in ISOTOPES])
    generator = -np.diag(decay)
    # dA_Xe133/dt = -lambda_Xe133 A_Xe133 + lambda_Xe133 A_Xe133m.
    generator[2, 4] = decay[2]
    return generator


def _integrated_evolution(duration):
    generator = _activity_generator()
    return quad_vec(lambda time: expm(generator * time), 0, duration,
                    epsabs=1e-8, epsrel=1e-12)[0]


def _sensitivities(data):
    # Радон здесь не используется: его коэффициент требует отдельной калибровки.
    return np.array([1.0] + [
        data["efficiency"][ROI_INFO[name]["roi"]] * ROI_INFO[name]["br"]
        for name in ISOTOPES[1:]
    ])


def _integrated_counts(data, sensitivities):
    """Counts = live_fraction * efficiency * integral(activity(t)) dt."""
    live_fraction = data["live_time"] / data["real_time"]
    return live_fraction * np.diag(sensitivities) @ _integrated_evolution(data["real_time"])


RADON_H = {"sample": .013, "gasbk": .017, "sample_memory": .019}


def _build(sample, gasbk):
    return build_physical_matrices(sample, gasbk, radon_h=RADON_H)


@pytest.mark.parametrize("detector_id,expected_rs,expected_rg", [
    (221, [
        [1, 0, 0, 0, 0],
        [.65, 1, 0, 0, 0],
        [1.4, .01, 1, .004, 0],
        [.054, 0, .32, 1, .05],
        [.072, .01, .17, .043, 1],
        [.03, .02, .34, .0456, 0],
        [.089, 0, .0576, 0, .05],
    ], [
        [1, 0, 0, 0, 0],
        [.65, 1, 0, 0, 0],
        [1.4, .01, 1, .004, 0],
        [.054, 0, .33, 1, .05],
        [.072, .01, .18, .049, 1],
        [.03, .01, .18, .0456, 0],
        [.089, 0, .0576, 0, .05],
    ]),
    (222, [
        [1, 0, 0, 0, 0],
        [.65, 1, 0, 0, 0],
        [1.33, .01, 1, .004, 0],
        [.054, 0, .33, 1, .05],
        [.072, .01, .17, .04, 1],
        [.03, .02, .347, .045, 0],
        [.089, 0, .0576, 0, .05],
    ], [
        [1, 0, 0, 0, 0],
        [.65, 1, 0, 0, 0],
        [1.33, .01, 1, .004, 0],
        [.054, 0, .33, 1, .05],
        [.072, .01, .18, .04, 1],
        [.03, .01, .18, .045, 0],
        [.089, 0, .0576, 0, .05],
    ]),
])
def test_detector_response_uses_user_calibration_exactly(
    measurement_metadata, detector_id, expected_rs, expected_rg,
):
    # Эталонные числа взяты из спецификации пользователя, независимо от constants.py.
    sample, gasbk = deepcopy(measurement_metadata)
    sample["detector_id"] = gasbk["detector_id"] = detector_id
    result = _build(sample, gasbk)
    assert_allclose(result["R_c0"], expected_rs, rtol=0, atol=0)
    assert_allclose(result["R_g0"], expected_rg, rtol=0, atol=0)
    assert_allclose(DETECTOR_RESPONSE_MATRICES[detector_id][0], expected_rs, rtol=0, atol=0)
    assert_allclose(DETECTOR_RESPONSE_MATRICES[detector_id][1], expected_rg, rtol=0, atol=0)
    assert result["metadata"]["calibration_detector_id"] == detector_id
    assert_allclose(result["R_c"], np.asarray(expected_rs) @ result["C_s_c"])
    assert_allclose(result["R_g1"], np.asarray(expected_rg) @ result["C_s_g"])
    assert_allclose(result["R_g2"], np.asarray(expected_rg) @ result["C_g_g"])


def test_legacy_aliases_and_branching_probabilities():
    assert Rs is DETECTOR_RESPONSE_MATRICES[221][0]
    assert Rg is DETECTOR_RESPONSE_MATRICES[221][1]
    assert ROI_INFO == {
        "Rn222": {"roi": 1, "br": .3700},
        "Xe135": {"roi": 2, "br": .9020},
        "Xe133": {"roi": 3, "br": .3700},
        "Xe131m": {"roi": 5, "br": .5400},
        "Xe133m": {"roi": 6, "br": .5577},
    }


@pytest.mark.parametrize("side", [0, 1], ids=["sample", "gasbk"])
@pytest.mark.parametrize("detector_id", [None, 999, 222], ids=["missing", "unknown", "mismatch"])
def test_physical_matrices_do_not_guess_detector(measurement_metadata, side, detector_id):
    data = deepcopy(measurement_metadata)
    if detector_id is None:
        del data[side]["detector_id"]
    else:
        data[side]["detector_id"] = detector_id
    with pytest.raises(ValueError, match="детектор"):
        _build(*data)


def test_daughter_counts_and_activity_scale_match_integrated_bateman(measurement_metadata):
    sample, gasbk = measurement_metadata
    result = _build(sample, gasbk)
    memory_in_sample = dict(sample, efficiency=gasbk["efficiency"])

    for data, c_key, h_key, sensitivity_key in (
        (sample, "C_s_c", "H", "I_c"),
        (memory_in_sample, "C_s_g", "H_g1", "I_g"),
        (gasbk, "C_g_g", "H_g2", "I_g"),
    ):
        sensitivities = _sensitivities(data)
        direct_counts = _integrated_counts(data, sensitivities)
        own_counts = np.diag(direct_counts)
        expected_C = direct_counts @ np.diag(1 / own_counts)
        assert_allclose(result[c_key], expected_C, rtol=3e-12, atol=1e-14)
        assert_allclose(np.diag(result[h_key])[1:], 1 / own_counts[1:], rtol=3e-12)
        assert_allclose(result[sensitivity_key][1:], sensitivities[1:])

    # Составы эффективности разные: один общий C или C1=1 исказит дочь.
    assert result["C_s_c"][2, 4] != pytest.approx(result["C_s_g"][2, 4])
    assert_allclose(result["R_c"], result["R_c0"] @ result["C_s_c"])
    assert_allclose(result["R_g1"], result["R_g0"] @ result["C_s_g"])
    assert_allclose(result["R_g2"], result["R_g0"] @ result["C_g_g"])
    # Дочерние отсчёты в опорной ROI изомера не удаляются нормировкой столбца.
    assert result["R_c"][4, 4] > 1


def test_memory_transfer_preserves_activity_evolution(measurement_metadata):
    sample, gasbk = measurement_metadata
    result = _build(sample, gasbk)
    tau = (sample["start_time"] - gasbk["start_time"]).total_seconds()
    activity_evolution = expm(_activity_generator() * tau)
    assert_allclose(result["B"], activity_evolution, rtol=3e-12, atol=1e-15)

    # Начальная смесь — произвольные физические активности в начале GASBK.
    activity_at_gas_start = np.array([1.2, .8, 1.7, 1.1, 2.3])
    gamma = np.linalg.solve(result["H_g2"], activity_at_gas_start)
    activity_at_sample_start = activity_evolution @ activity_at_gas_start
    assert_allclose(result["H_g1"] @ result["F"] @ gamma,
                    activity_at_sample_start, rtol=3e-12)

    # Независимая проверка полного вклада памяти: эволюция, интеграл и Rg.
    memory_in_sample = dict(sample, efficiency=gasbk["efficiency"])
    count_integral = _integrated_counts(memory_in_sample, _sensitivities(gasbk))
    expected_isotope_counts = count_integral @ activity_at_sample_start
    # Rn — отдельный откалиброванный коэффициент, не интеграл ксеноновой модели.
    expected_isotope_counts[0] = activity_at_sample_start[0] / RADON_H["sample_memory"]
    expected_roi_counts = result["R_g0"] @ expected_isotope_counts
    assert_allclose(result["R_g1"] @ result["F"] @ gamma, expected_roi_counts, rtol=3e-12)
    assert result["F"][0, 0] == pytest.approx(
        RADON_H["gasbk"] / RADON_H["sample_memory"] * activity_evolution[0, 0])


def test_collection_and_processing_chain_recovers_air_concentrations(measurement_metadata):
    sample, gasbk = measurement_metadata
    result = _build(sample, gasbk)
    collection_duration = (sample["collection_end"] - sample["collection_start"]).total_seconds()
    processing_duration = (sample["start_time"] - sample["collection_end"]).total_seconds()
    collection_average = _integrated_evolution(collection_duration) / collection_duration
    before_measurement = expm(_activity_generator() * processing_duration) @ collection_average
    # Подача постоянной концентрации с постоянным расходом, удержание дочери.
    true_air = np.array([.7, 1.1, 2.3, .8, 1.9])
    effective_volume = sample["xenon_volume"] / .087
    at_measurement_start = effective_volume * before_measurement @ true_air
    beta = np.linalg.solve(result["H"], at_measurement_start)
    assert result["V0"] == pytest.approx(2.0)
    assert_allclose(result["T_c"] @ beta, true_air, rtol=3e-12)

    expected_air_transform = np.linalg.inv(effective_volume * before_measurement)
    assert_allclose(result["T_c"], expected_air_transform @ result["H"], rtol=3e-12, atol=1e-15)
    diagonal_correction = result["G_s"] @ result["G_p"] / effective_volume
    assert_allclose(result["Gamma"], np.linalg.solve(diagonal_correction, expected_air_transform),
                    rtol=3e-12, atol=1e-15)
    assert result["Gamma"][2, 4] < 0
    assert result["T_c"][2, 4] < 0


def test_live_time_changes_counts_scale_but_not_elapsed_decay(measurement_metadata):
    sample, gasbk = measurement_metadata
    original = _build(sample, gasbk)
    shorter_live_sample = dict(sample, live_time=sample["live_time"] / 2)
    changed = _build(shorter_live_sample, gasbk)
    # Калиброванные Rn H заданы извне; проверяем четыре ксеноновые координаты.
    assert_allclose(changed["H"][1:, 1:], 2 * original["H"][1:, 1:])
    assert_allclose(changed["T_c"][1:, 1:], 2 * original["T_c"][1:, 1:])
    assert_allclose(changed["F"][1:, 1:], .5 * original["F"][1:, 1:])
    for key in ("C_s_c", "C_s_g", "C_g_g", "Gamma", "G_p", "G_s", "B"):
        assert_allclose(changed[key], original[key])


def test_volume_is_extracted_xenon_and_uncertainties_use_numeric_constants(measurement_metadata):
    sample, gasbk = measurement_metadata
    original = _build(sample, gasbk)
    changed = _build(dict(sample, xenon_volume=.348, air_volume=.0001), gasbk)
    assert changed["V0"] == pytest.approx(4)
    assert_allclose(changed["T_c"], original["T_c"] / 2)
    assert_allclose(changed["M"], original["M"])
    assert_allclose(original["delta_I"], np.full(5, .001))
    assert original["delta_V0"] == pytest.approx(.001)
    assert np.array_equal(original["mdc_component_indices"], [1, 2, 3, 4])


def test_radon_approximation_requires_explicit_opt_in(measurement_metadata):
    sample, gasbk = measurement_metadata
    with pytest.raises(ValueError, match="radon_h|калибров"):
        build_physical_matrices(sample, gasbk)
    legacy = build_physical_matrices(sample, gasbk, radon_mode="legacy")
    calibrated = _build(sample, gasbk)
    assert legacy["metadata"]["radon_mode"] == "legacy"
    assert any("0.3" in warning and "legacy" in warning for warning in legacy["warnings"])
    assert calibrated["H"][0, 0] == RADON_H["sample"]
    assert calibrated["H_g1"][0, 0] == RADON_H["sample_memory"]
    assert calibrated["H_g2"][0, 0] == RADON_H["gasbk"]
    with pytest.raises(ValueError):
        build_physical_matrices(sample, gasbk, radon_mode="legacy", radon_h=RADON_H)
    with pytest.raises(ValueError):
        build_physical_matrices(sample, gasbk, radon_h={"sample": 1, "gasbk": 1})


@pytest.mark.parametrize("side,key,value", [
    (0, "real_time", 0), (0, "live_time", 0), (1, "live_time", -1),
    (0, "live_time", 8000), (1, "real_time", float("nan")),
    (0, "xenon_volume", 0), (0, "xenon_volume", None),
    (0, "start_time", None), (1, "start_time", None),
    (0, "collection_start", None), (0, "collection_end", None),
    (0, "efficiency", {2: .2, 3: .3, 5: .4}),
    (1, "efficiency", {2: .2, 3: .3, 5: .4, 6: 0}),
    (0, "efficiency", {2: .2, 3: 1.1, 5: .4, 6: .5}),
    (1, "efficiency", {2: .2, 3: float("nan"), 5: .4, 6: .5}),
])
def test_invalid_measurement_metadata_fails_without_defaults(measurement_metadata, side, key, value):
    data = deepcopy(measurement_metadata)
    data[side][key] = value
    with pytest.raises(ValueError):
        _build(*data)


@pytest.mark.parametrize("problem", ["negative_collection", "unfinished_collection", "overlap_gasbk", "wrong_isotope"])
def test_temporal_order_and_efficiency_identity_are_checked(measurement_metadata, problem):
    sample, gasbk = deepcopy(measurement_metadata)
    if problem == "negative_collection":
        sample["collection_start"] = sample["collection_end"] + timedelta(seconds=1)
    elif problem == "unfinished_collection":
        sample["collection_end"] = sample["start_time"] + timedelta(seconds=1)
    elif problem == "overlap_gasbk":
        gasbk["start_time"] = sample["start_time"] - timedelta(seconds=gasbk["real_time"] - 1)
    else:
        gasbk["roi_to_isotope"] = {3: "XE133M"}
    with pytest.raises(ValueError):
        _build(sample, gasbk)


@pytest.mark.parametrize("side", [0, 1], ids=["sample", "gasbk"])
@pytest.mark.parametrize("identities", [
    {2: "XE135-10_81:30", 3: "XE133-10_81:30", 5: "XE131M-10_81:30", 6: "XE133M-10_81:30"},
    {2: "XE-135", 3: "XE-133", 5: "XE-131M", 6: "XE-133M"},
    {2: "135-XE", 3: "133-XE", 5: "131M-XE", 6: "133M-XE"},
], ids=["spectral_suffix", "element_first", "mass_first"])
def test_efficiency_labels_preserve_physical_matrices(measurement_metadata, side, identities):
    reference = _build(*measurement_metadata)
    data = deepcopy(measurement_metadata)
    data[side]["roi_to_isotope"] = identities
    result = _build(*data)
    for key in ("M", "T_c", "I_c", "I_g", "H", "H_g1", "H_g2"):
        assert_allclose(result[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("side", [0, 1], ids=["sample", "gasbk"])
@pytest.mark.parametrize("roi,identity", [
    (3, "XE133M-10_81:30"), (3, "XE135-10_81:30"), (6, "XE133-10_81:30"),
])
def test_spectral_suffix_does_not_hide_wrong_isotope(measurement_metadata, side, roi, identity):
    data = deepcopy(measurement_metadata)
    data[side]["roi_to_isotope"] = {roi: identity}
    with pytest.raises(ValueError, match="ожидается"):
        _build(*data)


def test_short_exposure_limit_does_not_cancel_daughter_counts(measurement_metadata):
    sample, gasbk = measurement_metadata
    sample = dict(sample, real_time=1e-5, live_time=8e-6)
    result = _build(sample, gasbk)
    counts = _integrated_counts(sample, _sensitivities(sample))
    expected_daughter_ratio = counts[2, 4] / counts[4, 4]
    assert expected_daughter_ratio > 0
    assert result["C_s_c"][2, 4] == pytest.approx(expected_daughter_ratio, rel=1e-11, abs=0)
    assert_allclose(decay_correction([1e-9, 1e-6], 0), [1, 1])


def test_short_collection_and_processing_keep_small_chain_subtraction(measurement_metadata):
    sample, gasbk = measurement_metadata
    sample = dict(sample,
                  collection_start=sample["start_time"] - timedelta(microseconds=2),
                  collection_end=sample["start_time"] - timedelta(microseconds=1))
    result = _build(sample, gasbk)
    average = _integrated_evolution(1e-6) / 1e-6
    to_measurement = expm(_activity_generator() * 1e-6) @ average
    inverse_evolution = np.linalg.inv(to_measurement)
    expected_gamma = inverse_evolution[2, 4] / inverse_evolution[2, 2]
    assert expected_gamma < 0
    assert result["Gamma"][2, 4] == pytest.approx(expected_gamma, rel=1e-11, abs=0)
