"""V86: отсчёты DETBK — третье наблюдение, zeta — отдельные параметры."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from xemat.calibration import build_physical_matrices
from xemat.activities import compute_activities
from xemat.io import jsonable, load_prepared_model
from xemat.model import JointModel, build_joint_matrix


@pytest.fixture
def blocks():
    return {
        "R_c": np.array([[1., 0], [0, 1], [.5, .2]]),
        "R_g1": np.array([[2., 0], [0, 3], [.4, .8]]),
        "R_g2": np.array([[1., 0], [0, 1], [.1, .6]]),
        "F": np.array([[.5, .1], [0, .25]]),
        "kappa_s": .2,
        "kappa_g": .3,
    }


def _model(blocks, **overrides):
    arguments = dict(M=build_joint_matrix(**blocks), x=[24, 26, 21, 23, 28, 28, 50, 60, 70],
                     b=np.zeros(9), n_sample_parameters=2, T_c=[[2., -.4], [0, 3]],
                     coordinates="beta_gamma_zeta", n_detector_parameters=3)
    arguments.update(overrides)
    return JointModel(**arguments)


def test_joint_expectation_matches_three_independent_measurements(blocks):
    model = _model(blocks)
    theta = np.array([4., 6, 8, 10, 50, 60, 70])
    # Independent arithmetic: fresh + transferred memory + κ zeta;
    # GASBK memory + κ zeta; finally the unscaled mean DETBK itself.
    assert_allclose(model.M @ theta, [24, 25.5, 21.2, 23, 28, 27.8, 50, 60, 70])
    assert_array_equal(model.M[6:], np.c_[np.zeros((3, 4)), np.eye(3)])
    assert model.n_memory_parameters == 2
    assert model.n_detector_parameters == 3
    assert model.has_detector_background
    assert model.input_coordinate_label == "[beta; gamma; zeta]"
    assert model.physical_coordinate_label == "[a_V; gamma; zeta]"
    assert not model.is_activity_coordinates
    assert model.M.shape[0] - np.linalg.matrix_rank(model.M) == 2


def test_nondiagonal_activity_transform_leaves_memory_and_detbk_unchanged(blocks):
    model = _model(blocks)
    theta = np.array([4., 6, 8, 10, 50, 60, 70])
    physical = model.to_physical(theta)
    assert_allclose(physical, [5.6, 18, 8, 10, 50, 60, 70])
    assert_array_equal(model.T_V, [[2, -.4], [0, 3]])
    assert_allclose(model.from_physical(physical), theta)
    assert_allclose(model.physical_matrix @ physical, model.M @ theta)
    # beta >= 0 is insufficient for a_V >= 0 in a decay chain.
    assert model.to_physical([1, 10, 8, 10, 50, 60, 70])[0] == -2
    physical_model = _model(blocks, M=model.physical_matrix,
                            coordinates="a_V_gamma_zeta", T_c=None)
    assert physical_model.is_activity_coordinates
    assert physical_model.coordinate_transform.shape == (7, 7)
    assert_array_equal(physical_model.to_physical(physical), physical)
    assert_array_equal(physical_model.physical_matrix, model.physical_matrix)


@pytest.mark.parametrize("field,value", [
    ("n_detector_parameters", 0), ("n_detector_parameters", 2),
    ("n_detector_parameters", -1), ("n_detector_parameters", 3.5),
    ("n_detector_parameters", True), ("n_detector_parameters", np.inf),
    ("n_detector_parameters", "3"), ("coordinates", "beta_gamma"),
    ("coordinates", "a_c_gamma"), ("b", np.ones(9)),
    ("x", [24, 26, 21, 23, 28, 28, 50.5, 60, 70]),
])
def test_reject_ambiguous_detector_coordinates_and_observations(blocks, field, value):
    with pytest.raises(ValueError):
        _model(blocks, **{field: value})


@pytest.mark.parametrize("row,column,value", [
    (6, 0, 1),  # DETBK cannot contain fresh activity.
    (6, 2, 1),  # DETBK cannot contain gas memory.
    (6, 4, .2),  # zeta is in DETBK counts, not rate or SAMPLE counts.
    (3, 0, 1),  # GASBK precedes fresh sample.
    (0, 4, -.2),  # Live time factors must be positive.
    (0, 5, .1),  # Each detector parameter belongs to its own ROI.
    (1, 5, .4),  # One common live time ratio, not one ratio per ROI.
])
def test_reject_non_article_detector_blocks(blocks, row, column, value):
    matrix = build_joint_matrix(**blocks)
    matrix[row, column] = value
    with pytest.raises(ValueError):
        _model(blocks, M=matrix)


@pytest.mark.parametrize("value", [None, 0, -1, np.nan, np.inf, True, [.2]])
def test_both_live_time_ratios_must_be_positive_finite_scalars(blocks, value):
    with pytest.raises(ValueError, match="kappa_s"):
        build_joint_matrix(**{**blocks, "kappa_s": value})


def test_joint_response_requires_same_roi_and_isotope_counts(blocks):
    with pytest.raises(ValueError, match="k × n"):
        build_joint_matrix(**{**blocks, "R_g2": blocks["R_g2"][:2]})
    old = build_joint_matrix(*(blocks[name] for name in ("R_c", "R_g1", "R_g2", "F")))
    assert old.shape == (6, 4)
    model = JointModel(old, [1] * 6, [2] * 6, 2, np.eye(2))
    assert not model.has_detector_background
    assert model.n_memory_parameters == 2
    assert model.physical_coordinate_label == "[a_c; gamma]"


@pytest.mark.parametrize("from_blocks", [False, True])
@pytest.mark.parametrize("explicit_detector_count", [False, True])
def test_prepared_joint_npz_supports_T_V_and_zero_offset_default(
    blocks, tmp_path, from_blocks, explicit_detector_count,
):
    expected = _model(blocks)
    data = dict(blocks) if from_blocks else {"M": expected.M}
    data.update(x=expected.x, n_sample_parameters=2, coordinates="beta_gamma_zeta",
                T_V=expected.T_V)
    if explicit_detector_count:
        data["n_detector_parameters"] = 3
    path = tmp_path / "joint.npz"
    np.savez(path, **data)
    actual, extras = load_prepared_model(path)
    assert_array_equal(actual.M, expected.M)
    assert_array_equal(actual.x, expected.x)
    assert_array_equal(actual.b, np.zeros(9))
    assert actual.n_detector_parameters == 3
    assert_array_equal(actual.T_V, expected.T_V)
    assert_array_equal(extras["T_c"], extras["T_V"])
    replacement, _ = load_prepared_model(path, x=[0] * 9)
    assert_array_equal(replacement.x, np.zeros(9))


def test_prepared_physical_coordinates_preserve_detbk_block(blocks, tmp_path):
    expected = _model(blocks)
    path = tmp_path / "physical.npz"
    np.savez(path, M=expected.physical_matrix, x=expected.x,
             coordinates="a_V_gamma_zeta", n_sample_parameters=2)
    actual, _ = load_prepared_model(path)
    assert actual.is_activity_coordinates
    assert_array_equal(actual.M, actual.physical_matrix)
    assert_array_equal(actual.M[6:, 4:], np.eye(3))


@pytest.mark.parametrize("updates,remove", [
    ({"T_c": np.eye(2)}, None),  # Conflicting aliases cannot silently override.
    ({"b": np.ones(9)}, None),  # New b is zero additive offset, never DETBK data.
    ({"n_detector_parameters": 3.0}, None),
    ({"coordinates": "a_V_gamma_zeta"}, None),  # Raw R blocks are beta coords.
    ({"coordinates": "beta_gamma"}, None),
    ({}, "kappa_g"),
])
def test_prepared_joint_npz_rejects_ambiguous_schema(blocks, tmp_path, updates, remove):
    data = dict(blocks, x=np.ones(9), n_sample_parameters=2,
                coordinates="beta_gamma_zeta", T_V=[[2, -.4], [0, 3]])
    data.update(updates)
    if remove:
        data.pop(remove)
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        load_prepared_model(path)


@pytest.fixture
def measurement_metadata():
    start = datetime(2026, 10, 14, 12)
    sample = dict(detector_id=221, start_time=start,
                  collection_start=start - timedelta(hours=20),
                  collection_end=start - timedelta(hours=4),
                  real_time=7200., live_time=5400., xenon_volume=.174,
                  efficiency={2: .18, 3: .37, 5: .24, 6: .12})
    gasbk = dict(detector_id=221, start_time=start - timedelta(hours=6),
                 real_time=10800., live_time=9000.,
                 efficiency={2: .27, 3: .15, 5: .31, 6: .40})
    return sample, gasbk


def test_calibration_uses_live_times_and_preserves_physical_response(measurement_metadata):
    old = build_physical_matrices(*measurement_metadata, radon_mode="legacy")
    new = build_physical_matrices(*measurement_metadata, radon_mode="legacy",
                                   detector_live_time=100000.)
    assert new["M"].shape == (21, 17)
    assert_array_equal(new["M"][:14, :10], old["M"])
    assert_array_equal(new["M"][:7, 10:], .054 * np.eye(7))
    assert_array_equal(new["M"][7:14, 10:], .09 * np.eye(7))
    assert_array_equal(new["M"][14:], np.c_[np.zeros((7, 10)), np.eye(7)])
    for key in ("T_c", "T_V", "G_s", "G_a", "H", "F", "C_s_c", "C_s_g", "C_g_g"):
        assert_array_equal(new[key], old[key])
    assert_array_equal(new["G_a"], new["G_s"])
    assert new["coordinates"] == "beta_gamma_zeta"
    assert new["n_detector_parameters"] == 7
    assert new["metadata"]["time_parameters"]["t_bl"] == 100000
    assert new["metadata"]["observation_blocks"] == ["SAMPLE", "GASBK", "DETBK"]
    assert_array_equal(new["parameter_names"][10:],
                       ["zeta:ROI1", "zeta:ROI2", "zeta:ROI3", "zeta:ROI5",
                        "zeta:ROI6", "zeta:ROI7", "zeta:ROI8"])
    model = JointModel(new["M"], np.zeros(21), np.zeros(21), 5, new["T_V"],
                       new["coordinates"], new["n_detector_parameters"])
    assert model.M.shape[0] - np.linalg.matrix_rank(model.M) == 4


@pytest.mark.parametrize("time", [0, -1, np.nan, np.inf, True])
def test_calibration_rejects_invalid_detector_live_time(measurement_metadata, time):
    with pytest.raises(ValueError, match="DETBK: живое время"):
        build_physical_matrices(*measurement_metadata, radon_mode="legacy", detector_live_time=time)


def test_prepared_G_a_alias_feeds_existing_activity_api(blocks, tmp_path):
    model = _model(blocks)
    data = dict(M=model.M, x=model.x, n_sample_parameters=2,
                coordinates="beta_gamma_zeta", T_V=model.T_V, H=np.eye(2),
                Gamma=[[1., -.2], [0, 1]], G_p=np.eye(2), G_a=np.diag([2., 3]), V0=1.)
    path = tmp_path / "aliases.npz"
    np.savez(path, **data)
    _, extras = load_prepared_model(path)
    assert_array_equal(extras["G_s"], extras["G_a"])
    result = compute_activities([4., 6], np.eye(2), **{
        key: extras[key] for key in ("H", "T_c", "Gamma", "G_p", "G_s", "V0")
    })
    assert_allclose(result.a_V, [5.6, 18.])
    for public, previous in (("a_V", "a_c"), ("T_V", "T_c"),
                             ("K_A_V_stat", "K_A_c_stat"), ("K_A_V_sys", "K_A_c_sys"),
                             ("K_A_V", "K_A_c"), ("std_A_V", "std_A_c"),
                             ("correlation_A_V", "correlation_A_c")):
        assert getattr(result, public) is getattr(result, previous)
        with pytest.raises(AttributeError):
            setattr(result, public, None)
    # The original JSON schema remains compatible: aliases are properties,
    # not duplicate dataclass fields that might diverge after serialization.
    saved = jsonable(result)
    assert "a_c" in saved and "a_V" not in saved


@pytest.mark.parametrize("old", [np.eye(2), np.eye(3), np.full((2, 2), np.nan)])
def test_prepared_conflicting_G_a_G_s_are_rejected(blocks, tmp_path, old):
    model = _model(blocks)
    path = tmp_path / "conflicting_aliases.npz"
    np.savez(path, M=model.M, x=model.x, n_sample_parameters=2,
             coordinates="beta_gamma_zeta", T_V=model.T_V,
             G_a=np.diag([2., 3]), G_s=old)
    with pytest.raises(ValueError, match="G_s и G_a"):
        load_prepared_model(path)


def test_activity_alias_preserves_missing_volume_uncertainty():
    result = compute_activities([2], H=[[1.]])
    assert result.a_V is None
    assert result.T_V is None
    assert result.K_A_V is None
    assert result.std_A_V is None
