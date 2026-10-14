import numpy as np
import pytest
from datetime import datetime

from xemat.model import JointModel, build_joint_matrix
from xemat.io import load_prepared_model, save_json


def test_joint_different_responses_and_coordinate_roundtrip():
    M = build_joint_matrix([[2, 1]], [[3, 4]], [[5, 6]], [[.4, .1], [0, .8]])
    np.testing.assert_allclose(M, [[2, 1, 1.2, 3.5], [0, 0, 5, 6]])
    T_c = np.array([[2, -1], [0, 3]])
    model = JointModel(M, [5, 6], [1, 1], 2, T_c)
    theta = np.array([2, 1, 3, 4])
    z = model.to_physical(theta)
    np.testing.assert_allclose(model.from_physical(z), theta)
    np.testing.assert_allclose(model.physical_matrix @ z, M @ theta)


@pytest.mark.parametrize('x', [[-.1], [.1], [np.nan]])
def test_counts_must_be_raw_nonnegative_integers(x):
    with pytest.raises(ValueError):
        JointModel([[1]], x, [0], 1)


def test_npz_load_override_and_strict_metadata(tmp_path):
    p = tmp_path / 'model.npz'
    np.savez(p, M=np.eye(2), x=[1, 2], b=[.1, .2],
             n_sample_parameters=1, coordinates='beta_gamma', T_c=[[2.]])
    model, extras = load_prepared_model(p, x=[3, 4])
    np.testing.assert_equal(model.x, [3, 4])
    assert 'T_c' in extras
    np.savez(p, M=np.eye(2), x=[1, 2], b=[.1, .2],
             n_sample_parameters=1.5, coordinates='beta_gamma')
    with pytest.raises(ValueError):
        load_prepared_model(p)


def test_export_unknown_uncertainty_as_null(tmp_path):
    p = tmp_path / 'result.json'
    save_json({'covariance': None, 'condition': np.inf, 'x': np.array([1, 2]),
               'acquisition': datetime(2026, 10, 14, 12, 0)}, p)
    assert 'NaN' not in p.read_text() and 'Infinity' not in p.read_text()
    assert '2026-10-14T12:00:00' in p.read_text()


def test_response_blocks_are_beta_coordinates(tmp_path):
    p = tmp_path / 'model.npz'
    np.savez(p, R_c=[[1]], R_g1=[[2]], R_g2=[[3]], F=[[.5]],
             x=[2, 3], b=[1, 1], n_sample_parameters=1,
             coordinates='a_c_gamma')
    with pytest.raises(ValueError, match='beta_gamma'):
        load_prepared_model(p)
