"""Independent analytic regressions for the four estimators.

References use the mathematical criteria rather than private solver helpers.
In particular, a zero *count* remains zero while only its WLS variance is
replaced by one; and constrained estimators constrain a_c, not beta.
"""

import numpy as np
import pytest

from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


METHODS = ("wls", "nnls", "poisson", "poisson_nonnegative")


def physical_model(M, x, b=None, n_sample=None):
    M = np.asarray(M, dtype=float)
    return JointModel(
        M, x, np.zeros(M.shape[0]) if b is None else b,
        M.shape[1] if n_sample is None else n_sample,
        coordinates="a_c_gamma",
    )


def poisson_objective(x, mu):
    """-log L without the data-only factorial, including zero counts."""
    x, mu = np.asarray(x), np.asarray(mu)
    positive = x > 0
    return np.sum(mu) - np.dot(x[positive], np.log(mu[positive]))


@pytest.mark.parametrize("input_form", ("vector", "matrix", "covariance"))
def test_external_wls_weighting_matches_generalized_least_squares(input_form):
    M = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    x, b = np.array([5, 4, 7, 12]), np.array([1, 2, 0, 1])
    if input_form == "vector":
        w = np.array([0.5, 0.25, 1.0, 2.0])
        W, arguments = np.diag(w), {"weights": w}
    else:
        Omega = np.array([
            [2, .3, .1, 0], [.3, 3, 0, .2],
            [.1, 0, 4, .4], [0, .2, .4, 5],
        ])
        W = np.linalg.inv(Omega)
        arguments = ({"weights": W} if input_form == "matrix"
                     else {"observation_covariance": Omega})
    information = M.T @ W @ M
    expected = np.linalg.solve(information, M.T @ W @ (x - b))
    residual = x - b - M @ expected
    result = fit(physical_model(M, x, b), **arguments)
    assert result.success
    np.testing.assert_allclose(result.theta, expected, rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(information), rtol=2e-13)
    assert result.objective == pytest.approx(residual @ W @ residual, rel=1e-13)
    # No reduced-chi-square rescaling of the specified observation covariance.
    np.testing.assert_allclose(result.physical_covariance, result.covariance)


@pytest.mark.parametrize("arguments", (
    {}, {"weights": np.array([1., .25, 1 / 9])},
    {"observation_covariance": np.diag([1., 4., 9.])},
))
def test_wls_nnls_share_zero_count_weights_and_interior_criterion(arguments):
    # The one-parameter interior solution is sum(w*x)/sum(w)=72/49.
    # 1/(x+1), clipping x itself, or counting zero as one gives another answer.
    model = physical_model(np.ones((3, 1)), [0, 4, 9])
    expected = 72 / 49
    w = np.array([1., .25, 1 / 9])
    loss = np.dot(w, (model.x - expected) ** 2)
    free = fit(model, "wls", **arguments)
    constrained = fit(model, "nnls", **arguments)
    for result in (free, constrained):
        assert result.success
        np.testing.assert_allclose(result.theta, [expected], rtol=1e-13)
        assert result.objective == pytest.approx(loss, rel=1e-13)
    np.testing.assert_array_equal(model.x, [0, 4, 9])
    assert free.covariance[0, 0] == pytest.approx(36 / 49)
    assert constrained.covariance is None


@pytest.mark.parametrize("method", ("nnls", "poisson_nonnegative"))
def test_physical_constraint_is_not_nonnegative_beta(method):
    # All beta are positive at the unrestricted optimum [1,2], but a_c[0]=-1.
    # The physical constraint is beta[0]>=beta[1], with optimum [1.5,1.5].
    T_c = np.array([[1., -1.], [0., 1.]])
    model = JointModel(np.eye(2), [1, 2], [0, 0], 2, T_c=T_c)
    arguments = {"weights": np.ones(2)} if method == "nnls" else {}
    result = fit(model, method, **arguments)
    assert result.success
    assert result.coordinates == "beta_gamma"
    np.testing.assert_allclose(result.theta, [1.5, 1.5], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result.physical_parameters, [0, 1.5], rtol=1e-7, atol=1e-8)
    assert np.all(result.physical_parameters >= 0)
    np.testing.assert_allclose(result.mu, model.M @ result.theta + model.b)
    assert result.covariance is None
    assert result.physical_covariance is None


@pytest.mark.parametrize("diagonal", ((1., 1.), (-1e-8, 1e8), (1e8, -1e-8)))
def test_free_poisson_signed_parameters_zero_boundary_and_column_scales(diagonal):
    M = np.diag(diagonal)
    model = physical_model(M, [0, 3], [2, 1])
    result = fit(model, "poisson")
    assert result.success, result.message
    np.testing.assert_allclose(result.theta, np.array([-2., 2.]) / diagonal,
                               rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(result.mu, [0, 3], atol=1e-11)
    assert np.all(result.mu >= 0)
    assert result.mu[1] > 0
    assert result.objective == pytest.approx(3 - 3 * np.log(3), abs=1e-10)
    assert result.covariance is None  # Boundary mean: regular Fisher variance is invalid.


def test_nonnegative_poisson_keeps_background_and_raw_zero_observation():
    model = physical_model(np.eye(2), [0, 3], [2, 1])
    free = fit(model, "poisson")
    physical = fit(model, "poisson_nonnegative")
    assert free.success and physical.success
    np.testing.assert_allclose(physical.theta, [0, 2], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(physical.mu, [2, 3], rtol=1e-9, atol=1e-10)
    assert physical.objective == pytest.approx(poisson_objective(model.x, physical.mu))
    assert physical.objective - free.objective == pytest.approx(2.)
    np.testing.assert_array_equal(model.x, [0, 3])
    assert physical.covariance is None


@pytest.mark.parametrize("method", ("poisson", "poisson_nonnegative"))
@pytest.mark.parametrize("M,x", (([[0], [0]], [1, 0]), ([[1], [-1]], [1, 1])))
def test_infeasible_poisson_means_are_reported(method, M, x):
    with pytest.raises(ValueError):
        fit(physical_model(M, x), method)


def test_fractional_counts_rejected_before_poisson_optimization():
    with pytest.raises(ValueError, match="целочисленные"):
        physical_model([[1]], [1.25])


@pytest.mark.parametrize("method", ("wls", "poisson"))
def test_full_rank_covariance_is_expected_information_inverse(method):
    M = np.array([[1., 0], [0, 2], [1, 1]])
    model = physical_model(M, [3, 4, 5])
    result = fit(model, method)
    assert result.success
    assert result.rank == 2
    np.testing.assert_allclose(result.theta, [3, 2], rtol=1e-10)
    information = M.T @ np.diag(1 / model.x) @ M
    np.testing.assert_allclose(result.covariance, np.linalg.inv(information), rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_rank_deficiency_does_not_produce_spurious_covariance(method):
    M = np.array([[1., 2], [2, 4], [3, 6]])
    result = fit(physical_model(M, [3, 6, 9]), method)
    assert result.success, result.message
    assert result.rank == 1
    assert result.condition_number == np.inf
    assert result.covariance is None
    assert result.physical_covariance is None
    np.testing.assert_allclose(result.mu, [3, 6, 9], rtol=1e-9)
    assert any("ранг" in warning for warning in result.warnings)


@pytest.mark.parametrize("method", METHODS)
def test_joint_distinct_sensitivities_strong_overlap_and_extreme_scales(method):
    # Near-collinear sample components, 16 decades between column units,
    # and independently calibrated SAMPLE-memory and GASBK sensitivities.
    eps = 1e-4
    R_c = np.array([[1., 1 + eps], [2, 2 - eps], [3, 3 + 2 * eps]])
    R_c = R_c @ np.diag([1e-8, 1e8])
    R_g1, R_g2, F = np.array([[1.], [3.], [2.]]), np.array([[4.], [2.]]), [[.2]]
    M = build_joint_matrix(R_c, R_g1, R_g2, F)
    theta = np.array([1e12, 2e-4, 5000.])
    b = np.array([3., 2., 1., 4., 2.])
    x = np.array([31005, 63000, 92005, 20004, 10002])
    np.testing.assert_allclose(M @ theta + b, x, rtol=1e-14)
    np.testing.assert_array_equal(M[3:, :2], 0)
    np.testing.assert_allclose(M[:3, 2], [.2, .6, .4])
    np.testing.assert_allclose(M[3:, 2], [4, 2])
    result = fit(physical_model(M, x, b, n_sample=2), method)
    assert result.success, (result.message, result.warnings)
    assert result.rank == 3
    assert result.condition_number > 1e4
    np.testing.assert_allclose(result.theta, theta, rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(result.mu, x, rtol=1e-10, atol=5e-6)
    np.testing.assert_allclose(result.physical_parameters, result.theta, rtol=2e-13)


def test_observation_weight_inputs_are_unambiguous_and_positive_definite():
    model = physical_model(np.eye(2), [1, 2])
    with pytest.raises(ValueError):
        fit(model, weights=[1, 1], observation_covariance=np.eye(2))
    for arguments in (
        {"weights": [0, 1]}, {"weights": [1, -1]},
        {"weights": [[1, 2], [2, 1]]},
        {"observation_covariance": [[1, 1], [1, 1]]},
        {"observation_covariance": [[1, .1], [.2, 1]]},
    ):
        with pytest.raises(ValueError):
            fit(model, **arguments)
    for method in ("poisson", "poisson_nonnegative"):
        with pytest.raises(ValueError):
            fit(model, method, weights=[1, 1])
