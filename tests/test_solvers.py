"""Регрессии математической постановки и численных границ статьи XeMat."""
import numpy as np
import pytest

from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


def physical(M, x, b=None, n=None):
    M = np.asarray(M, dtype=float)
    return JointModel(M, x, np.zeros(len(x)) if b is None else b,
                      M.shape[1] if n is None else n, coordinates="a_c_gamma")


@pytest.mark.parametrize("method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_original_zero_counts_are_preserved(method):
    model = physical(np.eye(3), [0, 2, 0])
    result = fit(model, method)
    assert result.success
    np.testing.assert_array_equal(model.x, [0, 2, 0])
    np.testing.assert_allclose(result.mu, [0, 2, 0], atol=1e-9)
    if "poisson" in method:
        assert np.all(result.mu >= 0)
        assert result.covariance is None


def test_wls_zero_variance_replacement_and_background_subtraction():
    M = np.array([[1, 0], [1, 1], [0, 2], [2, 1]], dtype=float)
    x, b = np.array([0, 3, 5, 7]), np.array([1, 1, 2, 1])
    W = np.diag([1, 1 / 3, 1 / 5, 1 / 7])
    expected_covariance = np.linalg.inv(M.T @ W @ M)
    result = fit(physical(M, x, b), "wls")
    np.testing.assert_allclose(result.theta, expected_covariance @ M.T @ W @ (x - b))
    np.testing.assert_allclose(result.covariance, expected_covariance)
    residual = x - result.mu
    assert result.objective == pytest.approx(residual @ W @ residual)


def test_quadratic_methods_share_weights_and_criterion():
    model = physical([[1], [1], [1]], [0, 2, 4], [1, 1, 1])
    free, positive = fit(model, "wls"), fit(model, "nnls")
    assert free.theta[0] > 0
    np.testing.assert_allclose(positive.theta, free.theta)
    assert positive.objective == pytest.approx(free.objective)
    assert positive.covariance is None


@pytest.mark.parametrize("method,expected", [("nnls", 1.5), ("poisson_nonnegative", 2.0)])
def test_chain_nonnegativity_applies_to_activity_not_beta(method, expected):
    model = JointModel(np.eye(2), [1, 3], [0, 0], 2, T_c=[[1, -1], [0, 1]])
    free = fit(model, "wls" if method == "nnls" else "poisson")
    np.testing.assert_allclose(free.theta, [1, 3])
    assert free.physical_parameters[0] < 0  # beta>=0 недостаточно.
    result = fit(model, method)
    assert result.success
    np.testing.assert_allclose(result.theta, [expected, expected], atol=1e-7)
    np.testing.assert_allclose(result.physical_parameters, [0, expected], atol=1e-7)
    assert np.all(result.physical_parameters >= 0)
    assert result.coordinates == "beta_gamma"
    assert result.covariance is None and result.physical_covariance is None


def test_full_spd_precision_and_covariance_are_equivalent():
    M = np.array([[1, 0], [1, 2], [3, 1]], dtype=float)
    covariance = np.array([[2, .3, .1], [.3, 4, .2], [.1, .2, 3]])
    W = np.linalg.inv(covariance)
    model = physical(M, [1, 6, 8], [1, 1, 1])
    result = fit(model, observation_covariance=covariance)
    precision_result = fit(model, weights=W)
    expected_covariance = np.linalg.inv(M.T @ W @ M)
    np.testing.assert_allclose(result.theta, expected_covariance @ M.T @ W @ (model.x - model.b))
    np.testing.assert_allclose(result.covariance, expected_covariance)
    np.testing.assert_allclose(result.theta, precision_result.theta)
    np.testing.assert_allclose(result.covariance, precision_result.covariance)


@pytest.mark.parametrize("method", ["wls", "nnls"])
def test_supplied_vector_weights_override_default(method):
    model = physical([[1], [1]], [1, 9])
    result = fit(model, method, weights=[3, 1])
    np.testing.assert_allclose(result.theta, [3])
    assert result.objective == pytest.approx(3 * (1 - 3) ** 2 + (9 - 3) ** 2)


@pytest.mark.parametrize("kwargs", [
    {"weights": [0, 1]}, {"weights": [1, -1]},
    {"weights": [[1, 2], [2, 1]]},
    {"observation_covariance": [[1, 1], [1, 1]]},
    {"observation_covariance": [[1, .1], [.2, 1]]},
    {"weights": [1, 1], "observation_covariance": np.eye(2)},
])
def test_invalid_uncertainty_input_is_rejected(kwargs):
    with pytest.raises(ValueError):
        fit(physical(np.eye(2), [1, 2]), **kwargs)


def test_poisson_free_allows_negative_parameters_without_boxes():
    # Миллионная граница старого решателя ограничивала эти допустимые оценки.
    model = physical(np.eye(2) * 1e-8, [2, 5], [7, 1])
    result = fit(model, "poisson")
    assert result.success
    np.testing.assert_allclose(result.theta, [-5e8, 4e8], rtol=1e-9)
    np.testing.assert_allclose(result.mu, [2, 5])
    np.testing.assert_allclose(result.covariance, np.diag([2e16, 5e16]), rtol=1e-8)
    assert result.objective == pytest.approx(7 - 2 * np.log(2) - 5 * np.log(5))


def test_background_enters_poisson_mean_not_counts():
    model = physical(np.eye(2), [0, 5], [3, 1])
    result = fit(model, "poisson")
    assert result.success
    np.testing.assert_allclose(result.theta, [-3, 4], atol=1e-8)
    np.testing.assert_array_equal(model.x, [0, 5])
    assert np.all(result.mu >= 0)
    assert result.covariance is None


def test_nonnegative_poisson_retains_background_at_boundary():
    model = physical(np.eye(2), [0, 5], [3, 1])
    result = fit(model, "poisson_nonnegative")
    assert result.success
    np.testing.assert_allclose(result.physical_parameters, [0, 4], atol=1e-7)
    np.testing.assert_allclose(result.mu, [3, 5], atol=1e-7)
    assert np.all(result.physical_parameters >= 0)


def test_canonical_physical_results_do_not_acquire_roundtrip_negatives():
    model = JointModel(np.eye(4), [10, 0, 3, 0], np.full(4, 2.), 2,
                       T_c=[[.1, -.05], [0, .2]])
    for method in ("nnls", "poisson_nonnegative"):
        result = fit(model, method)
        assert result.success and np.all(result.physical_parameters >= 0)
        np.testing.assert_allclose(model.from_physical(result.physical_parameters), result.theta, atol=1e-12)
        np.testing.assert_allclose(result.mu, model.M @ result.theta + model.b)


@pytest.mark.parametrize("method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_rank_deficiency_never_produces_finite_covariance(method):
    model = physical([[1, 1], [2, 2]], [4, 8], [0, 0])
    result = fit(model, method)
    assert result.success and result.rank == 1
    assert result.covariance is None and result.physical_covariance is None
    assert np.isinf(result.condition_number)


def test_rank_deficient_signed_cone_keeps_nullspace_directions():
    result = fit(physical([[1, -1]], [0], [1]), "poisson_nonnegative")
    assert result.success
    assert np.all(result.physical_parameters >= 0)
    assert np.all(result.mu >= 0)
    np.testing.assert_allclose(result.mu, [0], atol=1e-7)


@pytest.mark.parametrize("method", ["poisson", "poisson_nonnegative"])
def test_impossible_positive_observation_is_rejected(method):
    with pytest.raises(ValueError, match="Положительные|допустим"):
        fit(physical([[0]], [1], [0]), method)


@pytest.mark.parametrize("method", ["nnls", "poisson_nonnegative"])
def test_physical_constraints_require_explicit_conversion(method):
    model = JointModel(np.eye(2), [1, 2], [0, 0], 1)
    with pytest.raises(ValueError, match="T_c"):
        fit(model, method)
    assert fit(model, "wls").physical_parameters is None


@pytest.mark.parametrize("method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_overlap_and_extreme_column_units_do_not_change_fitted_spectrum(method):
    base = np.array([[1, 1], [2, 2.00001], [3, 3.00003], [4, 4.00006]])
    scales = np.array([1e-8, 1e8])
    unscaled = fit(physical(base, [5, 10, 16, 20], np.ones(4)), method)
    scaled = fit(physical(base * scales, [5, 10, 16, 20], np.ones(4)), method)
    assert scaled.success and unscaled.success
    assert np.all(np.isfinite(scaled.theta))
    np.testing.assert_allclose(scaled.mu, unscaled.mu, rtol=1e-6, atol=1e-6)
    if method in ("nnls", "poisson_nonnegative"):
        assert np.all(scaled.physical_parameters >= 0)


def test_scaled_saturated_zero_bins_are_feasible_in_original_model():
    rng = np.random.default_rng(736)
    for _ in range(25):
        M = rng.uniform(.05, 1, (6, 6)) * np.geomspace(1e-6, 1e6, 6)
        b, x = rng.uniform(.2, 3, 6), rng.poisson(2, 6)
        x[0] = 0
        model = physical(M, x, b, n=3)
        result = fit(model, "poisson", max_iter=250)
        assert result.success, result.warnings
        assert np.all(result.mu >= 0) and np.all(result.mu[x > 0] > 0)
        np.testing.assert_allclose(result.mu, x, atol=1e-6)
        np.testing.assert_allclose(result.mu, model.M @ result.theta + b, atol=0)
        assert result.covariance is None


def test_joint_overlap_preserves_nonnegative_physical_parameters():
    rng = np.random.default_rng(1489)
    for gap in (1e-3, 1e-6):
        matrices = [rng.uniform(.1, 1, (7, 5)) for _ in range(3)]
        for R in matrices:
            R[:, 2] = R[:, 1] + gap * rng.uniform(-1, 1, 7)
        M = build_joint_matrix(*matrices, np.diag(np.linspace(.3, .7, 5)))
        M *= np.geomspace(1e-5, 1e5, 10)
        model = physical(M, rng.poisson(2, 14), np.ones(14), n=5)
        result = fit(model, "poisson_nonnegative", max_iter=300)
        assert result.success, result.warnings
        assert np.all(result.physical_parameters >= 0) and np.all(result.mu > 0)


@pytest.mark.parametrize("alias,canonical", [("ls", "wls"), ("mle_free", "poisson"),
                                            ("mle", "poisson_nonnegative")])
def test_legacy_method_aliases(alias, canonical):
    result = fit(physical(np.eye(2), [2, 3]), alias)
    assert result.method == canonical and result.success


def test_poisson_rejects_external_quadratic_weights():
    with pytest.raises(ValueError, match="не принимает веса"):
        fit(physical([[1]], [2]), "poisson", weights=[1])
