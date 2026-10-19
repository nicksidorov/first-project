"""Независимые численные эталоны для совместного оценивания SAMPLE/GASBK/DETBK.

Используются исключение неизвестного фона, насыщенная аналитическая модель
и перебор активных ограничений; эталоны не вызывают решатели XeMat.
"""

from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import block_diag

from xemat.diagnostics import compute_fit_quality
from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


def _joint(Rc, Rg1, Rg2, F, x, ks, kg, transform=None):
    k, n = np.shape(Rc)
    matrix = build_joint_matrix(Rc, Rg1, Rg2, F, kappa_s=ks, kappa_g=kg)
    return JointModel(
        matrix, x, np.zeros(3 * k), n,
        np.eye(n) if transform is None else transform, "beta_gamma_zeta", k,
    )


@pytest.mark.parametrize("correlated_raw_spectra", [False, True])
def test_full_wls_equals_independent_gls_after_eliminating_shared_detector_background(correlated_raw_spectra):
    Rc = np.array([[1., .2], [.1, 1.], [.5, .4]])
    Rg1 = np.array([[1., .1], [.2, 1.], [.3, .6]])
    Rg2 = np.array([[1., .3], [.1, 1.], [.4, .5]])
    F = np.array([[.8, .03], [0., .6]])
    ks, kg, k, n = .2, .12, 3, 2
    model = _joint(Rc, Rg1, Rg2, F, [40, 70, 26, 21, 18, 11, 120, 90, 160], ks, kg)
    rng = np.random.default_rng(8601)
    if correlated_raw_spectra:
        factor = rng.normal(size=(9, 9))
        omega = factor @ factor.T + np.diag(np.arange(1., 10.))
    else:
        blocks = []
        for _ in range(3):
            factor = rng.normal(size=(k, k))
            blocks.append(factor @ factor.T + np.diag([3., 5., 7.]))
        omega = block_diag(*blocks)

    # D M=[A,0]: фон исключается из средних, но остаётся в ковариации.
    T = np.vstack((ks * np.eye(k), kg * np.eye(k)))
    D = np.column_stack((np.eye(2 * k), -T))
    y = D @ model.x
    A = np.block([[Rc, Rg1 @ F], [np.zeros((k, n)), Rg2]])
    corrected_omega = D @ omega @ D.T
    information = A.T @ np.linalg.solve(corrected_omega, A)
    expected_covariance = np.linalg.solve(information, np.eye(2 * n))
    expected_parameters = np.linalg.solve(information, A.T @ np.linalg.solve(corrected_omega, y))
    if not correlated_raw_spectra:
        np.testing.assert_allclose(corrected_omega[:k, k:], ks * kg * omega[2 * k:, 2 * k:])
        assert np.any(corrected_omega[:k, k:] != 0)

    result = fit(model, "wls", observation_covariance=omega)
    assert result.success
    np.testing.assert_allclose(result.theta[:2 * n], expected_parameters, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.covariance[:2 * n, :2 * n], expected_covariance, rtol=3e-12, atol=3e-12)
    residual = y - A @ expected_parameters
    assert result.objective == pytest.approx(residual @ np.linalg.solve(corrected_omega, residual), rel=2e-12)

    # Восстановление профилированной zeta независимо от полного решателя.
    background_columns = model.M[:, 2 * n:]
    W_background = np.linalg.solve(omega, background_columns)
    remainder = model.x - model.M[:, :2 * n] @ expected_parameters
    expected_zeta = np.linalg.solve(background_columns.T @ W_background, W_background.T @ remainder)
    np.testing.assert_allclose(result.theta[2 * n:], expected_zeta, rtol=2e-12, atol=2e-12)
    precision_result = fit(model, "wls", weights=np.linalg.solve(omega, np.eye(3 * k)))
    np.testing.assert_allclose(precision_result.theta, result.theta, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_one_roi_free_estimates_and_all_covariance_blocks_have_closed_form(method):
    c, d, detector = 7., 13., 40.
    ks, kg, transfer = .2, .1, .6
    model = _joint([[1.]], [[1.]], [[1.]], [[transfer]], [c, d, detector], ks, kg, [[2.5]])
    inverse = np.array([[1., -transfer, -ks + transfer * kg], [0., 1., -kg], [0., 0., 1.]])
    expected = np.array([c - transfer * d - (ks - transfer * kg) * detector, d - kg * detector, detector])
    covariance = inverse @ np.diag([c, d, detector]) @ inverse.T
    result = fit(model, method)
    assert result.success
    assert expected[0] < 0  # Свободная проба отрицательна при допустимых средних.
    np.testing.assert_allclose(result.theta, expected, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(result.mu, model.x, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(result.covariance, covariance, rtol=4e-8, atol=2e-8)
    assert result.covariance[0, 0] == pytest.approx(c + transfer**2 * d + (ks - transfer * kg)**2 * detector, rel=4e-8)
    Q = np.diag([2.5, 1., 1.])
    np.testing.assert_allclose(result.physical_covariance, Q @ covariance @ Q.T, rtol=4e-8, atol=2e-8)


@pytest.mark.parametrize("method", ["poisson", "poisson_nonnegative"])
def test_zero_observed_detbk_is_a_likelihood_boundary_without_pseudocount(method):
    model = _joint([[1.]], [[1.]], [[1.]], [[.5]], [8, 3, 0], .2, .1)
    result = fit(model, method)
    assert result.success
    np.testing.assert_array_equal(model.x, [8, 3, 0])
    np.testing.assert_allclose(result.theta, [6.5, 3., 0.], atol=2e-7)
    np.testing.assert_allclose(result.mu, model.x, atol=2e-7)
    assert result.covariance is None
    assert result.objective == pytest.approx(11 - 8 * np.log(8) - 3 * np.log(3), abs=2e-7)


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_long_detbk_limit_recovers_fixed_background_with_analytic_variance_inflation(method):
    c, d, background_rate, transfer = 50., 30., 100., .4
    sample_time, gas_time = .3, .2
    old = JointModel([[1., transfer], [0., 1.]], [c, d], [sample_time * background_rate, gas_time * background_rate], 1, [[1.]])
    fixed = fit(old, method)
    assert fixed.success
    added_variances = []
    for detector_time in (1, 100, 10000):
        ks, kg = sample_time / detector_time, gas_time / detector_time
        model = _joint([[1.]], [[1.]], [[1.]], [[transfer]], [c, d, background_rate * detector_time], ks, kg)
        result = fit(model, method)
        assert result.success
        np.testing.assert_allclose(result.theta[:2], fixed.theta, rtol=3e-7, atol=3e-7)
        response_to_detbk_error = np.array([-ks + transfer * kg, -kg])
        expected_extra = background_rate * detector_time * np.outer(response_to_detbk_error, response_to_detbk_error)
        np.testing.assert_allclose(result.covariance[:2, :2], fixed.covariance + expected_extra, rtol=3e-7, atol=3e-7)
        added_variances.append(result.covariance[0, 0] - fixed.covariance[0, 0])
    assert added_variances[0] > added_variances[1] > added_variances[2] > 0
    assert added_variances[-1] / added_variances[0] == pytest.approx(1e-4, rel=3e-3)


def _enumerated_nonnegative_minimum(A, y):
    """Малый независимый эталон NNLS: все возможные активные границы."""
    best_value, best = float("inf"), None
    for count in range(A.shape[1] + 1):
        for available in combinations(range(A.shape[1]), count):
            z = np.zeros(A.shape[1])
            if available:
                z[list(available)] = np.linalg.lstsq(A[:, available], y, rcond=None)[0]
            if np.any(z < -1e-10):
                continue
            residual = y - A @ z
            value = residual @ residual
            if value < best_value:
                best_value, best = value, z
    return best, best_value


@pytest.mark.parametrize("method", ["nnls", "poisson_nonnegative"])
def test_constraints_are_on_chained_activity_and_memory_and_detector_background(method):
    response = np.array([[1., 0.], [0., 1.], [1., 1.]])
    transform = np.array([[1., -1.], [0., 1.]])
    model = _joint(response, response, response, .3 * np.eye(2), [2, 18, 20, 5, 8, 13, 0, 2, 1], .5, .4, transform)
    free = fit(model, "wls")
    assert np.all(free.theta[:2] > 0)
    assert free.physical_parameters[0] < 0
    assert free.theta[4] < 0  # В WLS знак zeta не ограничен.
    result = fit(model, method)
    assert result.success
    assert np.all(result.physical_parameters >= 0)
    assert result.physical_parameters[0] == pytest.approx(0, abs=2e-8)
    assert result.physical_parameters[4] == pytest.approx(0, abs=2e-8)
    assert result.covariance is None and result.physical_covariance is None
    np.testing.assert_allclose(result.theta[:2], np.linalg.solve(transform, result.physical_parameters[:2]), atol=1e-12)

    S = model.physical_matrix
    if method == "nnls":
        root_variance = np.sqrt(np.maximum(model.x, 1))
        expected, criterion = _enumerated_nonnegative_minimum(S / root_variance[:, None], model.x / root_variance)
        np.testing.assert_allclose(result.physical_parameters, expected, atol=2e-10)
        assert result.objective == pytest.approx(criterion, rel=2e-12)
    else:
        residual = np.ones_like(model.x)
        positive = model.x > 0
        residual[positive] -= model.x[positive] / result.mu[positive]
        gradient = S.T @ residual
        active = result.physical_parameters < 1e-8
        assert np.all(gradient[active] >= -2e-7)
        np.testing.assert_allclose(gradient[~active], 0, atol=2e-7)


@pytest.mark.parametrize("method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_seven_roi_five_component_model_has_four_residual_degrees_of_freedom(method):
    rng = np.random.default_rng(8617)
    response = np.vstack((np.eye(5), np.full((2, 5), .15))) + rng.uniform(.01, .12, size=(7, 5))
    transform = np.eye(5)
    transform[2, 4] = -.2
    provisional = _joint(response, response * 1.1, response * .9, np.eye(5) * .4, np.ones(21), .1, .08, transform)
    truth = np.r_[np.full(5, 50.), np.full(5, 30.), np.full(7, 120.)]
    model = JointModel(provisional.M, rng.poisson(provisional.physical_matrix @ truth), np.zeros(21), 5, transform, "beta_gamma_zeta", 7)
    result = fit(model, method)
    assert result.success
    quality = compute_fit_quality(model, result)
    assert quality.n_observations == 21
    assert quality.rank == 17
    assert quality.degrees_of_freedom == 4
    assert quality.nominal_dof == (method in ("nnls", "poisson_nonnegative"))
    assert quality.reduced_statistic == pytest.approx(quality.statistic / 4)


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_overlapping_components_are_invariant_to_extreme_column_units(method):
    response = np.array([[1., 1.001], [2., 1.999], [3., 3.002]])
    memory = np.array([[1., 1.002], [2., 1.998], [3., 3.001]])
    original = _joint(response, memory, memory, np.eye(2) * .4, [3100, 6020, 9350, 2000, 3910, 6080, 1500, 1100, 1700], .2, .1)
    units = np.array([1e-9, 1e9, 1e7, 1e-7, 1., 1., 1.])
    scaled = JointModel(original.M * units, original.x, np.zeros(9), 2, np.diag(units[:2]), "beta_gamma_zeta", 3)
    reference, result = fit(original, method), fit(scaled, method)
    assert reference.success and result.success
    assert reference.rank == result.rank == 7
    assert reference.condition_number > 1000
    np.testing.assert_allclose(result.mu, reference.mu, rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(result.theta * units, reference.theta, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result.covariance * np.outer(units, units), reference.covariance, rtol=2e-6, atol=2e-6)
    assert result.objective == pytest.approx(reference.objective, rel=3e-9, abs=3e-9)
