"""Аналитические регрессии ММП на границах средних и физических параметров.

Матрицы синтетические: частные спектры пользователя здесь не используются.
Эталоны следуют из насыщенного пуассоновского критерия либо из явного
одномерного максимума на активной границе, без вызова оптимизатора.
"""

import numpy as np
import pytest

from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


def _poisson_nll(counts, means):
    positive = counts > 0
    return float(np.sum(means) - np.dot(counts[positive], np.log(means[positive])))


@pytest.mark.parametrize("scales", [
    np.ones(6), np.array([1e6, 1e-3, 1., .1, 1e-4, .01]),
])
def test_saturated_free_poisson_reaches_zero_mean_despite_mixed_column_scales(scales):
    # В насыщенной модели глобальный максимум имеет mu=x. Первый нулевой
    # mu возникает из вычитания больших компонент, а не нулевого столбца.
    # Старый решатель на масштабированных столбцах останавливался с
    # mu=x+0.000388... и невязкой ККТ 0.529, несмотря на сообщение gtol.
    response = np.array([
        [6, 11, 2, 1, 2, 2], [5, 5, 5, 11, 3, 8],
        [1, 3, 6, 2, 5, 8], [10, 8, 4, 1, 9, 4],
        [4, 8, 8, 8, 3, 4], [4, 2, 7, 3, 5, 4],
    ], dtype=float)
    counts = np.array([0., 791., 3., 4., 217., 805.])
    model = JointModel(response * scales, counts, np.zeros(6), 6,
                       coordinates="a_c_gamma")
    expected_scaled_parameters = np.linalg.solve(response, counts)
    result = fit(model, "poisson")
    assert result.success, (result.message, result.warnings)
    assert np.all(result.mu >= 0)
    np.testing.assert_allclose(result.theta * scales, expected_scaled_parameters, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.mu, counts, rtol=2e-8, atol=1e-7)
    assert result.objective == pytest.approx(_poisson_nll(counts, counts), abs=2e-7)
    assert result.covariance is None  # Регулярная информация на mu[0]=0 неприменима.

    # Независимый сертификат ККТ: в нормированных исходных координатах
    # градиент равен нормали единственной активной границы, lambda=1>0.
    residual = np.ones(6)
    residual[1:] -= counts[1:] / result.mu[1:]
    np.testing.assert_allclose(response.T @ residual, response[0], atol=2e-7)


@pytest.mark.parametrize("scales", [np.ones(3), np.array([1e5, .1, 1e6])])
def test_joint_sample_gasbk_detbk_free_poisson_boundary_has_closed_form(scales):
    sample_response = np.array([[4., 7., 2.], [7., 2., 3.], [5., 7., 6.]])
    memory_response = np.array([[1., 7., 4.], [2., 1., 1.], [4., 1., 5.]])
    counts = np.array([61., 20., 35., 0., 16., 3., 580., 238., 543.])
    matrix = build_joint_matrix(
        sample_response * scales, memory_response * scales,
        memory_response * scales, .4 * np.eye(3), kappa_s=.3, kappa_g=.1,
    )
    model = JointModel(matrix, counts, np.zeros(9), 3, np.diag(scales),
                       "beta_gamma_zeta", n_detector_parameters=3)
    # Последовательное исключение трёх квадратных блоков, без оптимизации.
    zeta = counts[6:]
    scaled_gamma = np.linalg.solve(memory_response, counts[3:6] - .1 * zeta)
    scaled_beta = np.linalg.solve(
        sample_response, counts[:3] - .4 * memory_response @ scaled_gamma - .3 * zeta,
    )
    result = fit(model, "poisson")
    assert result.success, (result.message, result.warnings)
    assert np.all(result.mu >= 0)
    np.testing.assert_allclose(result.mu, counts, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.theta[:3] * scales, scaled_beta, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.theta[3:6] * scales, scaled_gamma, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.theta[6:], zeta, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.physical_parameters[:3], scaled_beta, rtol=2e-8, atol=1e-7)
    assert result.objective == pytest.approx(_poisson_nll(counts, counts), abs=2e-7)
    assert result.covariance is None


@pytest.mark.parametrize("method", ["poisson", "poisson_nonnegative"])
@pytest.mark.parametrize("scales", [np.ones(2), np.array([1e-6, 1e6])])
def test_physical_activity_boundary_differs_from_zero_mean_boundary(method, scales):
    # a=[u0-u1;u1], mu=[u0;u1;u0+u1], u=diag(scales)@beta.
    # Свободный ММП: u0=0, u1=(8+24)/2=16, отрицательная a0=-16.
    # a>=0 требует u0>=u1. Максимум на u0=u1=8: mu=[8,8,16].
    # Нулевое наблюдение само по себе НЕ требует нулевого mu.
    response = np.array([[1., 0.], [0., 1.], [1., 1.]])
    transform = np.array([[1., -1.], [0., 1.]])
    counts = np.array([0., 8., 24.])
    model = JointModel(response * scales, counts, np.zeros(3), 2, transform * scales)
    result = fit(model, method)
    assert result.success, (result.message, result.warnings)
    expected_scaled = np.array([0., 16.]) if method == "poisson" else np.array([8., 8.])
    expected_activity = transform @ expected_scaled
    expected_means = response @ expected_scaled
    np.testing.assert_allclose(result.theta * scales, expected_scaled, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.physical_parameters, expected_activity, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(result.mu, expected_means, rtol=2e-8, atol=1e-7)
    assert np.all(result.mu >= 0)
    assert result.objective == pytest.approx(_poisson_nll(counts, expected_means), abs=2e-7)
    if method == "poisson_nonnegative":
        assert np.all(result.physical_parameters >= 0)
        assert result.mu[0] > 0
    else:
        assert result.physical_parameters[0] < 0
    assert result.covariance is None


@pytest.mark.parametrize("scales", [np.ones(3), np.array([1e-8, 1., 1e8])])
def test_nonnegative_poisson_can_have_coincident_parameter_and_mean_boundaries(scales):
    # a0=0 и mu0=0 задают одну и ту же активную грань. Эти два ограничения
    # нельзя считать независимыми уравнениями при уточнении решения.
    response = np.array([
        [1., 0., 0.], [4., 7., 2.], [3., 2., 5.],
        [8., 3., 1.], [1., 6., 4.],
    ])
    expected_scaled = np.array([0., 2., 5.])
    counts = response @ expected_scaled
    model = JointModel(response * scales, counts, np.zeros(5), 3,
                       coordinates="a_c_gamma")
    result = fit(model, "poisson_nonnegative")
    assert result.success, (result.message, result.warnings)
    assert np.all(result.physical_parameters >= 0)
    assert np.all(result.mu >= 0)
    np.testing.assert_allclose(result.theta * scales, expected_scaled, rtol=3e-8, atol=1e-7)
    np.testing.assert_allclose(result.mu, counts, rtol=3e-8, atol=1e-7)
    assert result.objective == pytest.approx(_poisson_nll(counts, counts), abs=2e-7)
    assert result.covariance is None


@pytest.mark.parametrize("method", ["poisson", "poisson_nonnegative"])
def test_zero_observation_with_positive_optimum_is_not_pinned_to_zero(method):
    # f(a,b)=2a+2b-4log(a)-10log(b): a=2,b=5, несмотря на x0=0.
    # Критерий содержит mu0=a+b: убирать строку нулевого наблюдения нельзя.
    response = np.array([[1., 1.], [1., 0.], [0., 1.]])
    model = JointModel(response, [0, 4, 10], np.zeros(3), 2, coordinates="a_c_gamma")
    result = fit(model, method)
    assert result.success, (result.message, result.warnings)
    np.testing.assert_allclose(result.theta, [2., 5.], atol=1e-7)
    np.testing.assert_allclose(result.mu, [7., 2., 5.], atol=1e-7)
    assert result.mu[0] > 0
    if method == "poisson":
        assert result.covariance is not None
