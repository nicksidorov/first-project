"""Независимые эталоны приведённых статистик и граничных случаев."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from xemat.diagnostics import compute_fit_quality, poisson_deviance
from xemat.model import JointModel
from xemat.solvers import fit


@pytest.mark.parametrize("method", ["wls", "nnls"])
@pytest.mark.parametrize("weight_kind", ["default", "vector", "matrix", "covariance"])
def test_quadratic_quality_retains_actual_weights(method, weight_kind):
    model = JointModel(np.ones((3, 1)), [0, 4, 9], [0, 0, 0], 1, coordinates="a_c_gamma")
    covariance = np.array([[2., .2, 0.], [.2, 3., .1], [0., .1, 5.]])
    if weight_kind == "default":
        W, options = np.diag([1., 1/4, 1/9]), {}
    elif weight_kind == "vector":
        W, options = np.diag([.5, 2., 3.]), {"weights": [.5, 2., 3.]}
    elif weight_kind == "matrix":
        W = np.linalg.inv(covariance)
        options = {"weights": W}
    else:
        W = np.linalg.inv(covariance)
        options = {"observation_covariance": covariance}
    solution = fit(model, method, **options)
    quality = compute_fit_quality(model, solution)
    # Один общий параметр: аналитическая взвешенная средняя.
    mean = float(np.ones(3) @ W @ model.x / (np.ones(3) @ W @ np.ones(3)))
    residual = model.x - mean
    expected = float(residual @ W @ residual)
    assert quality.statistic == pytest.approx(expected)
    assert quality.degrees_of_freedom == 2
    assert quality.reduced_statistic == pytest.approx(expected / 2)
    assert quality.nominal_dof == (method == "nnls")


@pytest.mark.parametrize("method", ["poisson", "poisson_nonnegative"])
def test_poisson_quality_is_full_deviance_with_raw_zeros_and_background(method):
    model = JointModel(np.ones((3, 1)), [0, 4, 8], [1, 1, 1], 1, coordinates="a_c_gamma")
    solution = fit(model, method)
    quality = compute_fit_quality(model, solution)
    np.testing.assert_allclose(solution.mu, 4, rtol=1e-8)
    assert solution.objective < 0  # NLL без log(x!) не является девиансом.
    assert quality.statistic == pytest.approx(16 * np.log(2))
    assert quality.reduced_statistic == pytest.approx(8 * np.log(2))
    assert quality.n_observations == 3  # Нулевое наблюдение не исключено.
    assert quality.degrees_of_freedom == 2


def test_deviance_zero_contributions_and_domain():
    assert poisson_deviance([0, 2, 0], [3, 2, 0]) == 6
    assert poisson_deviance([0, 0], [0, 0]) == 0
    assert poisson_deviance([1], [0]) == float("inf")
    assert poisson_deviance([0], [-1e-12]) == float("inf")


@pytest.mark.parametrize("mu", [1e12 - 1, 1e12 + 1, 1e12 + 1e8, 1e-300])
def test_deviance_stability_against_decimal_reference(mu):
    # Прямое вычитание больших NLL теряет малый девианс при mu≈x.
    with localcontext() as context:
        context.prec = 80
        x_decimal, mu_decimal = Decimal(10**12), Decimal.from_float(mu)
        expected = float(2 * (x_decimal * (x_decimal / mu_decimal).ln() + mu_decimal - x_decimal))
    assert poisson_deviance([1e12], [mu]) == pytest.approx(expected, rel=1e-11, abs=1e-25)


@pytest.mark.parametrize("method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_saturated_model_has_no_reduced_statistic(method):
    model = JointModel(np.eye(2), [4, 8], [1, 1], 2, coordinates="a_c_gamma")
    quality = compute_fit_quality(model, fit(model, method))
    assert quality.degrees_of_freedom == 0
    assert quality.reduced_statistic is None
    assert any("ν≤0" in note for note in quality.warnings)


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_dof_uses_rank_instead_of_number_of_columns(method):
    model = JointModel(np.ones((3, 2)), [3, 5, 7], [1, 1, 1], 2, coordinates="a_c_gamma")
    quality = compute_fit_quality(model, fit(model, method))
    assert quality.rank == 1
    assert quality.degrees_of_freedom == 2


@pytest.mark.parametrize("method", ["nnls", "poisson_nonnegative"])
def test_constraints_keep_full_model_nominal_dof(method):
    model = JointModel(np.vstack([np.eye(2), np.eye(2)]), [0, 4, 0, 9], [2, 1, 2, 1],
                       2, coordinates="a_c_gamma")
    solution = fit(model, method)
    assert solution.physical_parameters[0] == pytest.approx(0, abs=1e-9)
    quality = compute_fit_quality(model, solution)
    assert quality.degrees_of_freedom == 2  # Не N минус число положительных оценок.
    assert quality.nominal_dof
    assert any("номинальное" in note for note in quality.warnings)
