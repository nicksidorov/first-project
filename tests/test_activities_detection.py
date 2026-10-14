"""Проверки пересчёта активностей и MDC по формулам статьи, не по legacy."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from xemat.activities import (
    compose_activity_transform,
    compute_activities,
    covariance_to_correlation,
)
from xemat.detection import compute_detection_limits, model_covariance


def test_activity_chain_and_correlated_volume_error():
    beta = np.array([12.0, 5.0])
    K_beta = np.array([[4.0, 1.0], [1.0, 2.0]])
    H = np.diag([2.0, 3.0])
    Gamma = np.array([[1.0, -0.4], [0.0, 1.0]])
    G_p = np.diag([1.1, 1.8])
    G_s = np.diag([1.2, 1.5])
    V0 = 10.0
    delta_I = np.array([0.1, 0.2])
    delta_V0 = 0.15

    result = compute_activities(
        beta, K_beta, H=H, Gamma=Gamma, G_p=G_p, G_s=G_s,
        V0=V0, delta_I=delta_I, delta_V0=delta_V0,
    )
    P = G_s @ G_p @ Gamma / V0
    a = H @ beta
    a_c = P @ a
    K_A_stat = H @ K_beta @ H.T
    K_A_sys = np.diag(a**2 * delta_I**2)
    expected_stat = P @ K_A_stat @ P.T
    expected_systematic = P @ K_A_sys @ P.T + delta_V0**2 * np.outer(a_c, a_c)
    assert_allclose(result.a, a)
    assert_allclose(result.a_tilde, Gamma @ a)
    assert_allclose(result.a_c, a_c)
    assert_allclose(result.T_c, P @ H)
    assert_allclose(result.K_A, K_A_stat + K_A_sys)
    assert_allclose(result.K_A_c_stat, expected_stat)
    assert_allclose(result.K_A_c, expected_stat + expected_systematic)
    assert_allclose(result.std_A_c**2, np.diag(result.K_A_c))
    assert_allclose(result.correlation_A_c, result.K_A_c / np.outer(result.std_A_c, result.std_A_c))


def test_prepared_T_c_and_H_match_composed_transform():
    beta = [4.0, 3.0]
    H = np.diag([0.2, 0.3])
    Gamma = np.array([[1.0, -0.5], [0.0, 1.0]])
    G_p, G_s, V0 = np.eye(2), np.diag([1.0, 1.5]), 5.0
    T_c = compose_activity_transform(H, Gamma, G_p, G_s, V0)
    direct = compute_activities(beta, np.eye(2), H=H, T_c=T_c, delta_I=[0.2, 0.1])
    composed = compute_activities(
        beta, np.eye(2), H=H, Gamma=Gamma, G_p=G_p, G_s=G_s,
        V0=V0, delta_I=[0.2, 0.1],
    )
    assert_allclose(direct.a_c, composed.a_c)
    assert_allclose(direct.K_A_c, composed.K_A_c)


def test_chain_transform_preserves_negative_physical_estimate():
    # beta >= 0 не гарантирует a_c >= 0 при отрицательной цепочечной поправке.
    result = compute_activities([1.0, 4.0], T_c=[[1.0, -0.5], [0.0, 1.0]])
    assert_allclose(result.a_c, [-1.0, 4.0])
    assert result.a is None
    assert result.K_A_c is None
    assert result.std_A_c is None


def test_constrained_estimate_gets_no_fictitious_covariance():
    result = compute_activities([0.0, 3.0], H=np.eye(2), T_c=np.eye(2), delta_I=[0.1, 0.1])
    assert_allclose(result.a, [0, 3])
    assert result.K_A is None
    assert result.K_A_c is None
    assert result.correlation_A is None
    assert_allclose(result.K_A_sys, np.diag([0, 0.09]))


@pytest.mark.parametrize("kwargs", [
    {"T_c": np.eye(2), "delta_I": [0.1, 0.1]},
    {"H": np.eye(2), "G_s": np.eye(2)},
    {"H": np.eye(2), "delta_I": [-0.1, 0]},
    {"T_c": np.eye(2), "delta_V0": -0.1},
])
def test_invalid_activity_inputs_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        compute_activities([1.0, 2.0], **kwargs)


def test_inconsistent_prepared_and_component_transform_rejected():
    with pytest.raises(ValueError, match="не соответствует"):
        compute_activities(
            [1, 2], H=np.eye(2), T_c=2 * np.eye(2), Gamma=np.eye(2),
            G_p=np.eye(2), G_s=np.eye(2), V0=1,
        )


def test_covariance_zero_variance_is_undefined_correlation():
    correlation = covariance_to_correlation(np.diag([0.0, 2.0]))
    assert np.isnan(correlation[0, 0])
    assert np.isnan(correlation[0, 1])
    assert correlation[1, 1] == pytest.approx(1)
    with pytest.raises(ValueError, match="полуопределённая"):
        compute_activities([1, 2], [[1, 2], [2, 1]], H=np.eye(2))


def test_mdc_one_channel_matches_independent_analytic_solution():
    sensitivity, background = 2.0, 4.0
    k_a, k_b = norm.ppf(0.95), norm.ppf(0.90)
    LC = k_a * np.sqrt(background) / sensitivity
    # L_D=LC+k_b sqrt(b+s L_D)/s; положительный корень для D=L_D-LC.
    linear = k_b**2 / sensitivity
    constant = k_b**2 * (background + sensitivity * LC) / sensitivity**2
    LD = LC + (linear + np.sqrt(linear**2 + 4 * constant)) / 2
    result = compute_detection_limits(
        [[sensitivity]], [background], [LC + 0.1], [20.0],
        component_indices=[0], alpha=0.05, beta_ii=0.10, rtol=1e-11,
    )
    assert_allclose(result.L_C, [LC])
    assert_allclose(result.MDC, [LD], rtol=1e-10)
    assert_allclose(result.MDC, result.L_C + k_b * result.sigma_at_MDC, rtol=1e-10)
    assert result.converged[0]
    # Основная оценка между LC и MDC всё равно означает обнаружение.
    assert LC + 0.1 < LD
    assert result.detected[0]


def test_detection_uses_free_estimate_and_includes_nuisance_variance():
    # theta=[a_c,gamma], mu=[a_c+gamma,gamma]+b.
    S_c = np.array([[1.0, 1.0], [0.0, 1.0]])
    result = compute_detection_limits(S_c, [4, 9], [-1, 2], [30, 2], component_indices=[0])
    # Удалили только a_c, сохранили gamma и полный столбец сигнала в информации.
    assert_allclose(result.null_spectra, [[6, 11]])
    assert_allclose(result.sigma0**2, [17])
    assert_allclose(result.null_covariances[0], [[17, -11], [-11, 11]])
    assert not result.detected[0]
    assert_allclose(result.free_activity, [-1])
    assert_allclose(result.positive_activity, [30])


def test_zero_estimate_nuisance_column_is_retained():
    S = np.array([[1.0, 1.0], [0.0, 1.0]])
    result = compute_detection_limits(S, [4, 9], [10, 0], [10, 0], component_indices=[0])
    assert_allclose(result.sigma0**2, [13])
    # Удаление нулевого nuisance-столбца ошибочно дало бы дисперсию 4.
    assert result.null_covariances.shape == (1, 2, 2)


def test_same_isotope_memory_is_preserved_in_joint_model():
    S = np.array([
        [1, 0, 0.5, 0], [0, 1, 0, 0.8],
        [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=float)
    result = compute_detection_limits(S, np.ones(4), [1, 2, 3, 4], [5, 6, 7, 8], component_indices=[0])
    assert_allclose(result.null_spectra, [[4.5, 13.4, 8, 9]])


def test_mdc_uses_entire_physical_parent_column_including_daughter():
    # T_c [[1,-.5],[0,1]] => T_c^{-1} [[1,.5],[0,1]].
    # Исключая родительский a_c[1], удаляем одновременно оба спектральных вклада.
    S = np.array([[1.0, 0.5], [0.0, 1.0]])
    result = compute_detection_limits(S, [2, 3], [2, 4], [3, 4], component_indices=[1])
    assert_allclose(result.null_spectra, [[5, 3]])
    assert_allclose(result.sigma0**2, [3])


def test_covariance_column_scaling_invariance():
    S = np.array([[1, 1], [0, 1], [1, 2]], dtype=float)
    expected = [5, 6, 7]
    covariance = model_covariance(S, expected)
    scales = np.array([1e-12, 1e12])
    scaled_covariance = model_covariance(S * scales, expected)
    assert_allclose(scaled_covariance * np.outer(scales, scales), covariance, rtol=1e-12)


def test_strong_overlap_not_squared_by_normal_equations():
    epsilon = 1e-7
    S = np.array([[1, 1], [1, 1 + epsilon], [1, 1 - epsilon]], dtype=float)
    K = model_covariance(S, [3, 3, 3])
    # Независимый вывод: сумма квадратов контраста 2 epsilon² / 3.
    assert_allclose(K[1, 1], 3 / (2 * epsilon**2), rtol=1e-8)
    assert K[0, 1] < 0


def test_fixed_variance_mdc_is_explicit_approximation():
    fixed = compute_detection_limits([[1]], [1], [0], [5], component_indices=[0], mode="fixed")
    iterative = compute_detection_limits([[1]], [1], [0], [5], component_indices=[0])
    assert_allclose(fixed.MDC, 2 * fixed.L_C)
    assert fixed.iterations[0] == 0
    assert fixed.MDC[0] < iterative.MDC[0]
    assert any("приближении постоянной" in message for message in fixed.warnings)


def test_nonconvergence_is_not_replaced_by_two_LC():
    result = compute_detection_limits([[1]], [1], [0], [0], component_indices=[0], max_iter=1)
    assert not result.converged[0]
    assert np.isnan(result.MDC[0])
    assert np.isfinite(result.last_iterate[0])
    assert any("не сошёлся" in message for message in result.warnings)


def test_default_xenon_components_exclude_radon():
    result = compute_detection_limits(np.eye(10), np.ones(10), np.ones(10), np.ones(10))
    assert_allclose(result.component_indices, [1, 2, 3, 4])
    assert result.null_covariances.shape == (4, 10, 10)


@pytest.mark.parametrize("S,background,positive,error", [
    ([[1]], [0], [0], "нулевой модельный спектр"),
    ([[1]], [1], [-1], "Вспомогательная оценка"),
    ([[1, 1], [1, 1]], [1, 1], [0, 1], "неидентифицируема"),
])
def test_invalid_mdc_inputs_are_not_abs_or_epsilon_corrected(S, background, positive, error):
    with pytest.raises(ValueError, match=error):
        compute_detection_limits(S, background, np.zeros(len(positive)), positive, component_indices=[0])


def test_model_covariance_rejects_zero_mu_and_zero_columns():
    with pytest.raises(ValueError, match="mu > 0"):
        model_covariance([[1]], [0])
    with pytest.raises(ValueError, match="нулевой столбец"):
        model_covariance([[0]], [1])
