"""Независимые статистические проверки экспериментальной процедуры LC/LD."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm, poisson

import xemat.monte_carlo as mc
from xemat.detection import compute_detection_limits
from xemat.model import JointModel


def scalar_model(background=1.0, sensitivity=1.0):
    return JointModel(
        [[sensitivity]], [round(background)], [background], 1,
        coordinates="a_c_gamma",
    )


def scalar_oracle(result, background, sensitivity, seed, n_trials):
    """Одноканальная задача решается явно, без вызова решателя/расчёта LC."""
    z = norm.ppf(1 - result["alpha"])
    expected_lc = z * np.sqrt(background) / sensitivity
    # При alpha=beta уравнение Керри имеет это явное положительное решение.
    expected_ld = (2 * z * np.sqrt(background) + z**2) / sensitivity
    assert result["reference_L_C"] == pytest.approx(expected_lc, rel=1e-13)
    assert result["reference_MDC"] == pytest.approx(expected_ld, rel=3e-8)
    rng = np.random.default_rng(seed)
    generated = []
    for scenario, activity in zip(result["scenarios"], (0.0, expected_ld)):
        counts = rng.poisson(background + sensitivity * activity, size=n_trials)
        generated.append(counts)
        expected_estimates = (counts - background) / sensitivity
        np.testing.assert_allclose(scenario["free_activity_samples"], expected_estimates, atol=2e-8)
        np.testing.assert_allclose(scenario["L_C_samples"], expected_lc, rtol=1e-13)
        # Strict >: целочисленные отсчёты должны превысить порог, а не достичь его.
        expected_detected = np.count_nonzero(counts > background + sensitivity * expected_lc)
        assert scenario["n_detected"] == expected_detected
        assert scenario["n_valid"] == scenario["n_attempted"] == n_trials
        assert scenario["n_failed"] == 0
        assert scenario["detection_rate"] == expected_detected / n_trials
    return generated


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_low_count_poisson_oracle_includes_zeros_and_non_nominal_alpha(method):
    n_trials, seed = 500, 8127
    result = mc.run_monte_carlo(
        scalar_model(), [3.0], component_index=0,
        method=method, n_trials=n_trials, seed=seed,
    )
    generated = scalar_oracle(result, 1.0, 1.0, seed, n_trials)
    assert np.any(generated[0] == 0)
    # Нормальное приближение при b=1 действительно даёт ~8%, а не 5%.
    exact_alpha = poisson.sf(2, 1)
    exact_power = poisson.sf(2, 1 + 2 * norm.ppf(.95) + norm.ppf(.95)**2)
    assert exact_alpha == pytest.approx(0.08030139707139418)
    assert exact_power == pytest.approx(0.9702575515140656)
    for scenario, probability in zip(result["scenarios"], (exact_alpha, exact_power)):
        standard_error = np.sqrt(probability * (1 - probability) / n_trials)
        assert abs(scenario["detection_rate"] - probability) < 5 * standard_error


def test_high_count_poisson_oracle_and_activity_units():
    n_trials, seed = 400, 713
    result = mc.run_monte_carlo(
        scalar_model(background=100, sensitivity=7), [4.0], component_index=0,
        n_trials=n_trials, seed=seed,
    )
    scalar_oracle(result, 100, 7, seed, n_trials)
    cutoff = np.floor(100 + norm.ppf(.95) * 10)
    assert poisson.sf(cutoff, 100) == pytest.approx(0.05221548926299169)
    assert abs(result["scenarios"][0]["detection_rate"] - .05) < .04
    assert abs(result["scenarios"][1]["detection_rate"] - .95) < .04


def test_detection_excludes_exact_equality_at_lc():
    alpha = float(norm.sf(1))
    result = mc.run_monte_carlo(
        scalar_model(), [2.0], component_index=0,
        n_trials=100, seed=73, alpha=alpha, beta_ii=alpha,
    )
    generated = scalar_oracle(result, 1, 1, 73, 100)
    assert result["reference_L_C"] == 1
    assert np.any(generated[0] == 2)  # a_hat=1=LC, эти реализации не обнаружены.
    assert result["scenarios"][0]["n_detected"] < np.count_nonzero(generated[0] >= 2)


def test_nondiagonal_coordinate_transform_preserves_memory_and_experiment():
    sensitivity = np.array([
        [2, .2, 1, .1], [.5, 1.5, .2, .8], [.2, .4, .7, .4],
        [0, 0, 1, .2], [0, 0, .3, 1], [0, 0, .8, .5],
    ])
    transform = np.array([[2, .7], [0, 3.]])
    full_transform = np.eye(4)
    full_transform[:2, :2] = transform
    positive = np.array([1.3, 2.4, 8., 4.])
    background = np.array([2., 3., 4., 1., 2., 3.])
    counts = np.rint(background + sensitivity @ positive)
    physical_model = JointModel(sensitivity, counts, background, 2, coordinates="a_c_gamma")
    beta_model = JointModel(sensitivity @ full_transform, counts, background, 2, transform)
    kwargs = dict(component_index=0, n_trials=20, seed=77)
    physical_result = mc.run_monte_carlo(physical_model, positive, **kwargs)
    beta_result = mc.run_monte_carlo(beta_model, positive, **kwargs)
    expected_null = np.array([0., 2.4, 8., 4.])
    for result in (physical_result, beta_result):
        np.testing.assert_allclose(result["reference_null_physical"], expected_null)
        np.testing.assert_allclose(result["reference_null_mu"], background + sensitivity @ expected_null)
        np.testing.assert_allclose(
            result["reference_signal_mu"],
            background + sensitivity @ expected_null + result["reference_MDC"] * sensitivity[:, 0],
        )
        # Мешающие оценки заново подгоняются: LC не заморожен на исходном значении.
        assert np.ptp(result["scenarios"][0]["L_C_samples"]) > .01
    for first, second in zip(physical_result["scenarios"], beta_result["scenarios"]):
        np.testing.assert_allclose(first["free_activity_samples"], second["free_activity_samples"], atol=2e-13)
        np.testing.assert_allclose(first["L_C_samples"], second["L_C_samples"], atol=2e-13)
        assert first["n_detected"] == second["n_detected"]
    np.testing.assert_array_equal(positive, [1.3, 2.4, 8, 4])


@pytest.mark.parametrize("weight_mode", ["observed", "external_weights", "external_covariance"])
def test_wls_weights_have_expected_per_trial_behavior(weight_mode):
    response = np.array([1., 2.])
    background = np.array([5., 8.])
    model = JointModel(response[:, None], [7, 12], background, 1, coordinates="a_c_gamma")
    options = {}
    if weight_mode == "external_weights":
        options["weights"] = np.array([.25, 4.])
        fixed_precision = np.diag(options["weights"])
    elif weight_mode == "external_covariance":
        options["observation_covariance"] = np.array([[2., .6], [.6, 5.]])
        fixed_precision = np.array([[5., -.6], [-.6, 2.]]) / (10 - .6**2)
    result = mc.run_monte_carlo(model, [2.], component_index=0, n_trials=25, seed=42, **options)
    rng = np.random.default_rng(42)
    for scenario, mu in zip(result["scenarios"], (result["reference_null_mu"], result["reference_signal_mu"])):
        estimates = []
        for _ in range(25):
            counts = rng.poisson(mu)
            precision = np.diag(1 / np.maximum(counts, 1)) if weight_mode == "observed" else fixed_precision
            estimates.append((response @ precision @ (counts - background)) / (response @ precision @ response))
        np.testing.assert_allclose(scenario["free_activity_samples"], estimates, atol=1e-12)


def test_failed_fits_are_counted_without_resampling(monkeypatch):
    original_fit = mc.fit
    free_calls = 0

    def some_fits_fail(model, method, **options):
        nonlocal free_calls
        result = original_fit(model, method=method, **options)
        if method == "wls":
            free_calls += 1
            # Первый вызов проверяет исходную конфигурацию; затем каждый 5-й опыт.
            if free_calls > 1 and (free_calls - 1) % 5 == 0:
                return replace(result, success=False, message="injected trial failure")
        return result

    monkeypatch.setattr(mc, "fit", some_fits_fail)
    progress = []
    result = mc.run_monte_carlo(
        scalar_model(), [3.], component_index=0, n_trials=12, seed=13,
        progress=lambda done, total: progress.append((done, total)),
    )
    rng = np.random.default_rng(13)
    for start, scenario, mu in zip((1, 13), result["scenarios"], (1., result["reference_signal_mu"][0])):
        counts = rng.poisson(mu, size=12)
        valid = np.arange(start, start + 12) % 5 != 0
        expected_estimates = counts[valid] - 1
        detected = int(np.count_nonzero(expected_estimates > norm.ppf(.95)))
        np.testing.assert_allclose(scenario["free_activity_samples"], expected_estimates, atol=1e-12)
        assert scenario["n_attempted"] == 12
        assert scenario["n_valid"] == 10
        assert scenario["n_failed"] == 2
        assert scenario["n_detected"] == detected
        assert scenario["detection_rate"] == detected / 10
        assert scenario["failure_probability_bounds"] == (detected / 12, (detected + 2) / 12)
        assert sum(scenario["failure_reasons"].values()) == 2
    assert progress == [(i, 24) for i in range(25)]


def test_cancellation_keeps_partial_trials_and_exact_interval():
    progress = []
    result = mc.run_monte_carlo(
        scalar_model(), [2.], component_index=0, n_trials=12, seed=1,
        progress=lambda done, total: progress.append((done, total)),
        should_cancel=lambda: bool(progress) and progress[-1][0] >= 7,
    )
    assert result["cancelled"]
    assert len(result["scenarios"]) == 1
    scenario = result["scenarios"][0]
    assert scenario["n_requested"] == 12
    assert scenario["n_attempted"] == scenario["n_valid"] == 7
    counts = np.random.default_rng(1).poisson(1, size=7)
    np.testing.assert_allclose(scenario["free_activity_samples"], counts - 1, atol=1e-12)
    assert scenario["n_detected"] == 0
    assert scenario["confidence_interval"][0] == 0
    assert scenario["confidence_interval"][1] == pytest.approx(1 - .025**(1 / 7))
    assert progress[-1] == (7, 24)


def test_cancel_before_any_trial_has_no_fabricated_zero_rate():
    result = mc.run_monte_carlo(
        scalar_model(), [1.], component_index=0, n_trials=3,
        should_cancel=lambda: True,
    )
    assert result["cancelled"]
    scenario = result["scenarios"][0]
    assert scenario["n_attempted"] == scenario["n_valid"] == scenario["n_failed"] == 0
    assert scenario["detection_rate"] is None
    assert scenario["confidence_interval"] is None
    assert scenario["failure_probability_bounds"] is None


def test_model_with_zero_null_mean_is_rejected_without_epsilon():
    with pytest.raises(ValueError, match="mu_i"):
        mc.run_monte_carlo(scalar_model(background=0), [1.], component_index=0, n_trials=3)


def test_all_trials_failing_report_unknown_rate_and_full_failure_bounds(monkeypatch):
    original_fit = mc.fit
    free_calls = 0

    def only_preflight_succeeds(model, method, **options):
        nonlocal free_calls
        if method == "wls":
            free_calls += 1
            if free_calls > 1:
                raise RuntimeError("all simulated fits fail")
        return original_fit(model, method=method, **options)

    monkeypatch.setattr(mc, "fit", only_preflight_succeeds)
    result = mc.run_monte_carlo(scalar_model(), [1.], component_index=0, n_trials=3)
    assert not result["cancelled"]
    for scenario in result["scenarios"]:
        assert scenario["n_attempted"] == scenario["n_failed"] == 3
        assert scenario["n_valid"] == scenario["n_detected"] == 0
        assert scenario["detection_rate"] is None
        assert scenario["confidence_interval"] is None
        assert scenario["failure_probability_bounds"] == (0, 1)


def test_changing_analysis_family_keeps_original_fit_truth():
    model = JointModel([[1.], [2.]], [2, 13], [1., 1.], 1, coordinates="a_c_gamma")
    free = mc.fit(model, method="wls")
    positive_wls = mc.fit(model, method="nnls")
    positive_poisson = mc.fit(model, method="poisson_nonnegative")
    assert not np.allclose(positive_wls.physical_parameters, positive_poisson.physical_parameters)
    detection = compute_detection_limits(
        model.physical_matrix, model.b, free.physical_parameters,
        positive_wls.physical_parameters, component_indices=[0],
    )
    source = {
        "model": model, "fit": free, "detection": detection,
        "isotope_names": ["Xe133"],
        "quadratic_options": {"weights": None, "observation_covariance": None},
    }
    first = mc.run_from_results(source, component_index=0, n_trials=5, seed=51, method="wls")
    second = mc.run_from_results(source, component_index=0, n_trials=5, seed=51, method="poisson")
    for result in (first, second):
        assert result["reference_fit_method"] == "nnls"
        np.testing.assert_allclose(result["reference_positive_physical"], positive_wls.physical_parameters)
        assert result["component_name"] == "Xe133"
    np.testing.assert_array_equal(first["reference_signal_mu"], second["reference_signal_mu"])
    assert first["reference_MDC"] == second["reference_MDC"]
