"""V86: аналитический учёт статистики DETBK и розыгрыш всех трёх спектров."""

import numpy as np
import pytest
from scipy.stats import norm

import xemat.monte_carlo as mc
from xemat.detection import compute_detection_limits, model_covariance
from xemat.diagnostics import compute_fit_quality
from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


def scalar_joint_model():
    # c = a_V + f gamma + ks zeta; d = gamma + kg zeta; b_DETBK = zeta.
    f, ks, kg = .5, .3, .2
    response = np.array([[1., f, ks], [0., 1., kg], [0., 0., 1.]])
    positive = np.array([4., 30., 100.])
    model = JointModel(
        response, np.rint(response @ positive), np.zeros(3), 1,
        coordinates="a_V_gamma_zeta", n_detector_parameters=1,
    )
    return model, positive, f, ks, kg


def seven_roi_model():
    response = np.vstack((np.eye(5), [.7, .3, .4, .5, .2], [.2, .4, .8, .3, .5]))
    matrix = build_joint_matrix(
        response, response, response, .4 * np.eye(5), kappa_s=.3, kappa_g=.2,
    )
    positive = np.r_[np.arange(5) + 5., np.arange(5) + 80., np.arange(7) * 10 + 1000.]
    model = JointModel(
        matrix, np.rint(matrix @ positive), np.zeros(21), 5,
        coordinates="a_V_gamma_zeta", n_detector_parameters=7,
    )
    return model, positive


def test_scalar_lc_and_mdc_include_detector_noise_once():
    model, positive, f, ks, kg = scalar_joint_model()
    limits = compute_detection_limits(
        model.physical_matrix, model.b, positive, positive, component_indices=[0],
    )
    mu0 = model.M @ np.array([0., positive[1], positive[2]])
    # Явная свободная оценка: a_V = c − f d − (ks − f kg) b_DETBK.
    variance0 = mu0[0] + f**2 * mu0[1] + (ks - f * kg)**2 * mu0[2]
    z = norm.ppf(.95)
    assert limits.sigma0[0] ** 2 == pytest.approx(variance0, rel=1e-13)
    assert limits.L_C[0] == pytest.approx(z * np.sqrt(variance0), rel=1e-13)
    # При добавлении L только Var(c) возрастает на L; alpha=beta.
    assert limits.MDC[0] == pytest.approx(2 * z * np.sqrt(variance0) + z**2, rel=3e-8)
    inverse_response = np.array([[1, -f, f * kg - ks], [0, 1, -kg], [0, 0, 1]])
    expected_covariance = inverse_response @ np.diag(mu0) @ inverse_response.T
    np.testing.assert_allclose(limits.null_covariances[0], expected_covariance, atol=1e-12)

    # Старый известный фон не имел последнего слагаемого. Это ровно
    # дополнительная дисперсия, не сумма ks² Var(b) и f² kg² Var(b):
    # одна и та же оценка DETBK создаёт корреляцию двух фоновых поправок.
    known_covariance = model_covariance(model.M[:2, :2], mu0[:2])
    assert limits.sigma0[0] ** 2 - known_covariance[0, 0] == pytest.approx(
        (ks - f * kg)**2 * mu0[2], rel=1e-12,
    )


def test_null_spectrum_preserves_memory_detector_and_all_fisher_columns():
    model, positive = seven_roi_model()
    limits = compute_detection_limits(model.physical_matrix, model.b, positive, positive)
    np.testing.assert_array_equal(limits.component_indices, [1, 2, 3, 4])
    assert limits.null_covariances.shape == (4, 17, 17)
    fitted = model.M @ positive
    for row, component in enumerate(limits.component_indices):
        np.testing.assert_allclose(limits.null_spectra[row, 7:], fitted[7:], atol=1e-12)
        np.testing.assert_allclose(
            limits.null_spectra[row, :7],
            fitted[:7] - positive[component] * model.M[:7, component], atol=1e-12,
        )
    quality = compute_fit_quality(model, fit(model, method="wls"))
    assert quality.n_observations == 21
    assert quality.rank == 17
    assert quality.degrees_of_freedom == 4


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_scalar_mc_refits_detector_parameter_from_new_counts(method, monkeypatch):
    model, positive, f, ks, kg = scalar_joint_model()
    original_fit = mc.fit
    free_models, free_solutions = [], []

    def capture_fit(trial, method, **options):
        solution = original_fit(trial, method=method, **options)
        if trial is not model and method in ("wls", "poisson"):
            free_models.append(trial)
            free_solutions.append(solution)
        return solution

    monkeypatch.setattr(mc, "fit", capture_fit)
    result = mc.run_monte_carlo(
        model, positive, component_index=0, method=method, n_trials=8, seed=51,
    )
    assert len(free_solutions) == 16
    for trial, solution in zip(free_models, free_solutions):
        assert trial.has_detector_background
        assert trial.n_detector_parameters == 1
        assert trial.coordinates == "a_V_gamma_zeta"
        # Насыщенная свободная модель воспроизводит каждый новый DETBK;
        # старую оценку zeta не фиксируем при повторном решении.
        assert solution.physical_parameters[2] == pytest.approx(trial.x[2], abs=2e-7)
        expected_activity = trial.x[0] - f * trial.x[1] - (ks - f * kg) * trial.x[2]
        assert solution.physical_parameters[0] == pytest.approx(expected_activity, abs=2e-7)
    assert np.ptp([solution.physical_parameters[2] for solution in free_solutions]) > 0
    assert np.ptp(result["scenarios"][0]["L_C_samples"]) > 0
    assert all(row["n_failed"] == 0 for row in result["scenarios"])


def test_mc_draws_all_twenty_one_raw_counts_with_fixed_true_means(monkeypatch):
    model, positive = seven_roi_model()
    original_fit = mc.fit
    trials = []

    def capture_fit(trial, method, **options):
        if trial is not model and method == "wls":
            trials.append(trial)
        return original_fit(trial, method=method, **options)

    monkeypatch.setattr(mc, "fit", capture_fit)
    result = mc.run_monte_carlo(model, positive, component_index=2, n_trials=6, seed=341)
    rng = np.random.default_rng(341)
    expected = []
    for mean in (result["reference_null_mu"], result["reference_signal_mu"]):
        for _ in range(6):
            expected.append(rng.poisson(mean))
    assert len(trials) == len(expected) == 12
    for trial, counts in zip(trials, expected):
        np.testing.assert_array_equal(trial.x, counts)
        assert trial.x.shape == (21,)
        assert trial.n_detector_parameters == 7
        np.testing.assert_array_equal(trial.b, np.zeros(21))
    assert np.all(np.ptp(np.array([trial.x[14:] for trial in trials]), axis=0) > 0)
    np.testing.assert_array_equal(result["reference_null_mu"][14:], positive[10:])
    np.testing.assert_array_equal(result["reference_signal_mu"][14:], positive[10:])
    assert result["experiment_version"] == 2
    report = mc.format_monte_carlo_report(result)
    assert "[a_V; gamma; zeta]" in report
    assert "Разыгрываются SAMPLE, GASBK и DETBK" in report
    assert "детекторный фон zeta" in report
    assert "b (DETBK)" not in report


def test_zero_detector_expectation_rejects_asymptotic_lc_without_pseudocount():
    model, positive, *_ = scalar_joint_model()
    positive[2] = 0
    with pytest.raises(ValueError, match=r"mu_i\^\(0\) > 0"):
        compute_detection_limits(
            model.M, model.b, positive, positive, component_indices=[0],
        )
    with pytest.raises(ValueError, match=r"mu_i\^\(0\) > 0"):
        mc.run_monte_carlo(model, positive, component_index=0, n_trials=1)
