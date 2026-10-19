"""Независимые эталоны для распределений свободных оценок Монте-Карло.

Насыщенная трёхспектральная модель имеет явное решение, поэтому проверяем
не только повторяемость, но и каждый результат по разыгранным отсчётам.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import xemat.distribution_mc as mc
from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


def scalar_source(counts=(21, 30, 40), *, transform=2.5, source_method="wls"):
    # beta = c − f*d − (ks − f*kg)*b_DETBK; a_V = transform*beta.
    f, ks, kg = .5, .25, .2
    matrix = np.array([[1., f, ks], [0., 1., kg], [0., 0., 1.]])
    model = JointModel(matrix, counts, np.zeros(3), 1, [[transform]],
                       "beta_gamma_zeta", n_detector_parameters=1)
    source = {
        "model": model, "fit": fit(model, method=source_method),
        "isotope_names": ["Xe133"], "detection": None,
        "quadratic_options": {"weights": None, "observation_covariance": None},
    }
    contrast = transform * np.array([1., -f, -ks + f * kg])
    return source, contrast


@pytest.mark.parametrize("method", ["wls", "poisson"])
def test_observed_generation_matches_seeded_three_spectrum_closed_form(method, monkeypatch):
    source, contrast = scalar_source()
    trials, methods = [], []
    original_fit = mc.fit

    def capture(model, method, **options):
        methods.append(method)
        if model is not source["model"]:
            trials.append(model)
        return original_fit(model, method=method, **options)

    monkeypatch.setattr(mc, "fit", capture)
    result = mc.run_from_results(source, generation="observed", method=method,
                                 n_trials=32, seed=8603)
    draws = np.random.default_rng(8603).poisson(source["model"].x, size=(32, 3))
    expected = draws @ contrast
    np.testing.assert_allclose(result["activity_samples"][:, 0], expected, atol=3e-7)
    assert np.any(expected < 0), "Отрицательные свободные оценки должны сохраняться"
    assert len(trials) == 32
    for trial, draw in zip(trials, draws):
        np.testing.assert_array_equal(trial.x, draw)
        assert trial.n_detector_parameters == 1
        assert trial.coordinates == "beta_gamma_zeta"
    assert np.ptp(draws[:, 2]) > 0
    assert methods == [method] * 33  # Исходная свободная оценка и 32 повтора.
    assert result["generation_activity"] is None
    assert result["generator_physical_parameters"] is None
    np.testing.assert_array_equal(result["generation_mu"], source["model"].x)
    assert result["n_valid"] == result["n_attempted"] == 32
    assert result["n_failed"] == 0

    # Для насыщенной свободной модели обе ковариации имеют тот же эталон:
    # Var(a_V)=T²*(c + f²*d + (ks−f*kg)²*b_DETBK), включая шум DETBK один раз.
    variances = draws @ contrast**2
    np.testing.assert_allclose(result["trial_stat_sigma_samples"][:, 0],
                               np.sqrt(variances), rtol=4e-7, atol=3e-7)
    np.testing.assert_allclose(result["original_activity"], [source["model"].x @ contrast], atol=3e-7)
    np.testing.assert_allclose(result["original_stat_sigma"],
                               [np.sqrt(source["model"].x @ contrast**2)], rtol=4e-7)
    summary = result["summary"][0]
    assert summary["component_index"] == 0
    assert summary["name"] == "Xe133"
    assert summary["mean"] == pytest.approx(np.mean(expected), abs=3e-7)
    assert summary["std"] == pytest.approx(np.std(expected, ddof=1), abs=3e-7)
    assert summary["median"] == pytest.approx(np.median(expected), abs=3e-7)
    assert summary["q025"] == pytest.approx(np.quantile(expected, .025), abs=3e-7)
    assert summary["q975"] == pytest.approx(np.quantile(expected, .975), abs=3e-7)
    assert summary["mean_stat_sigma"] == pytest.approx(np.mean(np.sqrt(variances)), rel=4e-7)
    assert summary["sigma_count"] == 32


@pytest.mark.parametrize("source_method", ["wls", "nnls", "poisson", "poisson_nonnegative"])
def test_fitted_generator_is_one_positive_poisson_fit_independent_of_source_and_analysis(source_method, monkeypatch):
    # Исходный свободный beta отрицателен: использование свободного/NNLS
    # спектра вместо неотрицательного ММП изменит средние этого эксперимента.
    source, _ = scalar_source(counts=(7, 30, 40), source_method=source_method)
    reference = fit(source["model"], method="poisson_nonnegative")
    original_fit = mc.fit
    captured = []

    def capture(model, method, **options):
        captured.append((model, method, options))
        return original_fit(model, method=method, **options)

    monkeypatch.setattr(mc, "fit", capture)
    results = []
    for method in ("wls", "poisson"):
        captured.clear()
        result = mc.run_from_results(source, generation="fitted", method=method,
                                     n_trials=5, seed=934)
        positive_calls = [call for call in captured if call[1] == "poisson_nonnegative"]
        assert len(positive_calls) == 1
        assert positive_calls[0][0] is source["model"]
        assert not positive_calls[0][2]
        assert len(captured) == 7
        assert result["generator_fit_method"] == "poisson_nonnegative"
        np.testing.assert_allclose(result["generation_mu"], reference.mu, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(result["generator_physical_parameters"],
                                   reference.physical_parameters, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(result["generation_activity"], reference.physical_parameters[:1], atol=1e-10)
        results.append(result)
    np.testing.assert_array_equal(results[0]["generation_mu"], results[1]["generation_mu"])
    # Одинаковые средние и seed дают одни отсчёты; насыщенные свободные
    # МНК и ММП имеют одинаковое аналитическое решение для каждого повтора.
    np.testing.assert_allclose(results[0]["activity_samples"], results[1]["activity_samples"], atol=4e-7)


def test_physical_activities_use_full_chain_transform_not_beta_coordinates():
    transform = np.array([[2., -.4], [0., 3.]])
    f, ks, kg = .3, .35, .2
    matrix = build_joint_matrix(np.eye(2), np.eye(2), np.eye(2), f * np.eye(2),
                                kappa_s=ks, kappa_g=kg)
    model = JointModel(matrix, [25, 45, 30, 50, 60, 80], np.zeros(6), 2,
                       transform, "beta_gamma_zeta", n_detector_parameters=2)
    source = {"model": model, "fit": fit(model, "wls"), "isotope_names": ["Xe133", "Xe133m"]}
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=12, seed=170)
    draws = np.random.default_rng(170).poisson(model.x, size=(12, 6))
    beta = draws[:, :2] - f * draws[:, 2:4] - (ks - f * kg) * draws[:, 4:]
    expected = beta @ transform.T
    np.testing.assert_allclose(result["activity_samples"], expected, atol=2e-12)
    beta_variance = draws[:, :2] + f**2 * draws[:, 2:4] + (ks - f * kg)**2 * draws[:, 4:]
    expected_sigma = np.sqrt(beta_variance @ (transform**2).T)
    np.testing.assert_allclose(result["trial_stat_sigma_samples"], expected_sigma, atol=2e-12)
    np.testing.assert_array_equal(result["component_indices"], [0, 1])
    assert list(result["component_names"]) == ["Xe133", "Xe133m"]


@pytest.mark.parametrize("method", ["wls", "poisson"])
@pytest.mark.parametrize("counts,zero_index", [((0, 30, 40), 0), ((21, 30, 0), 2)])
def test_zero_observed_roi_is_not_replaced_by_pseudocount(method, counts, zero_index, monkeypatch):
    source, contrast = scalar_source(counts=counts)
    original_fit = mc.fit
    trial_counts = []

    def capture(model, method, **options):
        if model is not source["model"]:
            trial_counts.append(model.x.copy())
        return original_fit(model, method=method, **options)

    monkeypatch.setattr(mc, "fit", capture)
    result = mc.run_from_results(source, generation="observed", method=method, n_trials=6, seed=18)
    assert result["generation_mu"][zero_index] == 0
    assert all(counts[zero_index] == 0 for counts in trial_counts)
    assert result["n_valid"] == 6
    draws = np.array(trial_counts)
    np.testing.assert_allclose(result["activity_samples"][:, 0], draws @ contrast, atol=2e-7)
    if method == "wls":
        expected_sigma = np.sqrt(np.maximum(draws, 1) @ contrast**2)
        np.testing.assert_allclose(result["trial_stat_sigma_samples"][:, 0], expected_sigma, atol=1e-12)
    else:
        # Граничный ММП имеет допустимые оценки, но обычная информация
        # для sigma не определена. Не выбрасываем повторы, не рисуем sigma=0.
        assert np.all(np.isnan(result["trial_stat_sigma_samples"]))
        assert result["summary"][0]["mean_stat_sigma"] is None
        assert result["summary"][0]["sigma_count"] == 0


def test_selected_xenon_indices_do_not_require_detection_limits():
    n = 5
    matrix = build_joint_matrix(np.eye(n), np.eye(n), np.eye(n), .2 * np.eye(n),
                                kappa_s=.3, kappa_g=.1)
    model = JointModel(matrix, np.arange(15) + 40, np.zeros(15), n,
                       np.eye(n), "beta_gamma_zeta", n_detector_parameters=n)
    source = {"model": model, "fit": fit(model, "nnls"),
              "isotope_names": ["Rn 222", "Xe135", "Xe133", "Xe131m", "Xe133m"],
              "detection": None, "detection_unavailable_reason": "нулевой фоновый ROI"}
    np.testing.assert_array_equal(mc.eligible_components(source), [1, 2, 3, 4])
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=3)
    assert result["activity_samples"].shape == (3, 4)
    assert result["n_failed"] == 0


def test_standard_seven_roi_model_draws_all_twenty_one_counts_including_detbk(monkeypatch):
    response = np.vstack((np.eye(5), [.7, .3, .4, .5, .2], [.2, .4, .8, .3, .5]))
    matrix = build_joint_matrix(response, response, response, .4 * np.eye(5),
                                kappa_s=.3, kappa_g=.2)
    physical = np.r_[np.arange(5) + 5., np.arange(5) + 80., np.arange(7) * 10 + 1000.]
    model = JointModel(matrix, np.rint(matrix @ physical), np.zeros(21), 5,
                       coordinates="a_V_gamma_zeta", n_detector_parameters=7)
    source = {"model": model, "fit": fit(model, "wls"),
              "isotope_names": ["Rn222", "Xe135", "Xe133", "Xe131m", "Xe133m"]}
    original_fit, trials = mc.fit, []

    def capture(trial, method, **options):
        if trial is not model:
            trials.append(trial)
        return original_fit(trial, method=method, **options)

    monkeypatch.setattr(mc, "fit", capture)
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=8, seed=341)
    draws = np.random.default_rng(341).poisson(model.x, size=(8, 21))
    assert result["activity_samples"].shape == (8, 4)
    assert len(trials) == 8
    for trial, counts in zip(trials, draws):
        np.testing.assert_array_equal(trial.x, counts)
        np.testing.assert_array_equal(trial.b, np.zeros(21))
        assert trial.M.shape == (21, 17)
        assert trial.n_detector_parameters == 7
    assert np.all(np.ptp(draws[:, 14:], axis=0) > 0)
    np.testing.assert_array_equal(model.x, np.rint(matrix @ physical))


def test_reported_sigma_is_statistical_not_total_with_systematics():
    source, contrast = scalar_source(counts=(100, 60, 90))
    source["activities"] = SimpleNamespace(
        std_A_c=np.array([1e12]), K_A_c=np.array([[1e24]]),
        K_A_c_sys=np.array([[1e24]]), K_A_c_stat=np.array([[1e-20]]),
    )
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=10, seed=77)
    draws = np.random.default_rng(77).poisson(source["model"].x, size=(10, 3))
    expected_sigmas = np.sqrt(draws @ contrast**2)
    np.testing.assert_allclose(result["original_stat_sigma"], [np.sqrt(source["model"].x @ contrast**2)])
    np.testing.assert_allclose(result["trial_stat_sigma_samples"][:, 0], expected_sigmas)


@pytest.mark.parametrize("option,value", [
    ("weights", np.array([.1, .5, .8])),
    ("weights", np.array([[.3, .05, 0], [.05, .5, .1], [0, .1, .8]])),
    ("observation_covariance", np.array([[7., 1., 0], [1., 11., 2.], [0, 2., 19.]])),
])
def test_external_quadratic_options_survive_poisson_source_and_are_kept_fixed(option, value):
    sensitivity = np.array([1., 2., 3.])
    background = np.full(3, 5.)
    model = JointModel(sensitivity[:, None], [8, 17, 25], background, 1, coordinates="a_c_gamma")
    source = {"model": model, "fit": fit(model, "poisson"), "isotope_names": ["Xe133"],
              "quadratic_options": {option: value}}
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=8, seed=37)
    W = np.linalg.inv(value) if option == "observation_covariance" else (
        np.diag(value) if value.ndim == 1 else value)
    draws = np.random.default_rng(37).poisson(model.x, size=(8, 3))
    expected = (draws - background) @ W @ sensitivity / (sensitivity @ W @ sensitivity)
    np.testing.assert_allclose(result["activity_samples"][:, 0], expected, atol=1e-12)
    expected_sigma = 1 / np.sqrt(sensitivity @ W @ sensitivity)
    np.testing.assert_allclose(result["trial_stat_sigma_samples"], expected_sigma, atol=1e-12)
    np.testing.assert_array_equal(result["quadratic_options"][option], value)
    old_value = value.copy()
    value[...] = 999
    np.testing.assert_array_equal(result["quadratic_options"][option], old_value)


def test_failed_trial_is_counted_and_does_not_redraw_or_shift_rng(monkeypatch):
    source, contrast = scalar_source()
    original_fit = mc.fit
    trials = []

    def fail_one(model, method, **options):
        solution = original_fit(model, method=method, **options)
        if model is not source["model"]:
            trials.append(model.x.copy())
            if len(trials) == 4:
                # Оптимизатор мог остановиться по шагу без подтверждения ККТ.
                solution.success = False
                solution.message = "xtol termination condition is satisfied"
                solution.warnings.append("искусственная ошибка сходимости: ККТ не подтверждены")
        return solution

    monkeypatch.setattr(mc, "fit", fail_one)
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=8, seed=980)
    draws = np.random.default_rng(980).poisson(source["model"].x, size=(8, 3))
    np.testing.assert_array_equal(trials, draws)
    np.testing.assert_allclose(result["activity_samples"][:, 0], np.delete(draws @ contrast, 3), atol=1e-12)
    assert result["n_requested"] == result["n_attempted"] == 8
    assert result["n_valid"] == 7
    assert result["n_failed"] == 1
    assert sum(result["failure_reasons"].values()) == 1
    assert "искусственная ошибка" in " ".join(result["failure_reasons"])
    assert "ККТ не подтверждены" in " ".join(result["failure_reasons"])
    assert not result["cancelled"]


def test_cancellation_retains_successful_prefix_and_partial_counts():
    source, contrast = scalar_source()
    progress = []
    result = mc.run_from_results(
        source, generation="observed", method="wls", n_trials=10, seed=117,
        progress=lambda done, total: progress.append((done, total)),
        should_cancel=lambda: bool(progress) and progress[-1][0] >= 3,
    )
    draws = np.random.default_rng(117).poisson(source["model"].x, size=(3, 3))
    np.testing.assert_allclose(result["activity_samples"][:, 0], draws @ contrast, atol=1e-12)
    assert result["cancelled"]
    assert result["n_requested"] == 10
    assert result["n_attempted"] == result["n_valid"] == 3
    assert result["n_failed"] == 0
    assert progress[0] == (0, 10)
    assert progress[-1] == (3, 10)


def test_single_success_has_no_invented_sample_standard_deviation():
    source, _ = scalar_source()
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=1, seed=117)
    assert result["summary"][0]["std"] is None
    assert result["summary"][0]["mean"] == pytest.approx(result["activity_samples"][0, 0])


def test_immediate_cancellation_preserves_empty_two_dimensional_samples():
    source, _ = scalar_source()
    result = mc.run_from_results(source, generation="observed", method="wls", n_trials=10,
                                 should_cancel=lambda: True)
    assert result["cancelled"]
    assert result["n_attempted"] == result["n_valid"] == result["n_failed"] == 0
    assert result["activity_samples"].shape == (0, 1)
    assert result["trial_stat_sigma_samples"].shape == (0, 1)
    assert result["summary"][0]["mean"] is None
    assert result["summary"][0]["q025"] is None


@pytest.mark.parametrize("kwargs", [
    {"method": "nnls"}, {"method": "poisson_nonnegative"},
    {"generation": "unknown"}, {"n_trials": 0}, {"n_trials": 1.5},
    {"n_trials": True}, {"seed": -1}, {"seed": False},
])
def test_invalid_experiment_options_fail_before_trials(kwargs):
    source, _ = scalar_source()
    options = {"generation": "observed", "method": "wls", "n_trials": 2, "seed": 1}
    options.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        mc.run_from_results(source, **options)
