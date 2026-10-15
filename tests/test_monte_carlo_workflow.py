"""Эксперимент получает калибровку/веса из снимка основного расчёта."""

import numpy as np
import pytest

from process_measurement import process_prepared_measurement
from xemat.monte_carlo import run_from_results


@pytest.mark.parametrize("option,value", [
    ("weights", np.array([.1, .5, .8])),
    ("weights", np.array([[.3, .05, 0], [.05, .5, .1], [0, .1, .8]])),
    ("observation_covariance", np.array([[7., 1., 0], [1., 11., 2.], [0, 2., 19.]])),
])
def test_npz_weights_survive_file_removal_and_poisson_to_wls_experiment(tmp_path, option, value):
    path = tmp_path / "model.npz"
    sensitivity = np.array([1., 2., 3.])
    background = np.full(3, 5.)
    np.savez(path, M=sensitivity[:, None], x=[8, 17, 25], b=background,
             n_sample_parameters=1, coordinates="a_c_gamma",
             isotope_names=["Xe133"], mdc_component_indices=[0], **{option: value})
    source = process_prepared_measurement(path, method="poisson")
    np.testing.assert_array_equal(source["quadratic_options"][option], value)
    path.unlink()  # Эксперимент не должен перечитывать NPZ и менять настройки.
    result = run_from_results(source, component_index=0, n_trials=5, seed=37, method="wls")
    W = np.linalg.inv(value) if option == "observation_covariance" else (
        np.diag(value) if value.ndim == 1 else value
    )
    rng = np.random.default_rng(37)
    for scenario in result["scenarios"]:
        observed = rng.poisson(background + scenario["activity"] * sensitivity, size=(5, 3))
        expected = (observed - background) @ W @ sensitivity / (sensitivity @ W @ sensitivity)
        np.testing.assert_allclose(scenario["free_activity_samples"], expected, rtol=1e-13, atol=1e-13)
        assert scenario["n_failed"] == 0
    assert result["reference_fit_method"] == "poisson_nonnegative"
