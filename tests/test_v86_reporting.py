"""V86 exports identify the fitted DETBK nuisance parameters and retain the UI."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from process_measurement import process_prepared_measurement
from project import MainWindow
from results_saver import format_results, parameter_names_for_results
from xemat.activities import compute_activities
from xemat.model import JointModel, build_joint_matrix
from xemat.solvers import fit


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def joint_result(*, method="wls", activity_coordinates=False):
    matrix = build_joint_matrix([[1], [2]], [[1], [3]], [[1], [3]], [[.5]],
                                kappa_s=.2, kappa_g=.1)
    model = JointModel(matrix, [48, 91, 22, 46, 102, 199], np.zeros(6), 1,
                       [[.1]], "beta_gamma_zeta", 2)
    if activity_coordinates:
        model = JointModel(model.physical_matrix, model.x, model.b, 1, [[.1]],
                           "a_V_gamma_zeta", 2)
    solution = fit(model, method)
    beta = model.from_physical(solution.physical_parameters)[:1]
    if activity_coordinates:
        beta = solution.theta[:1] / .1
    return {
        "model": model, "fit": solution, "isotope_names": ["Xe133"],
        "weights_source": "1/max(x,1)",
        "activities": compute_activities(beta, H=[[.5]], T_c=[[.1]]),
        "metadata": {"sample_roi_numbers": [3, 8], "gasbk_roi_numbers": [3, 8],
                     "detector_roi_numbers": [3, 8]},
        "detection": SimpleNamespace(
            mode="iterative", alpha=.05, beta_ii=.05, component_indices=np.array([0]),
            free_activity=np.array([2.]), L_C=np.array([.1]), MDC=np.array([.2]),
            converged=np.array([True]), detected=np.array([True]), iterations=np.array([3])),
        "warnings": [], "calibration_matrices": {"kappa_s": .2, "kappa_g": .1},
    }


@pytest.mark.parametrize("activity_coordinates, first", [(False, "beta[0]"), (True, "a_V[0]")])
def test_parameter_and_coordinate_labels_distinguish_three_blocks(activity_coordinates, first):
    result = joint_result(activity_coordinates=activity_coordinates)
    assert parameter_names_for_results(result) == [first, "gamma[0]", "zeta[0]", "zeta[1]"]
    text = format_results(result)
    assert "SAMPLE + предшествующий GASBK + DETBK" in text
    assert "ФИЗИЧЕСКИЕ КООРДИНАТЫ z=[a_V;gamma;zeta]" in text
    assert "ОБЪЁМНАЯ АКТИВНОСТЬ a_V (Бк/м³)" in text
    assert "T_V — переход beta → a_V" in text
    memory = text.split("ПАРАМЕТРЫ ПАМЯТИ γ", 1)[1].split("ПАРАМЕТРЫ ФОНА ДЕТЕКТОРА ζ", 1)[0]
    assert "gamma[0]" in memory
    assert "zeta" not in memory
    background = text.split("ПАРАМЕТРЫ ФОНА ДЕТЕКТОРА ζ", 1)[1].split("4. ФИНАЛЬНЫЕ", 1)[0]
    assert "zeta[0]" in background and "zeta[1]" in background


def test_residual_table_shows_fitted_background_and_raw_detbk_separately():
    result = joint_result()
    model, solution = result["model"], result["fit"]
    text = format_results(result)
    assert "Фон, оценка: κ_s ζ в SAMPLE, κ_g ζ в GASBK, ζ в DETBK" in text
    labels = [f"{role} ROI {roi}" for role in ("SAMPLE", "GASBK", "DETBK") for roi in (3, 8)]
    background = model.M[:, -2:] @ solution.theta[-2:]
    for i, label in enumerate(labels):
        row = next(line for line in text.splitlines() if line.startswith(label))
        observed, estimated_background, mu, residual = map(float, row[len(label):].split())
        assert observed == model.x[i]
        assert estimated_background == pytest.approx(background[i], rel=1e-10)
        assert mu == pytest.approx(solution.mu[i], rel=1e-10)
        assert residual == pytest.approx(model.x[i] - solution.mu[i], abs=1e-10)
    assert not np.allclose(solution.theta[-2:], model.x[-2:])
    correlations = text.split("6. КОРРЕЛЯЦИОННЫЕ МАТРИЦЫ ПИРСОНА", 1)[1].split("7. ПРЕДУПРЕЖДЕНИЯ", 1)[0]
    assert "Параметры theta=[beta; gamma; zeta]" in correlations
    assert "zeta[0]" in correlations and "zeta[1]" in correlations


def test_nonnegative_report_includes_background_in_physical_constraints():
    text = format_results(joint_result(method="nnls"))
    assert "Неотрицательность наложена на z=[a_V;gamma;zeta]" in text
    assert "Условие beta ≥ 0 не используется вместо a_V ≥ 0" in text


def test_joint_background_gui_keeps_activity_tables_and_updates_coordinate_text(app):
    result = joint_result()
    window = MainWindow()
    try:
        html = window.generate_activities_html(result)
        assert html.count("<td width='49%' valign='top'>") == 2
        assert "1. АКТИВНОСТЬ В ДЕТЕКТОРЕ" in html
        assert "2. ОБЪЁМНАЯ АКТИВНОСТЬ" in html
        assert "[a_V; gamma; zeta]" in html
        assert "свободная оценка a_V &gt; L_C" in html
        assert "#fde2e2" in html
        window.on_processing_finished(result)
        displayed = window.results_output.toPlainText()
        assert displayed.index("2. ОБЪЁМНАЯ АКТИВНОСТЬ") < displayed.index("Координаты theta:")
        assert "beta_gamma_zeta" in displayed
    finally:
        window.close()


@pytest.mark.parametrize("method", ["wls", "nnls"])
def test_physical_npz_without_activity_transforms_exports_available_concentration(tmp_path, method):
    source = joint_result(activity_coordinates=True)["model"]
    path = tmp_path / "direct-physical-model.npz"
    np.savez(path, M=source.M, x=source.x, n_sample_parameters=1,
             n_detector_parameters=2, coordinates="a_V_gamma_zeta",
             isotope_names=["Xe133"], mdc_component_indices=[0])
    result = process_prepared_measurement(path, method=method)
    assert result["model"].T_V is None
    assert result["activities"] is None
    solution = result["fit"]
    text = format_results(result)
    final_activity = text.split("4. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ", 1)[1].split("5. ПОРОГ", 1)[0]
    assert "Активность в детекторе не рассчитана" in final_activity
    assert "из оценённых физических координат" in final_activity
    assert "статистическая; систематические вклады здесь не рассчитаны" in final_activity
    row = next(line for line in final_activity.splitlines() if line.startswith("Xe133"))
    value, sigma, relative = row[len("Xe133"):].split()
    assert float(value) == pytest.approx(solution.physical_parameters[0], rel=1e-6)
    if method == "wls":
        expected_sigma = np.sqrt(solution.physical_covariance[0, 0])
        assert float(sigma) == pytest.approx(expected_sigma, rel=1e-6)
        assert float(relative) == pytest.approx(100 * expected_sigma / abs(solution.physical_parameters[0]), rel=1e-4)
    else:
        assert sigma == relative == "—"
