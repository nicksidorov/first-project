"""Freeze new-calibration clean-background models from the original May PHD files.

Only detector 221 efficiencies are changed. Original spectra provide timing,
volume, ROI definitions and detector-background counts; fitted SAMPLE/GASBK
activities are never used to construct the clean generating background.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("XEMAT_REPO", "/workspace/first-project"))
PREVIOUS = Path(os.environ.get("XEMAT_PRIOR_CALIBRATED", "/workspace/scratch/may-calibrated-nominal-mdc"))
FORMULA_PREVIOUS = Path(os.environ.get("XEMAT_PRIOR_FORMULA", "/workspace/scratch/may-clean-background-mdc"))
sys.path.insert(0, str(REPO))

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import norm
from process_measurement import process_prepared_measurement
from xemat.detection import compute_detection_limits, model_covariance
from xemat.io import jsonable
from xemat.model import JointModel

EFFICIENCIES = [0.241169, 0.367784, 0.233937, 0.233937]
ISOTOPES = {1: "Xe135", 2: "Xe133", 3: "Xe131m", 4: "Xe133m"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("The generating model is already frozen; use a fresh study directory")
    prior = json.loads((PREVIOUS / "manifest.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert head == "69ff5a62ccaaddeb6372f7f003109ce5ff6cf333", head
    for name, expected in prior["production_sha256"].items():
        if name != "constants.py":
            assert sha256(REPO / name) == expected, name
    baseline_hashes = {}
    for label, directory in (("formula", FORMULA_PREVIOUS), ("calibrated", PREVIOUS)):
        for filename in ("manifest.json", "summary.json"):
            destination = ROOT / f"prior_{label}_{filename}"
            payload = (directory / filename).read_bytes()
            if destination.exists():
                assert destination.read_bytes() == payload
            else:
                destination.write_bytes(payload)
            baseline_hashes[destination.name] = sha256(destination)

    datasets = {}
    checks = {}
    formula_rows = []
    z = norm.ppf(.95)
    for name in ("may22", "may26"):
        old = prior["datasets"][name]
        paths = old["input_paths"]
        for key, path in paths.items():
            assert sha256(path) == old["input_sha256"][key], (name, key)
        result = process_prepared_measurement(**paths, use_fixed_roi=True,
            use_fixed_efficiencies=True, method="wls", radon_mode="legacy")
        model = result["model"]
        metadata = jsonable(result["metadata"])
        matrices = jsonable(result["calibration_matrices"])
        assert metadata["sample_efficiencies"] == EFFICIENCIES
        assert metadata["gasbk_efficiencies"] == EFFICIENCIES
        assert metadata["calibration_detector_id"] == 221
        old_model = JointModel(**old["model"])
        np.testing.assert_array_equal(model.x, old_model.x)
        np.testing.assert_array_equal(model.b, old_model.b)
        invariant_metadata = ["sample_temporal", "background_temporal", "detector_temporal",
            "time_parameters", "xenon_volume_cm3", "effective_air_volume_m3", "sample_roi_counts",
            "background_roi_counts", "detector_roi_counts", "kappa_s", "kappa_g", "radon_mode",
            "sample_roi_numbers", "gasbk_roi_numbers", "detector_roi_numbers"]
        for key in invariant_metadata:
            assert metadata[key] == old["metadata"][key], (name, key)
        for key in ("R_c0", "R_g0", "V0", "kappa_s", "kappa_g", "Gamma", "G_p", "G_s"):
            np.testing.assert_array_equal(matrices[key], old["calibration_matrices"][key])

        # Physical truth is [a_V(5), gamma(5), zeta(7)]. Background is fixed
        # from the ORIGINAL DETBK measurement, without fitting real mixtures.
        truth = np.zeros(17)
        truth[10:] = model.x[14:]
        np.testing.assert_array_equal(truth[10:], [522, 854, 494, 66, 64, 89, 97])
        S = model.physical_matrix
        assert S.shape == (21, 17)
        rank = np.linalg.matrix_rank(S / np.linalg.norm(S, axis=0))
        assert rank == 17
        np.testing.assert_array_equal(S[14:, 10:], np.eye(7))
        np.testing.assert_array_equal(S[14:, :10], np.zeros((7, 10)))
        np.testing.assert_array_equal(S[7:, :5], np.zeros((14, 5)))
        mu0 = S @ truth + model.b
        assert np.all(mu0 > 0)
        np.testing.assert_array_equal(mu0, old["mu0"])
        limits = compute_detection_limits(S, model.b, truth, truth, mode="iterative")
        assert np.all(limits.converged)

        # Independent scalar root finder checks the production fixed-point MDC.
        independent_roots = []
        residuals = []
        for col, target in enumerate(ISOTOPES):
            s0 = np.sqrt(model_covariance(S, mu0)[target, target])
            lc = z * s0
            def equation(activity):
                sigma = np.sqrt(model_covariance(S, mu0 + activity * S[:, target])[target, target])
                return activity - lc - z * sigma
            upper = max(2 * limits.MDC[col], 1e-6)
            while equation(upper) < 0:
                upper *= 2
            root = brentq(equation, 0, upper, xtol=1e-15, rtol=1e-13)
            np.testing.assert_allclose(root, limits.MDC[col], rtol=5e-9, atol=1e-14)
            np.testing.assert_allclose(lc, limits.L_C[col], rtol=1e-13)
            independent_roots.append(root)
            residuals.append(equation(limits.MDC[col]))
            formula_rows.append([name, ISOTOPES[target], limits.L_C[col] * 1000,
                limits.MDC[col] * 1000, old["nominal_MDC"][col] * 1000,
                100 * (limits.MDC[col] / old["nominal_MDC"][col] - 1)])

        # Diagnose whether a change is pure column rescaling; never assume it.
        previous_S = old_model.physical_matrix
        norms = np.linalg.norm(previous_S, axis=0)
        transform = np.linalg.lstsq(previous_S / norms, S, rcond=None)[0] / norms[:, None]
        reconstructed = previous_S @ transform
        span_residual = float(np.linalg.norm((reconstructed - S) / np.linalg.norm(S, axis=0)))
        diagonal_scale = np.sum(previous_S * S, axis=0) / np.sum(previous_S**2, axis=0)
        scale_residual = np.linalg.norm((previous_S * diagonal_scale - S) / np.linalg.norm(S, axis=0), axis=0)
        datasets[name] = dict(model=jsonable(model), truth0=truth, mu0=mu0,
            nominal_LC=limits.L_C, nominal_MDC=limits.MDC, nominal_sigma0=limits.sigma0,
            nominal_sigma_at_MDC=limits.sigma_at_MDC, iterations=limits.iterations,
            metadata=metadata, input_paths=paths, input_sha256=old["input_sha256"],
            calibration_matrices=matrices,
            source="Rebuilt directly from original PHD inputs with updated constants.py",
            original_source=old["source"], original_source_sha256=old["source_sha256"],
            previous_sample_efficiencies=old["metadata"]["sample_efficiencies"],
            previous_gasbk_efficiencies=old["metadata"]["gasbk_efficiencies"])
        checks[name] = dict(all_input_hashes_match=True, original_counts_unchanged=True,
            invariant_metadata=invariant_metadata, original_DETBK_truth=truth[10:],
            generating_null_means_unchanged=True, normalized_physical_rank=int(rank),
            independent_mdc_brentq=independent_roots, mdc_equation_residuals=residuals,
            least_squares_old_to_new_physical_transform=transform,
            relative_column_span_residual=span_residual,
            pure_scaling_column_relative_residuals=scale_residual,
            column_change_interpretation="Fresh Xe135, Xe133 and Xe131m columns rescale by their epsilon ratios. Fresh Xe133m additionally changes daughter Xe133 contribution. Memory Xe133m changes its coupled SAMPLE/GASBK shape, so the entire new joint model is not an exact invertible reparameterization of the old one. No old fitted scores may be converted by one global scaling.",
            new_chain_sensitivity_ratio=matrices["I_c"][2] / matrices["I_c"][4],
            old_chain_sensitivity_ratio=old["calibration_matrices"]["I_c"][2] / old["calibration_matrices"]["I_c"][4])

    source_files = list(prior["production_sha256"]) + ["xemat/calibration.py", "process_measurement.py"]
    manifest = dict(scope="Clean-background nominal MDC with detector 221 efficiencies updated 2026-10-22",
        datasets=datasets, alpha=.05, beta_ii=.05, master_seed=2026102401,
        coordinates="[a_V(5); gamma(5); zeta(7)]", unit="Bq/m3; output tables and plots use mBq/m3",
        isotopes=ISOTOPES, families={"wls": "nnls", "poisson": "poisson_nonnegative"},
        production_sha256={name: sha256(REPO / name) for name in source_files},
        production_commit=head, systematics=False, numpy=np.__version__, scipy=scipy.__version__,
        prior_studies={"formula": str(FORMULA_PREVIOUS), "calibrated": str(PREVIOUS)},
        prior_document_sha256=baseline_hashes,
        prior_published_commit="20ec986349cf93e9b2102a57366199bb62bc29d2",
        efficiency_update=dict(detector=221, isotope_order=list(ISOTOPES.values()),
            sample=EFFICIENCIES, gasbk=EFFICIENCIES, convention="epsilon only; branching multiplied separately"),
        generator="zeta equals original DETBK ROI counts; all other truth parameters zero except the injected fresh isotope",
        sampling="All 21 SAMPLE/GASBK/DETBK counts independently Poisson; all 17 parameters refitted",
        lc_rule="Unchanged production Lc, nonnegative pilot with target subtraction, free estimate > Lc",
        mc_mdc_mode="fixed for per-realization Lc; iterative nominal formula MDC calculated separately",
        calibration_rule="R=free target activity / per-realization standard Lc; c=R_(ceil((n+1)*.95)); strict R>c. Separate cutoff per dataset/method/isotope.",
        calibration_n_per_dataset=20000, validation_n_per_dataset=10000,
        validation_rule="Independent null validation after c frozen; no retuning using validation",
        signal_reuse="No previous fitted scores reused. Fresh model-specific simulations and refits; prior reports only baseline comparison.",
        bootstrap=dict(repetitions=2000, master_seed=2026102409, paired=True,
            total="Paired calibration row bootstrap and paired signal decisions; one c_b across each whole power curve",
            conditional="Freeze calibrated c, resample paired signal decisions"),
        extension_policy=dict(max_additional_points_per_curve=4, n_per_point=3000,
            reason="Extend independent signal grid if .95 crossing or bootstrap interval not covered; retain all previous points"),
        background_note="Original real SAMPLE/GASBK counts do not determine generating background; original DETBK means are fixed in generation, but fresh DETBK is generated and refitted",
        planned_at_utc=datetime.now(timezone.utc).isoformat(),
        signal_plan=dict(factors=[.94, 1.02, 1.10, 1.18], n_per_point=3000,
            n_total=96000,
            selection="Fixed before all study observations; grid expressed relative to newly recomputed formula MDC"),
        stage_codes=dict(calibration=10, validation=11, signal=12, extend=13, tech=14),
        technical_vectors_excluded=60)
    write_json(ROOT / "manifest.json", manifest)
    write_json(ROOT / "inputs_review.json", dict(status="passed", datasets=checks,
        prior_document_sha256=baseline_hashes, production_commit=head,
        note="Only detector 221 efficiencies changed; input observations/times/volume/base R/radon convention preserved. Formula MDC checked by an independent scalar root solver."))
    with (ROOT / "nominal_formula.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "isotope", "LC_mBq_m3", "new_formula_MDC_mBq_m3", "old_formula_MDC_mBq_m3", "change_percent"])
        writer.writerows(formula_rows)
    print(json.dumps({name: {"LC_mBq_m3": (np.asarray(data["nominal_LC"]) * 1000).tolist(),
        "MDC_mBq_m3": (np.asarray(data["nominal_MDC"]) * 1000).tolist()}
        for name, data in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()
