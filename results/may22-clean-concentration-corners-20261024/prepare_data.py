"""Reuse saved XeMat free fits; never simulate or refit observations.

The single model_covariance call below evaluates the full 17-parameter
information matrix at the fixed true mean spectrum. No detection thresholds
or constrained estimates are calculated or used by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import scipy


ISOTOPES = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
FAMILIES = {"ls": "wls", "poisson_ml": "poisson"}
SCATTER_SEED = 2026102401
N_EXPECTED = 100_000
UNITS = "mBq/m^3"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, header, rows):
    # UTF-8 BOM makes the English CSV files convenient to open in Excel.
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_matrix(path, matrix):
    write_csv(path, ["isotope"] + ISOTOPES,
              [[name] + matrix[i].tolist() for i, name in enumerate(ISOTOPES)])


def correlation(covariance):
    sd = np.sqrt(np.diag(covariance))
    if np.any(sd <= 0):
        raise ValueError("Correlation requires positive component variances")
    return covariance / np.outer(sd, sd)


def write_scenario_text(output, scenario, summaries):
    """Regenerate the short description from saved metadata only."""
    short = [
        "XeMat clean-sample concentration-estimate distributions",
        "Scenario: May 22 configuration, detector 221; the fixed responses, times, volume and true background are unchanged from the saved study.",
        "N = 100,000 saved null observations, shared between LS and free Poisson ML. No new simulation or fitting.",
        "Fresh sample and memory truth, including radon: all zero. DETBK means: " + str(scenario["true_DETBK_mean_counts"]),
        "All 21 SAMPLE/GASBK/DETBK counts were independently Poisson; all 17 parameters were fitted in the source run.",
        "Concentration order: " + ", ".join(ISOTOPES) + "; units: " + UNITS,
        "Source observation seeds: [2026102701, 0, 0, 0, 11, chunk], chunk = 0,...,99; 1,000 rows per chunk.",
        "Scatter seed: 2026102401; 5,000 shared random rows. Statistics use all valid rows, not this display subset.",
        "Live times (s): SAMPLE 50,400; GASBK 25,200; DETBK 259,200. Effective air volume (m^3): " + str(scenario["effective_air_volume_m3"]),
        "Efficiencies (SAMPLE = GASBK): " + str(scenario["efficiencies"]["sample"]) + "; branching is separate.",
        "Covariance: centered at each empirical mean, divisor N_valid - 1. Negative estimates and outliers are retained.",
        "Reference: one existing full-model covariance evaluation at true mu0; extract the xenon block only after retaining all nuisance columns.",
        "The reference is approximate for LS with observed-count weights, and is not an exact finite-sample guarantee for ML.",
        "Systematic uncertainties: not included. Per-fit covariance unavailability does not invalidate saved point estimates.",
        "Detailed model, provenance, hashes and status definitions: scenario.json and estimates.npz.",
    ]
    for name, result in summaries.items():
        short.append(f"{name}: n_valid = {result['n_valid']:,}; fit failures = {result['n_fit_failures']}; nonfinite successful fits = {result['n_success_nonfinite']}; point estimates retained despite unavailable fit covariance = {result['n_retained_without_fit_covariance']}.")
    (Path(output) / "scenario.txt").write_text("\n".join(short) + "\n", encoding="utf-8")


def collect(source, manifest, expected_mu, truth):
    """Read every saved free vector, preserving row identity and fit status."""
    plan = sorted(read_json(source / "validation_plan.json"), key=lambda j: j["chunk"])
    assert len(plan) == 100 and [j["chunk"] for j in plan] == list(range(100))
    assert sum(j["n"] for j in plan) == N_EXPECTED
    manifest_hash = sha256(source / "manifest.json")
    collected = {"x": [], "batch_chunk": [], "row_in_batch": []}
    for name in FAMILIES:
        for suffix in ("", "_success", "_covariance_available", "_full_native"):
            collected[name + suffix] = []
    provenance, seeds, free_errors = [], [], []
    for job in plan:
        assert job["dataset"] == "may22" and job["stage"] == "validation"
        assert job["target"] == 0 and job["factor"] == 0 and job["n"] == 1000
        chunk = job["chunk"]
        stem = f"may22_validation_i0_f00000_c{chunk}"
        json_path, npz_path = (source / "points" / (stem + extension)
                               for extension in (".json", ".npz"))
        metadata = read_json(json_path)
        assert all(metadata[key] == value for key, value in job.items())
        assert metadata["manifest_sha256"] == manifest_hash
        assert metadata["npz_sha256"] == sha256(npz_path)
        assert metadata["raw_activity_unit"] == "Bq/m3"
        assert np.array_equal(np.asarray(metadata["truth"]), truth)
        assert np.array_equal(np.asarray(metadata["mu"]), expected_mu)
        seed = [manifest["master_seed"], 0, 0, 0, manifest["stage_codes"]["validation"], chunk]
        assert metadata["seed"] == seed
        seeds.append(seed)
        for error in metadata["errors"]:
            if error["phase"] == "free":
                free_errors.append({"chunk": chunk, **error})
        with np.load(npz_path, allow_pickle=False) as data:
            x = data["x"]
            assert x.shape == (1000, 21) and np.issubdtype(x.dtype, np.integer)
            assert np.all(x >= 0)
            assert np.array_equal(data["indices"], np.arange(1000))
            collected["x"].append(x)
            collected["batch_chunk"].append(np.full(1000, chunk, dtype=np.int32))
            collected["row_in_batch"].append(data["indices"])
            for name, original in FAMILIES.items():
                free = data[original + "_free"]
                success = data[original + "_free_success"]
                available = data[original + "_free_covariance_available"]
                assert free.shape == (1000, 17)
                assert success.shape == available.shape == (1000,)
                assert success.dtype == available.dtype == np.dtype(bool)
                # First five native coordinates are fresh Bq/m^3, including Rn.
                # The four xenon columns become mBq/m^3. No sign clipping occurs.
                collected[name].append(free[:, 1:5] * 1000.0)
                collected[name + "_full_native"].append(free)
                collected[name + "_success"].append(success)
                collected[name + "_covariance_available"].append(available)
        provenance.append({"chunk": chunk, "n": 1000, "seed": seed,
                           "json_file": "points/" + json_path.name,
                           "json_sha256": sha256(json_path),
                           "npz_file": "points/" + npz_path.name,
                           "npz_sha256": metadata["npz_sha256"]})
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    assert all(len(values) == N_EXPECTED for values in arrays.values())
    arrays["row_id"] = np.arange(N_EXPECTED, dtype=np.int64)
    arrays["seed_table"] = np.asarray(seeds, dtype=np.int64)
    return arrays, provenance, free_errors


def prepare(source: Path, repo: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "estimates.npz").exists():
        raise FileExistsError("estimates.npz already exists: use a new output directory to rebuild")
    manifest = read_json(source / "manifest.json")
    source_hashes = {name: sha256(source / name)
                     for name in ("manifest.json", "validation_plan.json", "study.py")}
    for name, expected in manifest["production_sha256"].items():
        assert sha256(repo / name) == expected, f"Production source differs: {name}"
    for name, expected in read_json(source / "run_lock.json").items():
        assert sha256(source / name) == expected, f"Saved run lock differs: {name}"

    sys.path.insert(0, str(repo))
    from xemat.model import JointModel
    from xemat.detection import model_covariance

    dataset = manifest["datasets"]["may22"]
    model = JointModel(**dataset["model"])
    S = model.physical_matrix
    truth = np.asarray(dataset["truth0"], dtype=float)
    assert S.shape == (21, 17) and np.array_equal(truth[:10], np.zeros(10))
    assert model.physical_coordinate_label == "[a_V; gamma; zeta]"
    assert model.n_sample_parameters == 5 and model.n_detector_parameters == 7
    mu0 = S @ truth + model.b
    assert np.array_equal(mu0, np.asarray(dataset["mu0"])) and np.all(mu0 > 0)
    arrays, batches, free_errors = collect(source, manifest, mu0, truth)

    # Exactly one covariance evaluation, using the existing full-model formula.
    # Keep all 17 nuisance/signal columns first, then extract the xenon block.
    covariance_full_native = model_covariance(S, mu0)
    reference = covariance_full_native[1:5, 1:5] * 1_000_000.0
    reference_sd = np.sqrt(np.diag(reference))
    reference_correlation = correlation(reference)

    summaries = {}
    summary_rows, status_rows, comparison_rows = [], [], []
    for name in FAMILIES:
        values = arrays[name]
        success = arrays[name + "_success"]
        finite = np.all(np.isfinite(values), axis=1)
        valid = success & finite
        status = np.full(N_EXPECTED, "success", dtype="U32")
        status[~success] = "fit_failure"
        status[success & ~finite] = "nonfinite_estimate"
        arrays[name + "_status"] = status
        arrays[name + "_valid"] = valid
        retained = values[valid]
        n_valid = len(retained)
        if n_valid < 2:
            raise ValueError(f"Fewer than two available free estimates for {name}")
        mean = retained.mean(axis=0)
        centered = retained - mean
        empirical = centered.T @ centered / (n_valid - 1)
        sd = np.sqrt(np.diag(empirical))
        corr = correlation(empirical)
        difference = empirical - reference
        normalized = difference / np.outer(reference_sd, reference_sd)
        diag_ratio = np.diag(empirical) / np.diag(reference)
        unavailable_cov = int(np.count_nonzero(~arrays[name + "_covariance_available"]))
        summaries[name] = {
            "n_attempted": N_EXPECTED, "n_valid": n_valid,
            "n_fit_failures": int(np.count_nonzero(~success)),
            "n_success_nonfinite": int(np.count_nonzero(success & ~finite)),
            "n_excluded": int(np.count_nonzero(~valid)),
            "n_fit_covariance_unavailable": unavailable_cov,
            "n_retained_without_fit_covariance": int(np.count_nonzero(valid & ~arrays[name + "_covariance_available"])),
            "mean": mean.tolist(), "empirical_sd": sd.tolist(),
            "empirical_covariance": empirical.tolist(),
            "empirical_correlation": corr.tolist(),
            "sd_ratio_to_reference": (sd / reference_sd).tolist(),
            "variance_ratio_to_reference": diag_ratio.tolist(),
            "covariance_difference": difference.tolist(),
            "standardized_covariance_difference": normalized.tolist(),
            "correlation_difference": (corr - reference_correlation).tolist(),
            "n_negative_by_isotope": np.count_nonzero(retained < 0, axis=0).tolist(),
            "minimum": retained.min(axis=0).tolist(), "maximum": retained.max(axis=0).tolist(),
            "max_abs_standardized_covariance_difference": float(np.max(np.abs(normalized))),
        }
        arrays[name + "_mean"] = mean
        arrays[name + "_empirical_covariance"] = empirical
        arrays[name + "_empirical_correlation"] = corr
        arrays[name + "_empirical_sd"] = sd
        for i, isotope in enumerate(ISOTOPES):
            summary_rows.append([name, isotope, UNITS, N_EXPECTED, n_valid,
                                 summaries[name]["n_fit_failures"], summaries[name]["n_success_nonfinite"],
                                 0.0, mean[i], sd[i], reference_sd[i], sd[i]/reference_sd[i],
                                 diag_ratio[i], int(np.count_nonzero(retained[:, i] < 0)),
                                 retained[:, i].min(), retained[:, i].max()])
            for j, isotope_j in enumerate(ISOTOPES):
                comparison_rows.append([name, isotope, isotope_j, "(mBq/m^3)^2",
                                        empirical[i, j], reference[i, j], difference[i, j],
                                        normalized[i, j], corr[i, j], reference_correlation[i, j],
                                        corr[i, j] - reference_correlation[i, j]])
        status_rows.append([name, N_EXPECTED, int(success.sum()), n_valid,
                            summaries[name]["n_fit_failures"], summaries[name]["n_success_nonfinite"],
                            unavailable_cov, summaries[name]["n_retained_without_fit_covariance"]])
        write_matrix(output / f"covariance_{name}.csv", empirical)
        write_matrix(output / f"correlation_{name}.csv", corr)

    common_valid = np.flatnonzero(arrays["ls_valid"] & arrays["poisson_ml_valid"])
    if len(common_valid) < 5000:
        raise ValueError("Fewer than 5000 paired valid rows for a shared scatter subset")
    # This seed selects display points only; it does not simulate observations.
    scatter = np.sort(np.random.default_rng(SCATTER_SEED).choice(common_valid, 5000, replace=False))
    arrays.update({
        "scatter_indices": scatter, "scatter_seed": np.asarray(SCATTER_SEED, dtype=np.int64),
        "isotopes": np.asarray(ISOTOPES), "units": np.asarray(UNITS),
        "covariance_units": np.asarray("(mBq/m^3)^2"),
        "truth_concentration": np.zeros(4), "truth_physical_native": truth, "truth_native": truth,
        "physical_matrix_native": S, "model_M_native": model.M,
        "model_T_V": model.T_V, "model_b": model.b, "mu0": mu0,
        "reference_covariance": reference, "reference_correlation": reference_correlation,
        "reference_sd": reference_sd, "reference_covariance_full_native": covariance_full_native,
        "native_parameter_units": np.asarray(["Bq/m^3 (Rn auxiliary)"] + ["Bq/m^3"] * 4
                                               + ["native memory count coefficient"] * 5 + ["DETBK counts"] * 7),
        "full_native_covariance_units_note": np.asarray("Mixed units: entry(i,j) is the product of native_parameter_units[i] and [j]. Only reference_covariance is wholly (mBq/m^3)^2."),
        "physical_coordinate_label": np.asarray(model.physical_coordinate_label),
        "source_manifest_sha256": np.asarray(source_hashes["manifest.json"]),
        "source_stage": np.asarray("validation"),
        "status_legend": np.asarray(["success: finite free estimate from successful fit",
                                     "fit_failure: saved solver success flag is false",
                                     "nonfinite_estimate: fit success but at least one nonfinite xenon estimate"]),
    })
    np.savez_compressed(output / "estimates.npz", **arrays)
    write_csv(output / "summary_statistics.csv",
              ["method", "isotope", "concentration_units", "n_attempted", "n_valid",
               "n_fit_failures", "n_success_nonfinite", "true_concentration", "empirical_mean",
               "empirical_sd_ddof1", "reference_sd", "empirical_sd_over_reference_sd",
               "empirical_variance_over_reference_variance", "n_negative", "minimum", "maximum"],
              summary_rows)
    write_csv(output / "calculation_status.csv",
              ["method", "n_attempted", "n_fit_success", "n_valid", "n_fit_failures",
               "n_success_nonfinite", "n_fit_covariance_unavailable", "n_retained_without_fit_covariance"],
              status_rows)
    write_csv(output / "covariance_comparison.csv",
              ["method", "isotope_i", "isotope_j", "covariance_units", "empirical_covariance",
               "reference_covariance", "covariance_difference", "difference_over_reference_sd_i_sd_j",
               "empirical_correlation", "reference_correlation", "correlation_difference"], comparison_rows)
    write_matrix(output / "covariance_reference.csv", reference)
    write_matrix(output / "correlation_reference.csv", reference_correlation)

    stats = {"isotope_order": ISOTOPES, "concentration_units": UNITS,
             "covariance_units": "(mBq/m^3)^2", "covariance_divisor": "N_valid - 1",
             "centering": "empirical mean of each method's full valid sample",
             "methods": summaries, "reference_covariance": reference.tolist(),
             "reference_sd": reference_sd.tolist(), "reference_correlation": reference_correlation.tolist(),
             "scatter_count": 5000, "scatter_shared_row_indices": True,
             "estimates_sha256": sha256(output / "estimates.npz")}
    write_json(output / "statistics.json", stats)

    metadata = dataset["metadata"]
    scenario = {
        "scenario": "May 22 clean sample, detector 221; reused 100,000 validation observations",
        "data_source": "saved full free estimates; no new observations or fits",
        "n_observations": N_EXPECTED, "source_dataset": "may22", "source_stage": "validation",
        "isotope_order": ISOTOPES, "concentration_units": UNITS,
        "physical_coordinates": "[a_V; gamma; zeta]",
        "physical_parameter_order": ["Rn222_aux_fresh", "Xe135_fresh", "Xe133_fresh", "Xe131m_fresh", "Xe133m_fresh",
                                     "Rn222_memory", "Xe135_memory", "Xe133_memory", "Xe131m_memory", "Xe133m_memory"]
                                    + [f"zeta_ROI_{r}" for r in metadata["detector_roi_numbers"]],
        "native_units_note": "First five native parameters are Bq/m^3 (Rn is auxiliary); gamma uses the existing XeMat memory coordinates; zeta is expected DETBK counts. Only xenon columns 1:5 are multiplied by 1000 in ls/poisson_ml.",
        "true_fresh_and_memory_parameters": truth[:10].tolist(),
        "true_DETBK_mean_counts": truth[10:].tolist(), "true_mean_spectrum": mu0.tolist(),
        "observation_blocks": ["SAMPLE", "GASBK", "DETBK"], "roi_order": metadata["sample_roi_numbers"],
        "full_model_shape": list(S.shape),
        "detector_id": metadata["input_detector_id"],
        "time_parameters_seconds": metadata["time_parameters"],
        "effective_air_volume_m3": metadata["effective_air_volume_m3"],
        "xenon_volume_cm3": metadata["xenon_volume_cm3"],
        "efficiencies": manifest["efficiency_update"],
        "calibration_matrices": dataset["calibration_matrices"],
        "systematic_errors_used": False, "radon_model": metadata["radon_mode"],
        "reference_covariance": {
            "function": "xemat.detection.model_covariance", "number_of_calls": 1,
            "formula": "K_full=(S_V.T @ diag(1/mu0) @ S_V)^(-1)",
            "implementation": "existing column-scaled SVD; all 17 columns retained",
            "xenon_block": "K_full[1:5,1:5] * 1e6, units (mBq/m^3)^2",
            "note": "Expected-information reference at fixed true mu0. Approximate benchmark for LS with observation-dependent weights; not a claim of exact finite-sample covariance for either method.",
        },
        "ls_weights": "Existing W_ii=1/x_i for positive counts; W_ii=1 for zero counts; raw counts unchanged.",
        "poisson_ml": "Existing free physical parameters, with admissible expected counts; same saved observations as LS.",
        "row_order": "chunk = 0,...,99, then saved row = 0,...,999 within each chunk",
        "source_observation_seed": {"master": manifest["master_seed"], "stage_code": 11,
                                    "rule": "[2026102701, 0, 0, 0, 11, chunk]", "chunks": [0, 99]},
        "scatter_seed": SCATTER_SEED, "scatter_selection": "sorted default_rng(seed).choice(common_valid_row_ids, 5000, replace=False)",
        "fit_status_meaning": {"success": "finite free estimate and saved success flag",
                               "fit_failure": "saved success flag false",
                               "nonfinite_estimate": "success flag true but nonfinite four-vector"},
        "fit_covariance_availability": "Retained as a separate diagnostic; never used to remove otherwise valid estimates.",
        "source_root": str(source), "production_commit": manifest["production_commit"],
        "production_sha256": manifest["production_sha256"], "source_sha256": source_hashes,
        "source_batches": batches, "source_free_fit_errors": free_errors,
        "estimates_sha256": stats["estimates_sha256"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    write_json(output / "scenario.json", scenario)
    write_scenario_text(output, scenario, summaries)
    for name, expected in manifest["production_sha256"].items():
        assert sha256(repo / name) == expected, name
    for name, expected in source_hashes.items():
        assert sha256(source / name) == expected, name
    print(json.dumps({"n": N_EXPECTED, "source_batches_verified": len(batches),
                      "reference_covariance_calls": 1, "methods": summaries,
                      "npz_sha256": stats["estimates_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/workspace/scratch/may22-clean-mdc-100k-20261023"))
    parser.add_argument("--repo", type=Path, default=Path(os.environ.get("XEMAT_REPO", "/workspace/first-project")))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.repo.resolve(), args.output.resolve())
