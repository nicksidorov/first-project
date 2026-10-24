"""Variant B: refit saved observations with equal xenon memory/base responses.

No observations are generated in this script. All 100000 original variant-A
observation vectors and their IDs are reused, including A's failed fits.
A single free fit and a single nonnegative
auxiliary fit per family are shared by the four isotope tests. No null fit,
threshold multiplier, power curve or iterative detection-limit calculation is
performed. Each LC is calculated with model B and its own auxiliary estimate;
the existing LC API's fixed-MDC byproduct is discarded.

Run all batches: python study.py --workers 4
Run only first formal batch: python study.py --chunks 0 --workers 1
Resume: rerun the same command; finished batches must pass input/hash checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = ROOT / "kernel_snapshot" if (ROOT / "kernel_snapshot").is_dir() else Path("/workspace/first-project")
REPO = Path(os.environ.get("XEMAT_REPO", str(DEFAULT_REPO))).resolve()
sys.path.insert(0, str(REPO))

import numpy as np
from xemat.detection import compute_detection_limits
from xemat.model import JointModel
from xemat.solvers import fit

N = 100_000
BATCH_SIZE = 1_000
TARGETS = np.array([1, 2, 3, 4], dtype=np.int64)
ISOTOPES = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
FAMILIES = {"wls": "nnls", "poisson": "poisson_nonnegative"}
SOURCE_REPLICATES_SHA256 = "ede7d13cb1c7d3893550088b7b5ddc1232660f8831a780ac637cd05405da7804"
DEFAULT_SOURCE_REPLICATES = Path("/workspace/scratch/nov27-strong-memory-lc-100k-20261024/replicates.npz")
_INPUT_CACHE = None
CORE_FILES = ["constants.py", "ims_parser.py", "process_measurement.py",
              "xemat/model.py", "xemat/solvers.py", "xemat/detection.py",
              "xemat/calibration.py", "xemat/activities.py", "xemat/detectors.py"]
FIT_STATUS = {-2: "not attempted", -1: "exception", 0: "solver returned success=False",
              1: "successful finite physical estimate", 2: "missing/nonfinite/wrong-size physical estimate"}
LC_STATUS = {-2: "not attempted: required fit estimate unavailable", -1: "LC exception",
             0: "nonfinite/negative LC", 1: "available LC"}
DECISION_STATUS = {-1: "unavailable decision; never count as nondetection",
                    0: "available: free estimate <= LC", 1: "available: free estimate > LC"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def get_lock():
    """Freeze before refitting; never silently accept changed data or code."""
    current = dict(study_sha256=sha(__file__), scenario_sha256=sha(ROOT / "scenario_B.json"),
                   scenario_arrays_sha256=sha(ROOT / "scenario_arrays_B.npz"),
                   observations_sha256=sha(ROOT / "observations.npz"),
                   source_replicates_sha256=SOURCE_REPLICATES_SHA256,
                   core_sha256={name: sha(REPO / name) for name in CORE_FILES},
                   n=N, batch_size=BATCH_SIZE, target_indices=TARGETS.tolist(),
                   variant="B", no_new_observations=True, n_generated=0, n_reused=N,
                   families=FAMILIES, fit_calls="fit(trial, method), all default settings",
                   lc_call="compute_detection_limits(S, b, free, auxiliary, component_indices=[1,2,3,4], mode='fixed')",
                   no_mdc_output=True, no_null_fit=True, no_threshold_multiplier=True,
                   no_replacement=True)
    scenario = read(ROOT / "scenario_B.json")
    if scenario["n"] != N or scenario["batch_size"] != BATCH_SIZE:
        raise RuntimeError("Scenario dimensions differ from the fixed paired plan")
    if scenario["source_replicates_sha256"] != SOURCE_REPLICATES_SHA256:
        raise RuntimeError("Scenario refers to a different variant-A ensemble")
    for name, digest in current["core_sha256"].items():
        if scenario["production_sha256"][name] != digest:
            raise RuntimeError("Kernel differs from the unchanged source version: " + name)
    if scenario.get("observations_sha256", current["observations_sha256"]) != current["observations_sha256"]:
        raise RuntimeError("Saved observation arrays differ from frozen scenario")
    path = ROOT / "runner_lock.json"
    if path.exists():
        if read(path) != current:
            raise RuntimeError("Frozen scenario or runner/core code changed; resume refused")
    else:
        write(path, current)
    return current


def saved_inputs():
    """Load the existing observations once per worker; no random sampling."""
    global _INPUT_CACHE
    if _INPUT_CACHE is None:
        with np.load(ROOT / "observations.npz", allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in
                    ("x", "replicate_id", "seed_table", "source_replicates_sha256")}
        if str(data["source_replicates_sha256"].item()) != SOURCE_REPLICATES_SHA256:
            raise RuntimeError("Saved observations have the wrong baseline provenance")
        if data["x"].shape != (N, 21) or not np.issubdtype(data["x"].dtype, np.integer) or np.any(data["x"] < 0):
            raise ValueError("Saved observations must contain all 100000 original integer count vectors")
        np.testing.assert_array_equal(data["replicate_id"], np.arange(N))
        if data["seed_table"].shape != (N // BATCH_SIZE, 2):
            raise ValueError("Original seed table has the wrong shape")
        source_path = Path(os.environ.get("XEMAT_BASELINE_REPLICATES", str(DEFAULT_SOURCE_REPLICATES)))
        # The portable archive has the observations and their frozen digest.
        # When the original baseline is present, also verify the entire source
        # file and exact saved x/ID/seed equality before any refitting.
        if source_path.is_file():
            if sha(source_path) != SOURCE_REPLICATES_SHA256:
                raise RuntimeError("Original variant-A replicates have changed")
            with np.load(source_path, allow_pickle=False) as original:
                for key in ("x", "replicate_id", "seed_table"):
                    np.testing.assert_array_equal(data[key], original[key])
        for value in data.values():
            value.setflags(write=False)
        _INPUT_CACHE = data
    return _INPUT_CACHE


def seed_for(chunk):
    """Original A seed for provenance only; it is never used for sampling."""
    return saved_inputs()["seed_table"][chunk].tolist()


def batch_paths(chunk):
    stem = f"batch_{chunk:03d}"
    return ROOT / "points" / (stem + ".npz"), ROOT / "points" / (stem + ".json")


def finished(chunk, lock):
    raw, meta = batch_paths(chunk)
    if not raw.exists() and not meta.exists():
        return False
    if not raw.exists() or not meta.exists():
        raise RuntimeError(f"Incomplete checkpoint {chunk}; do not replace draws automatically")
    saved = read(meta)
    checks = (saved["chunk"] == chunk, saved["n_reused"] == BATCH_SIZE,
              saved["n_generated"] == 0, saved["variant"] == "B",
              saved["seed"] == seed_for(chunk), saved["npz_sha256"] == sha(raw),
              saved["runner_lock_sha256"] == sha(ROOT / "runner_lock.json"))
    if not all(checks):
        raise RuntimeError(f"Checkpoint verification failed: {chunk}")
    return True


def run_batch(chunk):
    started = time.monotonic()
    lock = get_lock()
    if finished(chunk, lock):
        return dict(chunk=chunk, already_finished=True)
    scenario = read(ROOT / "scenario_B.json")
    model = JointModel(**scenario["model"])
    with np.load(ROOT / "scenario_arrays_B.npz", allow_pickle=False) as source:
        S = source["physical_matrix"].copy()
    np.testing.assert_array_equal(S, model.physical_matrix)
    assert S.shape == (21, 17)
    n = BATCH_SIZE
    seed = seed_for(chunk)
    inputs = saved_inputs()
    selection = slice(chunk*n, (chunk+1)*n)
    counts = inputs["x"][selection].copy()
    arrays = dict(x=counts, replicate_id=inputs["replicate_id"][selection].copy(),
                  chunk=np.full(n, chunk, dtype=np.int16), row_in_batch=np.arange(n, dtype=np.int16),
                  isotope_order=np.array(ISOTOPES), concentration_units=np.array("mBq/m3"),
                  physical_coordinate_label=np.array(model.physical_coordinate_label),
                  seed=np.asarray(seed, dtype=np.int64))
    for family in FAMILIES:
        for phase in ("free", "auxiliary"):
            prefix = f"{family}_{phase}"
            arrays[prefix + "_physical"] = np.full((n, 17), np.nan)
            arrays[prefix + "_status"] = np.full(n, -2, dtype=np.int8)
            arrays[prefix + "_diagnostic_code"] = np.zeros(n, dtype=np.int32)
            arrays[prefix + "_iterations"] = np.full(n, -1, dtype=np.int32)
            arrays[prefix + "_objective"] = np.full(n, np.nan)
            arrays[prefix + "_rank"] = np.full(n, -1, dtype=np.int16)
            arrays[prefix + "_condition_number"] = np.full(n, np.nan)
        arrays[family + "_free_covariance_available"] = np.zeros(n, dtype=bool)
        arrays[family + "_concentrations"] = np.full((n, 4), np.nan)
        arrays[family + "_LC"] = np.full((n, 4), np.nan)
        arrays[family + "_LC_status"] = np.full((n, 4), -2, dtype=np.int8)
        arrays[family + "_LC_diagnostic_code"] = np.zeros((n, 4), dtype=np.int32)
        arrays[family + "_decision"] = np.full((n, 4), -1, dtype=np.int8)

    catalogue = [{"message": "No diagnostic"}]
    lookup = {json.dumps(catalogue[0], sort_keys=True, ensure_ascii=False): 0}
    errors = []

    def code(diagnostic):
        key = json.dumps(diagnostic, sort_keys=True, ensure_ascii=False)
        if key not in lookup:
            lookup[key] = len(catalogue)
            catalogue.append(diagnostic)
        return lookup[key]

    def record_error(row, family, phase, message, **extra):
        errors.append(dict(replicate_id=int(arrays["replicate_id"][row]), row_in_batch=row,
                           family=family, phase=phase, message=message, **extra))

    def fitting(row, trial, family, phase, method):
        prefix = f"{family}_{phase}"
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = fit(trial, method)  # No weight/tolerance/optimizer overrides.
            physical = result.physical_parameters
            valid = (physical is not None and np.shape(physical) == (17,)
                     and bool(np.all(np.isfinite(physical))))
            if physical is not None and np.shape(physical) == (17,):
                arrays[prefix + "_physical"][row] = physical
            if phase == "free":
                arrays[family + "_concentrations"][row] = arrays[prefix + "_physical"][row, 1:5] * 1000
                arrays[family + "_free_covariance_available"][row] = result.physical_covariance is not None
            status = 1 if result.success and valid else (0 if not result.success else 2)
            arrays[prefix + "_status"][row] = status
            arrays[prefix + "_iterations"][row] = int(result.iterations)
            arrays[prefix + "_objective"][row] = result.objective
            arrays[prefix + "_rank"][row] = int(result.rank)
            arrays[prefix + "_condition_number"][row] = result.condition_number
            diagnostic = dict(method=method, success=bool(result.success), message=str(result.message),
                              warnings=list(result.warnings),
                              python_warnings=[f"{w.category.__name__}: {w.message}" for w in caught])
            if not valid:
                diagnostic["invalid_physical_estimate"] = True
            arrays[prefix + "_diagnostic_code"][row] = code(diagnostic)
            if status != 1:
                record_error(row, family, phase, str(result.message), status=status,
                             invalid_physical_estimate=not valid)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            arrays[prefix + "_status"][row] = -1
            arrays[prefix + "_diagnostic_code"][row] = code(dict(method=method, exception=message,
                python_warnings=[f"{w.category.__name__}: {w.message}" for w in caught]))
            record_error(row, family, phase, message, status=-1)

    def store_lc(row, family, columns, result):
        values = np.asarray(result.L_C, dtype=float)
        arrays[family + "_LC"][row, columns] = values * 1000
        good = np.isfinite(values) & (values >= 0)
        arrays[family + "_LC_status"][row, columns] = good.astype(np.int8)
        for j, col in enumerate(columns):
            if not good[j]:
                message = "Returned LC is nonfinite or negative"
                arrays[family + "_LC_diagnostic_code"][row, col] = code(dict(message=message))
                record_error(row, family, "LC", message, target_index=int(TARGETS[col]))

    for row, x in enumerate(counts):
        trial = JointModel(model.M, x, model.b, model.n_sample_parameters,
                           model.T_c, model.coordinates, model.n_detector_parameters)
        for family, auxiliary_method in FAMILIES.items():
            # Run the two phases independently: one failure must not erase the
            # other phase's estimate or the other family's result.
            fitting(row, trial, family, "free", family)
            fitting(row, trial, family, "auxiliary", auxiliary_method)
            free = arrays[family + "_free_physical"][row]
            auxiliary = arrays[family + "_auxiliary_physical"][row]
            if (np.all(np.isfinite(free)) and arrays[family + "_auxiliary_status"][row] == 1):
                try:
                    result = compute_detection_limits(S, model.b, free, auxiliary,
                        component_indices=TARGETS, mode="fixed")
                    store_lc(row, family, np.arange(4), result)
                except Exception as exc:
                    # An unavailable target must not discard a different
                    # available target. Reuse exactly the SAME two fits.
                    message = f"{type(exc).__name__}: {exc}"
                    record_error(row, family, "LC_combined_call", message)
                    for col, target in enumerate(TARGETS):
                        try:
                            result = compute_detection_limits(S, model.b, free, auxiliary,
                                component_indices=[int(target)], mode="fixed")
                            store_lc(row, family, [col], result)
                        except Exception as single_exc:
                            individual = f"{type(single_exc).__name__}: {single_exc}"
                            arrays[family + "_LC_status"][row, col] = -1
                            arrays[family + "_LC_diagnostic_code"][row, col] = code(dict(exception=individual))
                            record_error(row, family, "LC", individual, target_index=int(target))
            if arrays[family + "_free_status"][row] == 1 and arrays[family + "_auxiliary_status"][row] == 1:
                valid = arrays[family + "_LC_status"][row] == 1
                # Compare in one explicit unit, mBq/m3; no multiplier c.
                arrays[family + "_decision"][row, valid] = (
                    arrays[family + "_concentrations"][row, valid] > arrays[family + "_LC"][row, valid])

    summaries = {}
    for family in FAMILIES:
        decision = arrays[family + "_decision"]
        no_cov = ~arrays[family + "_free_covariance_available"]
        summaries[family] = dict(
            free_status_counts={str(k): int(v) for k, v in Counter(arrays[family + "_free_status"].tolist()).items()},
            auxiliary_status_counts={str(k): int(v) for k, v in Counter(arrays[family + "_auxiliary_status"].tolist()).items()},
            n_available=(decision >= 0).sum(axis=0).tolist(),
            n_false_detections=(decision == 1).sum(axis=0).tolist(),
            n_failed_decisions=(decision < 0).sum(axis=0).tolist(),
            n_successful_free_covariance_unavailable=int(np.sum(no_cov & (arrays[family + "_free_status"] == 1))),
            n_covariance_unavailable_but_decision_available=((decision >= 0) & no_cov[:, None]).sum(axis=0).tolist())
    raw, meta = batch_paths(chunk)
    raw.parent.mkdir(exist_ok=True)
    temporary = raw.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, raw)
    metadata = dict(chunk=chunk, variant="B", n_observations=n, n_reused=n, n_generated=0, seed=seed,
                    seed_role="Original A seed retained only for provenance; no generation in B",
                    sampling="Existing observations.npz x[1000*chunk:1000*(chunk+1)], same x for both families and variants",
                    observations_sha256=lock["observations_sha256"], source_replicates_sha256=SOURCE_REPLICATES_SHA256,
                    runner_lock_sha256=sha(ROOT / "runner_lock.json"), npz_sha256=sha(raw),
                    elapsed_seconds=time.monotonic()-started, diagnostic_catalogue=catalogue,
                    errors=errors, summaries=summaries,
                    fit_status_legend=FIT_STATUS, lc_status_legend=LC_STATUS,
                    decision_legend=DECISION_STATUS)
    write(meta, metadata)
    print(f"DONE B batch {chunk:03d}: reused={n}, new_draws=0, seconds={metadata['elapsed_seconds']:.1f}, "
          f"error_records={len(errors)}, failed_decisions="
          f"{sum(sum(s['n_failed_decisions']) for s in summaries.values())}", flush=True)
    return dict(chunk=chunk, elapsed_seconds=metadata["elapsed_seconds"], error_records=len(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks", help="Comma-separated formal chunk indices (0..99); default all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    chunks = list(range(N // BATCH_SIZE)) if args.chunks is None else [int(v) for v in args.chunks.split(",")]
    if not chunks or len(set(chunks)) != len(chunks) or any(v < 0 or v >= N // BATCH_SIZE for v in chunks):
        parser.error("Invalid/duplicate formal chunk indices")
    lock = get_lock()
    pending = [chunk for chunk in chunks if not finished(chunk, lock)]
    print(f"PLAN B: {len(pending)} pending batches, {len(pending)*BATCH_SIZE} saved observation vectors to refit, "
          f"workers={args.workers}; no new draws; every original A ID retained", flush=True)
    if args.workers == 1:
        for chunk in pending:
            run_batch(chunk)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_batch, chunk) for chunk in pending]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
