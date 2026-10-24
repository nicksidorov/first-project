"""Paired false detections, November 27 detector 221: unchanged XeMat kernels.

Every planned draw is retained. A single free fit and a single nonnegative
auxiliary fit per family are shared by the four isotope tests. No null fit,
threshold multiplier, power curve or iterative detection-limit calculation is
performed. The existing LC API's fixed-MDC byproduct is discarded.

Run all batches: python study.py --workers 4
Run only first formal batch: python study.py --chunks 0 --workers 1
Resume: rerun the same command; finished batches must pass hash/seed checks.
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
    """Freeze before generation; do not silently accept changed code on resume."""
    current = dict(study_sha256=sha(__file__), scenario_sha256=sha(ROOT / "scenario.json"),
                   scenario_arrays_sha256=sha(ROOT / "scenario_arrays.npz"),
                   core_sha256={name: sha(REPO / name) for name in CORE_FILES},
                   n=N, batch_size=BATCH_SIZE, target_indices=TARGETS.tolist(),
                   families=FAMILIES, fit_calls="fit(trial, method), all default settings",
                   lc_call="compute_detection_limits(S, b, free, auxiliary, component_indices=[1,2,3,4], mode='fixed')",
                   no_mdc_output=True, no_null_fit=True, no_threshold_multiplier=True,
                   no_replacement=True)
    path = ROOT / "runner_lock.json"
    if path.exists():
        if read(path) != current:
            raise RuntimeError("Frozen scenario or runner/core code changed; resume refused")
    else:
        write(path, current)
    return current


def seed_for(chunk):
    return [int(read(ROOT / "scenario.json")["master_seed"]), int(chunk)]


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
    checks = (saved["chunk"] == chunk, saved["n_generated"] == BATCH_SIZE,
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
    scenario = read(ROOT / "scenario.json")
    model = JointModel(**scenario["model"])
    with np.load(ROOT / "scenario_arrays.npz", allow_pickle=False) as source:
        mu0 = source["mu0"].copy()
        truth = source["truth_physical"].copy()
        S = source["physical_matrix"].copy()
    np.testing.assert_array_equal(S, model.physical_matrix)
    np.testing.assert_allclose(mu0, S @ truth + model.b, rtol=1e-13, atol=1e-12)
    assert S.shape == (21, 17) and np.all(mu0 > 0)
    assert np.all(truth[1:5] == 0)
    n = BATCH_SIZE
    seed = seed_for(chunk)
    counts = np.random.default_rng(seed).poisson(mu0, size=(n, 21))
    arrays = dict(x=counts, replicate_id=np.arange(chunk*n, (chunk+1)*n, dtype=np.int64),
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
    metadata = dict(chunk=chunk, n_generated=n, seed=seed,
                    bit_generator="PCG64 via numpy.random.default_rng(seed)",
                    sampling="rng.poisson(mu0, size=(1000, 21)); same x for both families",
                    runner_lock_sha256=sha(ROOT / "runner_lock.json"), npz_sha256=sha(raw),
                    elapsed_seconds=time.monotonic()-started, diagnostic_catalogue=catalogue,
                    errors=errors, summaries=summaries,
                    fit_status_legend=FIT_STATUS, lc_status_legend=LC_STATUS,
                    decision_legend=DECISION_STATUS)
    write(meta, metadata)
    print(f"DONE batch {chunk:03d}: n={n}, seconds={metadata['elapsed_seconds']:.1f}, "
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
    print(f"PLAN {len(pending)} pending formal batches, {len(pending)*BATCH_SIZE} draws, "
          f"workers={args.workers}; all selected batches belong to the same 100000-draw plan", flush=True)
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
