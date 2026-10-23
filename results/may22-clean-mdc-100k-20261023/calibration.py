"""Freeze clean-background critical coefficients using calibration draws only.

The multiplier c is an order statistic of R = A_hat_free / Lc_standard.
Validation data are deliberately absent from this module's read paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parent
FAMILIES = ("wls", "poisson")
DATASETS = ("may22",)
TARGETS = (1, 2, 3, 4)
ISOTOPES = ("Xe135", "Xe133", "Xe131m", "Xe133m")
MASTER = 2026102701
BOOTSTRAP_MASTER = 2026102709
BOOTSTRAP_REPETITIONS = 2000
CALIBRATION_N = 100000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_rank(n: int, quantile: float = .95) -> int:
    """One-based rank giving marginal tail probability <= 1-quantile."""
    k = math.ceil((n + 1) * quantile)
    if not 1 <= k <= n:
        raise ValueError("Calibration sample is too small for a finite cutoff")
    return k


def quantile_interval(sorted_values: np.ndarray, confidence: float) -> dict:
    """Binomial inversion interval for the population .95 quantile."""
    n = len(sorted_values)
    delta = 1 - confidence
    lower_rank = int(binom.ppf(delta / 2, n, .95))
    upper_rank = int(binom.ppf(1 - delta / 2, n, .95)) + 1
    if not 1 <= lower_rank <= upper_rank <= n:
        raise ValueError("Non-finite quantile interval; use more calibration rows")
    return dict(
        interval=[float(sorted_values[lower_rank - 1]), float(sorted_values[upper_rank - 1])],
        order_ranks=[lower_rank, upper_rank],
        confidence=confidence,
        binomial_coverage=float(binom.cdf(upper_rank - 1, n, .95)
                                - binom.cdf(lower_rank - 1, n, .95)),
    )


def coefficient(values: np.ndarray, target: int, isotope: str) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Every calibration ratio must be finite")
    n = len(values)
    k = order_rank(n)
    ordered = np.sort(values)
    c = float(ordered[k - 1])
    if c <= 0:
        raise ValueError("Expected a positive calibrated critical coefficient")
    ci95 = quantile_interval(ordered, .95)
    ci975 = quantile_interval(ordered, .975)
    # With continuous R, F(R_(k)) ~ Beta(k,n+1-k), hence its upper tail
    # probability 1-F(R_(k)) ~ Beta(n+1-k,k) across calibration samples.
    alpha_interval = beta.ppf([.025, .975], n + 1 - k, k).tolist()
    return dict(
        target=target, isotope=isotope, c=c, ci95_c=ci95["interval"],
        ci975_c=ci975["interval"], n=n, k_order=k,
        achieved_training_rate=float(np.mean(values > c)),
        training_exceedances=int(np.sum(values > c)),
        ties=int(np.sum(values == c)),
        alpha_interval_from_calibration=alpha_interval,
        marginal_alpha_without_ties=(n + 1 - k) / (n + 1),
        ci95_rank_info=ci95, ci975_rank_info=ci975,
    )


def paired_bootstrap(ratios: np.ndarray, seed: list[int],
                     repetitions: int = BOOTSTRAP_REPETITIONS,
                     chunk_size: int = 4) -> np.ndarray:
    """Resample common rows for all eight ratios; partition each column."""
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 3 or ratios.shape[1:] != (2, 4):
        raise ValueError("Expected ratio array (n, two methods, four isotopes)")
    if not np.all(np.isfinite(ratios)) or chunk_size < 1:
        raise ValueError("Non-finite ratios or invalid chunk size")
    n = len(ratios)
    k = order_rank(n)
    rng = np.random.default_rng(seed)
    result = np.empty((repetitions, 2, 4))
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, n, size=(stop - start, n))
        sample = ratios[indices]
        # In-place partition caps working memory; a single row resample is
        # shared across methods and targets, preserving their dependence.
        sample.partition(k - 1, axis=1)
        result[start:stop] = sample[:, k - 1, :, :]
    return result


def load_calibration(root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Only the explicitly planned calibration files are read; no globbing."""
    manifest_path = root / "manifest.json"
    plan_path = root / "calibration_plan.json"
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    for name, digest in json.loads((root/'run_lock.json').read_text()).items():
        if sha256(root/name) != digest:
            raise ValueError(f'Locked file changed: {name}')
    if tuple(manifest["datasets"]) != DATASETS:
        raise ValueError("Unexpected dataset order")
    if manifest["master_seed"] != MASTER:
        raise ValueError("Unexpected Monte Carlo master seed for the updated-efficiency study")
    for dataset in DATASETS:
        metadata = manifest["datasets"][dataset]["metadata"]
        for key in ("sample_efficiencies", "gasbk_efficiencies"):
            np.testing.assert_array_equal(metadata[key], [.241169, .367784, .233937, .233937])
    if manifest["calibration_n_per_dataset"] != CALIBRATION_N:
        raise ValueError("The locked protocol requires 100000 calibration rows")
    if sum(job['n'] for job in plan) != CALIBRATION_N:
        raise ValueError("Calibration plan has the wrong total size")
    source_hashes = {"manifest.json": sha256(manifest_path),
                     "calibration_plan.json": sha256(plan_path),
                     "run_lock.json": sha256(root/'run_lock.json')}
    grouped = {dataset: [] for dataset in DATASETS}
    seen_keys, seen_seeds = set(), set()
    for job in plan:
        if job["stage"] != "calibration" or job["target"] != 0 or job["factor"] != 0:
            raise ValueError("Calibration plan contains a non-null/non-calibration batch")
        dataset = job["dataset"]
        key = (dataset, job["chunk"])
        if dataset not in grouped or key in seen_keys:
            raise ValueError("Duplicate or unexpected calibration batch")
        seen_keys.add(key)
        stem = f"{dataset}_calibration_i0_f00000_c{job['chunk']}"
        json_path = root / "points" / f"{stem}.json"
        npz_path = json_path.with_suffix(".npz")
        if not json_path.exists() or not npz_path.exists():
            raise RuntimeError(f"Calibration plan is incomplete: {stem}")
        metadata = json.loads(json_path.read_text())
        if metadata['manifest_sha256'] != source_hashes['manifest.json']:
            raise ValueError(f'Batch uses a different manifest: {stem}')
        if metadata['npz_sha256'] != sha256(npz_path):
            raise ValueError(f'Batch array checksum mismatch: {stem}')
        for field in ("dataset", "stage", "target", "factor", "chunk", "n"):
            if metadata[field] != job[field]:
                raise ValueError(f"Batch metadata disagrees with plan: {stem}, {field}")
        if metadata["errors"] or metadata["targets"] != list(TARGETS):
            raise ValueError(f"Missing estimates or unexpected targets in {stem}")
        np.testing.assert_array_equal(metadata["truth"], manifest["datasets"][dataset]["truth0"])
        np.testing.assert_allclose(metadata["mu"], manifest["datasets"][dataset]["mu0"], rtol=1e-13)
        seed = tuple(metadata["seed"])
        if seed in seen_seeds:
            raise ValueError("Duplicate seed in calibration")
        expected_seed = [manifest["master_seed"], DATASETS.index(dataset), 0, 0, 10, job["chunk"]]
        if list(seed) != expected_seed:
            raise ValueError("Calibration seed does not follow the locked plan")
        seen_seeds.add(seed)
        n = job["n"]
        with np.load(npz_path) as data:
            np.testing.assert_array_equal(data["indices"], np.arange(n))
            np.testing.assert_array_equal(data["targets"], TARGETS)
            if data["x"].shape != (n, 21):
                raise ValueError("Incomplete observation vectors")
            ratios = np.empty((n, 2, 4))
            for method_index, family in enumerate(FAMILIES):
                free = data[family + "_free"]
                positive = data[family + "_positive"]
                lc = data[family + "_LC"]
                decisions = data[family + "_decision"]
                if free.shape != (n, 17) or positive.shape != (n, 17) or lc.shape != (n, 4):
                    raise ValueError("Unexpected fit/threshold dimensions")
                if not np.all(np.isfinite(free)) or not np.all(np.isfinite(positive)):
                    raise ValueError("Failed or non-finite calibration fit")
                if not np.all(np.isfinite(lc)) or np.any(lc <= 0):
                    raise ValueError("Calibration requires finite strictly positive Lc")
                np.testing.assert_array_equal(decisions, (free[:, 1:5] > lc).astype(np.int8))
                ratios[:, method_index, :] = free[:, 1:5] / lc
                for column, stored_rate in enumerate(metadata["rates"][family]):
                    if stored_rate["n_valid"] != n or stored_rate["n_missing"] != 0:
                        raise ValueError("A calibration decision was dropped")
                    if stored_rate["k"] != int(decisions[:, column].sum()):
                        raise ValueError("Saved calibration detection count disagrees")
        grouped[dataset].append((job["chunk"], ratios))
        for path in (json_path, npz_path):
            source_hashes[str(path.relative_to(root))] = sha256(path)
    arrays = {}
    for dataset, batches in grouped.items():
        arrays[dataset] = np.concatenate([values for _, values in sorted(batches)], axis=0)
        if arrays[dataset].shape != (CALIBRATION_N, 2, 4):
            raise ValueError(f"Incomplete calibration for {dataset}: {arrays[dataset].shape}")
    return manifest, arrays, source_hashes


def freeze(root: Path = ROOT) -> None:
    destinations = [root / name for name in
                    ("calibration.json", "calibration_bootstrap.npz", "cutoff.json")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Calibration is immutable: a frozen output already exists")
    manifest, arrays, source_hashes = load_calibration(root)
    source_hashes["calibration.py"] = sha256(Path(__file__))
    result = dict(
        schema_version=1, datasets={}, target_alpha=.05,
        statistic="R=A_hat_free/Lc_standard", decision="R>c (strict)",
        calibration_rule="c=R_(ceil((n+1)*.95)); order ranks are one-based",
        family_order=list(FAMILIES), target_order=list(TARGETS),
        validation_not_used=True,
        validation_note="Only files named by calibration_plan.json were opened; validation outcomes were not read",
        alpha_interval_from_calibration_note=(
            "Central 95% Beta(n+1-k,k) law across repeated calibration samples under continuous R; "
            "not an independent measurement of frozen-c alpha. Strict ties are conservative. "
            "Use held-out validation intervals to report achieved size."),
        source_sha256=source_hashes,
        bootstrap=dict(repetitions=BOOTSTRAP_REPETITIONS, master_seed=BOOTSTRAP_MASTER,
                       seed_rule="[master_seed,dataset_index,0]",
                       shape=[BOOTSTRAP_REPETITIONS, 2, 4],
                       paired="Common null row resampling across both methods and all four targets"),
    )
    bootstrap_arrays = {}
    for dataset_index, dataset in enumerate(DATASETS):
        ratios = arrays[dataset]
        methods = {}
        for method_index, family in enumerate(FAMILIES):
            methods[family] = [coefficient(ratios[:, method_index, j], target, isotope)
                               for j, (target, isotope) in enumerate(zip(TARGETS, ISOTOPES))]
        result["datasets"][dataset] = dict(n=CALIBRATION_N, methods=methods,
                                           calibration_batches=len(ratios) // 1000)
        print(dataset, "coefficients", {f: [v["c"] for v in methods[f]] for f in FAMILIES}, flush=True)
        bootstrap_arrays[dataset] = paired_bootstrap(ratios, [BOOTSTRAP_MASTER, dataset_index, 0])
        print(dataset, "paired bootstrap complete", flush=True)
    if any(path.exists() for path in destinations):
        raise FileExistsError("Another calibration freeze created output during calculation")
    # Assemble complete temporary files first; hard links create each final file
    # atomically and fail if it already exists. The completion marker is last.
    temporary_paths = []
    try:
        fd, name = tempfile.mkstemp(prefix=".calibration-bootstrap-", suffix=".tmp", dir=root)
        temporary_paths.append(Path(name))
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **bootstrap_arrays)
        result["bootstrap"]["file"] = "calibration_bootstrap.npz"
        result["bootstrap"]["sha256"] = sha256(temporary_paths[0])
        result["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        fd, name = tempfile.mkstemp(prefix=".calibration-", suffix=".tmp", dir=root)
        temporary_paths.append(Path(name))
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        marker = dict(
            frozen=True, frozen_at_utc=result["frozen_at_utc"], validation_not_used=True,
            calibration_file="calibration.json", calibration_sha256=hashlib.sha256(payload).hexdigest(),
            bootstrap_file="calibration_bootstrap.npz", bootstrap_sha256=result["bootstrap"]["sha256"],
            coefficients={dataset: {family: [v["c"] for v in result["datasets"][dataset]["methods"][family]]
                                     for family in FAMILIES} for dataset in DATASETS},
        )
        fd, name = tempfile.mkstemp(prefix=".cutoff-", suffix=".tmp", dir=root)
        temporary_paths.append(Path(name))
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        for temporary, final in zip(temporary_paths,
                                    (destinations[1], destinations[0], destinations[2])):
            os.link(temporary, final)
        print("Frozen calibration.json, calibration_bootstrap.npz and cutoff.json; validation was not used", flush=True)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    arguments = parser.parse_args()
    freeze()
