"""Create a self-contained, hash-checked archive after the independent audit.

All new raw batches remain byte-for-byte unchanged. Former signal batches are
represented by their complete 21 counts and target estimates/thresholds, with
the original metadata and file hashes retained in a portable block index.
The production app and the previous study are read only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import analyze

ROOT = Path(__file__).resolve().parent
OLD = analyze.OLD
FAMILIES = ("wls", "poisson")
ARCHIVE = "XeMat_calibrated_nominal_MDC_May.zip"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def require_finished_audit():
    summary = json.loads((ROOT / "summary.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    audit = json.loads((ROOT / "verification.json").read_text())
    if audit.get("status") != "passed":
        raise RuntimeError("Run the independent verification successfully before packaging")
    if summary.get("needs_extension"):
        raise RuntimeError("The signal grid still has unresolved extension requests")
    expected = sum(summary["n_new_by_stage"].values())
    if audit.get("new_vectors") != expected:
        raise RuntimeError("Verification does not cover all newly reported observations")
    if audit.get("reused_signal_vectors") != summary["n_reused_signal"]:
        raise RuntimeError("Verification and summary disagree on reused signals")
    if audit.get("calibration_sha256") != sha256(ROOT / "calibration.json"):
        raise RuntimeError("The calibration file differs from the independently verified one")
    calibration = json.loads((ROOT / "calibration.json").read_text())
    cutoff = json.loads((ROOT / "cutoff.json").read_text())
    if not cutoff.get("frozen") or not cutoff.get("validation_not_used"):
        raise RuntimeError("Missing valid calibration freeze marker")
    if cutoff["calibration_sha256"] != sha256(ROOT / "calibration.json"):
        raise RuntimeError("Frozen calibration hash mismatch")
    if cutoff["bootstrap_sha256"] != sha256(ROOT / "calibration_bootstrap.npz"):
        raise RuntimeError("Frozen bootstrap hash mismatch")
    # Includes the original raw calibration batches and calibration.py itself.
    for name, expected_digest in calibration["source_sha256"].items():
        if sha256(ROOT / name) != expected_digest:
            raise RuntimeError(f"A frozen calibration source changed: {name}")
    return summary, manifest


def copy_prior_documents(manifest):
    """Copy exact source bytes, or retain already archived verified copies."""
    for source_name in ("manifest.json", "summary.json"):
        destination = ROOT / f"prior_{source_name}"
        original = OLD / source_name
        if original.exists():
            payload = original.read_bytes()
        elif destination.exists():
            payload = destination.read_bytes()
        else:
            raise FileNotFoundError(f"Need {original} or its archived copy {destination.name}")
        expected = manifest[f"prior_{source_name.removesuffix('.json')}_sha256"]
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError(f"Previous-study document hash mismatch: {source_name}")
        if destination.exists() and destination.read_bytes() != payload:
            raise RuntimeError(f"Refusing to replace a different {destination.name}")
        if not destination.exists():
            destination.write_bytes(payload)
    prior = json.loads((ROOT / "prior_manifest.json").read_text())
    if prior["datasets"] != manifest["datasets"]:
        raise RuntimeError("The previous and current generating models differ")


def validate_portable_signal(summary):
    index = json.loads((ROOT / "portable_signal_index.json").read_text())
    expected = summary["n_reused_signal"] + summary["n_new_by_stage"].get("signal_extension", 0)
    if index["n_old"] != summary["n_reused_signal"] or index["n_new"] != expected - index["n_old"]:
        raise RuntimeError("Portable signal counts disagree with the final summary")
    with np.load(ROOT / "portable_signal.npz") as archive:
        if set(archive.files) != {"ratios", "free", "LC", "positive", "x"}:
            raise RuntimeError("Unexpected portable signal arrays")
        arrays = {name: archive[name] for name in archive.files}
    for name in ("ratios", "free", "LC", "positive"):
        if arrays[name].shape != (expected, 2):
            raise RuntimeError(f"Wrong portable shape for {name}")
    if arrays["x"].shape != (expected, 21) or not np.issubdtype(arrays["x"].dtype, np.integer):
        raise RuntimeError("Portable observations must be full integer 21-vectors")
    if np.any(arrays["x"] < 0):
        raise RuntimeError("Negative portable counts")
    valid = np.isfinite(arrays["free"]) & np.isfinite(arrays["LC"]) & (arrays["LC"] > 0)
    ratios = np.divide(arrays["free"], arrays["LC"], out=np.full((expected, 2), np.nan), where=valid)
    np.testing.assert_array_equal(ratios, arrays["ratios"])
    cursor = 0
    for group in index["groups"]:
        if group["start"] != cursor or group["stop"] - group["start"] != group["n"]:
            raise RuntimeError("Portable groups do not partition the observations")
        if group["n_old"] + group["n_new"] != group["n"]:
            raise RuntimeError("Portable group provenance counts disagree")
        cursor = group["stop"]
    if cursor != expected:
        raise RuntimeError("Portable group index omits observations")
    cursor = 0
    blocks_old = blocks_new = 0
    for block in index["blocks"]:
        meta = block["original_meta"]
        if block["start"] != cursor or block["stop"] - block["start"] != meta["n"]:
            raise RuntimeError("Portable blocks do not partition the observations")
        if meta["targets"] != [meta["target"]]:
            raise RuntimeError("Expected one target in each formal signal block")
        # This also checks that every stored count was copied to the right row.
        generated = np.random.default_rng(meta["seed"]).poisson(meta["mu"], size=(meta["n"], 21))
        np.testing.assert_array_equal(generated, arrays["x"][block["start"]:block["stop"]])
        if block["is_old"]:
            blocks_old += meta["n"]
        else:
            blocks_new += meta["n"]
        cursor = block["stop"]
    if cursor != expected or (blocks_old, blocks_new) != (index["n_old"], index["n_new"]):
        raise RuntimeError("Portable blocks have incomplete provenance counts")
    return index


def build_portable_signal(summary):
    """Use precisely the group and source row order used by analyze.py."""
    if not (OLD / "points").exists():
        # Allows packaging a previously extracted, self-contained archive.
        return validate_portable_signal(summary)
    groups, errors, n_old, n_new = analyze.signal_groups()
    if n_old != summary["n_reused_signal"] or n_new != summary["n_new_by_stage"].get("signal_extension", 0):
        raise RuntimeError("Signal inputs do not match the final summary")
    all_arrays = {key: [] for key in ("ratios", "free", "LC", "positive", "x")}
    group_index = []
    blocks = []
    offset = 0
    for group in groups:
        start = offset
        group_ratios = []
        count_old = count_new = 0
        for source in group["sources"]:
            path = Path(source["path"])
            npz_path = path.with_suffix(".npz")
            if sha256(path) != source["sha256"] or sha256(npz_path) != source["npz_sha256"]:
                raise RuntimeError(f"Source changed while building the portable data: {path.name}")
            meta = json.loads(path.read_text())  # Insertion order is preserved.
            if (meta["dataset"], meta["target"], meta["factor"]) != (group["dataset"], group["target"], group["factor"]):
                raise RuntimeError("A signal source was assigned to the wrong group")
            if meta["targets"] != [meta["target"]]:
                raise RuntimeError("Expected one target per formal signal source")
            target = meta["target"]
            lc_column = meta["targets"].index(target)
            n = meta["n"]
            is_old = path.parent.resolve() == (OLD / "points").resolve()
            if is_old:
                if meta["stage"] not in ("confirm", "extend"):
                    raise RuntimeError("Only former formal signal batches may be reused")
                count_old += n
            else:
                if path.parent.resolve() != (ROOT / "points").resolve() or meta["stage"] != "extend":
                    raise RuntimeError("Unknown new signal source")
                count_new += n
            with np.load(npz_path) as data:
                values = {
                    "free": np.stack([data[f + "_free"][:, target] for f in FAMILIES], axis=1),
                    "LC": np.stack([data[f + "_LC"][:, lc_column] for f in FAMILIES], axis=1),
                    "positive": np.stack([data[f + "_positive"][:, target] for f in FAMILIES], axis=1),
                    "x": data["x"].copy(),
                }
                if values["x"].shape != (n, 21):
                    raise RuntimeError("Incomplete original signal observation vectors")
                valid = np.isfinite(values["free"]) & np.isfinite(values["LC"]) & (values["LC"] > 0)
                values["ratios"] = np.divide(values["free"], values["LC"],
                                              out=np.full((n, 2), np.nan), where=valid)
                for column, family in enumerate(FAMILIES):
                    decisions = data[family + "_decision"][:, lc_column]
                    available = decisions >= 0
                    np.testing.assert_array_equal(values["ratios"][available, column] > 1, decisions[available])
            for name, value in values.items():
                all_arrays[name].append(value)
            group_ratios.append(values["ratios"])
            blocks.append(dict(start=offset, stop=offset + n, original_meta=meta,
                               original_json_sha256=source["sha256"], original_npz_sha256=source["npz_sha256"],
                               is_old=is_old))
            offset += n
        np.testing.assert_array_equal(np.concatenate(group_ratios), group["ratios"])
        if (count_old, count_new) != (group["n_old"], group["n_new"]) or offset - start != group["n"]:
            raise RuntimeError("Portable group row counts differ from the analysis")
        group_index.append({**{key: group[key] for key in ("dataset", "target", "factor", "n", "n_old", "n_new", "sources")},
                            "start": start, "stop": offset})
    arrays = {name: np.concatenate(values) for name, values in all_arrays.items()}
    np.savez_compressed(ROOT / "portable_signal.npz", **arrays)
    index = dict(schema_version=1, family_order=list(FAMILIES),
                 units={"ratios": "dimensionless", "free": "Bq/m3", "LC": "Bq/m3", "positive": "Bq/m3", "x": "integer counts"},
                 observation_order="SAMPLE ROI [1,2,3,5,6,7,8], GASBK same ROI, DETBK same ROI",
                 note="Target-only estimates; all 21 original counts and metadata are retained. Original full NPZ hashes are provenance; full new batches remain in points.",
                 groups=group_index, blocks=blocks, errors=errors, n_old=n_old, n_new=n_new)
    write_json(ROOT / "portable_signal_index.json", index)
    return validate_portable_signal(summary)


def checked_raw_point_files(summary):
    json_paths = sorted((ROOT / "points").glob("*.json"))
    npz_paths = sorted((ROOT / "points").glob("*.npz"))
    if {p.stem for p in json_paths} != {p.stem for p in npz_paths}:
        raise RuntimeError("Unpaired new raw metadata/array files")
    counts = Counter()
    names = []
    for path in json_paths:
        meta = json.loads(path.read_text())
        if meta["stage"] not in ("calibration", "validation", "extend"):
            raise RuntimeError(f"Unexpected batch in publishable points: {path.name}")
        counts[meta["stage"]] += meta["n"]
        names.extend((path.relative_to(ROOT).as_posix(), path.with_suffix(".npz").relative_to(ROOT).as_posix()))
    expected = summary["n_new_by_stage"]
    for stage, key in (("calibration", "calibration"), ("validation", "validation"), ("extend", "signal_extension")):
        if counts[stage] != expected.get(key, 0):
            raise RuntimeError(f"New raw points do not match final {key} count")
    return names


def payload_names(summary):
    required = [
        "README.md", "protocol_ru.md", "design_review.md", "manifest.json", "prior_manifest.json", "prior_summary.json",
        "calibration_plan.json", "validation_plan.json", "technical_plan.json", "technical_review.json",
        "calibration.json", "calibration_bootstrap.npz", "cutoff.json", "summary.json", "verification.json",
        "report_ru.md", "report_ru.txt", "results.csv", "portable_signal.npz", "portable_signal_index.json",
        "study.py", "calibration.py", "analyze.py", "report.py", "verify.py", "package.py",
    ]
    required.extend(f"{stem}.{suffix}" for stem in ("calibrated_alpha", "calibrated_ld", "calibrated_power") for suffix in ("pdf", "png"))
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Required final artifact missing: {name}")
    # Root-level research artifacts only. Logs, technical trials and caches are
    # intentionally excluded; raw publishable batches are added explicitly.
    suffixes = {".py", ".md", ".json", ".txt", ".csv", ".pdf", ".png", ".npz"}
    names = {path.name for path in ROOT.iterdir()
             if path.is_file() and path.suffix in suffixes and not path.name.startswith(".") and path.name != "sha256.json"}
    names.update(required)
    names.update(checked_raw_point_files(summary))
    return sorted(names)


def main():
    summary, manifest = require_finished_audit()
    copy_prior_documents(manifest)
    index = build_portable_signal(summary)
    names = payload_names(summary)
    hashes = {name: sha256(ROOT / name) for name in names}
    write_json(ROOT / "sha256.json", hashes)
    names.append("sha256.json")
    names.sort()
    fd, temporary_name = tempfile.mkstemp(prefix=".xemat-calibrated-archive-", suffix=".tmp", dir=ROOT)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                archive.write(ROOT / name, name)
        with ZipFile(temporary) as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise RuntimeError("Archive member list or CRC integrity check failed")
            for name, expected in hashes.items():
                with archive.open(name) as stream:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"Archive payload hash mismatch: {name}")
            if json.loads(archive.read("sha256.json")) != hashes:
                raise RuntimeError("Archive checksum manifest differs from payload hashes")
        destination = ROOT / ARCHIVE
        os.replace(temporary, destination)
        print("Verified archive", destination.name, "bytes", destination.stat().st_size,
              "files", len(names), "new raw vectors", sum(summary["n_new_by_stage"].values()),
              "portable signal vectors", index["n_old"] + index["n_new"], flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
