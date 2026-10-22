"""Package the audited efficiency-update report and all new raw trials.

Every raw batch is retained unchanged, including all 21 observations and all
17 fitted parameters in the full archive. A separate small archive contains
the report, figures, tables and protocols. Both are checked member by member.
Only repeating model fits requires the recorded XeMat code version.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
ARCHIVE = "XeMat_nominal_MDC_eff221_20261022.zip"
REPORT_ARCHIVE = "XeMat_MDC_report_eff221_20261022.zip"
FIGURES = ("calibrated_alpha", "calibrated_ld", "calibrated_power", "efficiency_comparison")
HASH_FILES = {"sha256.json", "sha256_report.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text())


def require_finished_audit():
    summary = read_json("summary.json")
    manifest = read_json("manifest.json")
    audit = read_json("verification.json")
    if audit.get("status") != "passed":
        raise RuntimeError("Successful independent verification is required before packaging")
    if summary.get("needs_extension"):
        raise RuntimeError("The formal signal grid still requires extension")
    if summary.get("n_reused_signal", 0) != 0:
        raise RuntimeError("This efficiency-update study must not reuse old fitted signal results")
    expected = sum(summary["n_new_by_stage"].values())
    if audit.get("new_vectors") != expected:
        raise RuntimeError("Independent audit does not cover all reported raw vectors")
    if audit.get("summary_sha256") is not None and audit["summary_sha256"] != sha256(ROOT / "summary.json"):
        raise RuntimeError("The reported summary differs from the independently audited one")
    if audit.get("calibration_sha256") != sha256(ROOT / "calibration.json"):
        raise RuntimeError("Independent audit used a different calibration")
    cutoff = read_json("cutoff.json")
    if not cutoff.get("frozen") or not cutoff.get("validation_not_used"):
        raise RuntimeError("The calibration was not frozen before independent validation")
    for name, key in (("calibration.json", "calibration_sha256"),
                      ("calibration_bootstrap.npz", "bootstrap_sha256")):
        if cutoff[key] != sha256(ROOT / name):
            raise RuntimeError(f"Frozen-file hash mismatch: {name}")
    calibration = read_json("calibration.json")
    for name, digest in calibration["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Frozen calibration source changed: {name}")
    expected_prior = {
        "prior_formula_manifest.json", "prior_formula_summary.json",
        "prior_calibrated_manifest.json", "prior_calibrated_summary.json",
    }
    if set(manifest["prior_document_sha256"]) != expected_prior:
        raise RuntimeError("Missing provenance for the previous-study documents")
    for name, digest in manifest["prior_document_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Previous-study document changed: {name}")
    return summary


def checked_raw_points(summary):
    json_paths = sorted((ROOT / "points").glob("*.json"))
    npz_paths = sorted((ROOT / "points").glob("*.npz"))
    if {p.stem for p in json_paths} != {p.stem for p in npz_paths}:
        raise RuntimeError("Unpaired raw metadata and arrays")
    counts = Counter()
    names = []
    aliases = {"calibration": "calibration", "validation": "validation", "signal": "signal",
               "confirm": "signal", "extend": "signal_extension", "extension": "signal_extension"}
    for path in json_paths:
        meta = json.loads(path.read_text())
        if meta["stage"] not in aliases:
            raise RuntimeError(f"Unexpected raw stage in final points: {meta['stage']}")
        counts[aliases[meta["stage"]]] += meta["n"]
        names.extend((path.relative_to(ROOT).as_posix(), path.with_suffix(".npz").relative_to(ROOT).as_posix()))
    expected = summary["n_new_by_stage"]
    for key in ("calibration", "validation", "signal", "signal_extension"):
        if counts[key] != expected.get(key, 0):
            raise RuntimeError(f"Raw and summary count mismatch for {key}: {counts[key]} vs {expected.get(key, 0)}")
    if sum(counts.values()) != sum(expected.values()):
        raise RuntimeError("The summary includes unsupported stages")
    return names


def payload_names(summary):
    required = [
        "README.md", "protocol_ru.md", "statistical_design.md", "manifest.json", "inputs_review.json",
        "prior_formula_manifest.json", "prior_formula_summary.json", "prior_calibrated_manifest.json",
        "prior_calibrated_summary.json", "calibration_plan.json", "validation_plan.json", "signal_plan.json",
        "calibration.json", "calibration_bootstrap.npz", "cutoff.json", "summary.json", "verification.json",
        "report_ru.md", "report_ru.txt", "results.csv", "efficiency_comparison.csv", "nominal_formula.csv",
        "prepare.py", "study.py", "calibration.py", "analyze.py", "report.py", "verify.py", "package.py",
    ]
    required.extend(f"{stem}.{suffix}" for stem in FIGURES
                    for suffix in ("pdf", "png"))
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Final artifact missing: {name}")
    suffixes = {".py", ".md", ".json", ".txt", ".csv", ".pdf", ".png", ".npz"}
    names = {p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in suffixes
             and p.name not in HASH_FILES and not p.name.startswith(".")}
    names.update(required)
    names.update(checked_raw_points(summary))
    # Input snapshots, if prepare.py provides them, are research provenance.
    for folder in ("inputs", "sources"):
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob("*") if p.is_file())
    return sorted(names)


def report_payload_names():
    """Explicit allow-list: no raw vectors, scripts, archives or input snapshots."""
    names = ["report_ru.md", "report_ru.txt", "protocol_ru.md", "statistical_design.md",
             "results.csv", "efficiency_comparison.csv", "nominal_formula.csv"]
    names.extend(f"{stem}.{suffix}" for stem in FIGURES for suffix in ("pdf", "png"))
    for name in names:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Final report artifact missing: {name}")
    return sorted(names)


def build_archive(archive_name, names, checksum_name, raw_vectors):
    """Build atomically, verify CRC and SHA256, and print downloadable-file details."""
    if len(names) != len(set(names)) or checksum_name in names:
        raise RuntimeError("Duplicate archive member or nested checksum manifest")
    if any(Path(name).suffix.lower() == ".zip" for name in names):
        raise RuntimeError("Archives must not contain other archives")
    hashes = {name: sha256(ROOT / name) for name in names}
    (ROOT / checksum_name).write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    names = sorted(names + [checksum_name])
    fd, path = tempfile.mkstemp(prefix=".xemat-new-efficiency-", suffix=".tmp", dir=ROOT)
    os.close(fd)
    temporary = Path(path)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                archive.write(ROOT / name, name)
        with ZipFile(temporary) as archive:
            if archive.namelist() != names or archive.testzip() is not None:
                raise RuntimeError("Archive member-list or CRC verification failed")
            for name, expected in hashes.items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"Archive member hash mismatch: {name}")
            if json.loads(archive.read(checksum_name)) != hashes:
                raise RuntimeError("Archived checksum manifest mismatch")
        destination = ROOT / archive_name
        os.replace(temporary, destination)
        size = destination.stat().st_size
        print(json.dumps(dict(archive=str(destination), bytes=size, size_MiB=round(size / 2**20, 3),
            above_95_MiB=size > 95 * 2**20,
            files=len(names), raw_vectors=raw_vectors,
            sha256=sha256(destination)), indent=2), flush=True)
        if size > 95 * 2**20:
            print("NOTICE: archive exceeds 95 MiB; review size before GitHub publication.", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    summary = require_finished_audit()
    # Resolve both payloads before writing either ZIP. The full payload is also
    # checked against audited raw trial counts, even when only reading the report.
    names = payload_names(summary)
    report_names = report_payload_names()
    build_archive(REPORT_ARCHIVE, report_names, "sha256_report.json", raw_vectors=0)
    build_archive(ARCHIVE, names, "sha256.json",
                  raw_vectors=sum(summary["n_new_by_stage"].values()))


if __name__ == "__main__":
    main()
