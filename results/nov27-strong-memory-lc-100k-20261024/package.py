"""Create and verify the single portable XeMat strong-GASBK study archive.

This script only packages existing results. It never runs Monte Carlo or fits.
Checkpoint NPZ files are omitted because replicates.npz contains every row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = "XeMat_Nov27_strong_GASBK_LC_100k"
FILES = [
    "scenario.json", "scenario_arrays.npz", "scenario.lock.json", "runner_lock.json",
    "replicates.npz", "replicates_schema.md", "summary.csv", "summary.json",
    "diagnostics.json", "report.md", "false_detections.pdf", "false_detections.png",
    "prepare_scenario.py", "study.py", "analyze.py", "plot_false_detections.py",
    "verify_study.py", "verification.json", "figure_verification.json", "requirements.txt", "package.py",
    "prepare_scenario.log", "first_batch.log", "run.log",
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    verified = read(ROOT / "verification.json")
    if verified.get("status") != "passed":
        raise RuntimeError("The completed study has not passed the independent audit")
    for name, expected in verified["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("A numerically audited file changed: " + name)
    figure_check = read(ROOT / "figure_verification.json")
    if figure_check.get("status") != "passed":
        raise RuntimeError("Figure verification did not pass")
    for name, expected in figure_check["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("An audited figure file changed: " + name)
    scenario = read(ROOT / "scenario.json")
    for name, expected in read(ROOT / "scenario.lock.json").items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen scenario changed: " + name)
    lock = read(ROOT / "runner_lock.json")
    for field, name in (("study_sha256", "study.py"), ("scenario_sha256", "scenario.json"),
                        ("scenario_arrays_sha256", "scenario_arrays.npz")):
        if lock[field] != digest(ROOT / name):
            raise RuntimeError("Frozen runner changed: " + name)
    for name, expected in scenario["production_sha256"].items():
        if digest(ROOT / "kernel_snapshot" / name) != expected:
            raise RuntimeError("Kernel snapshot changed: " + name)
    names = list(FILES)
    names.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "inputs").glob("*.PHD")))
    names.extend("kernel_snapshot/" + name for name in scenario["production_sha256"])
    names.append("kernel_snapshot/snapshot_version.json")
    # Keep original batch diagnostics without duplicating the combined arrays.
    # A separate folder avoids incomplete checkpoints when reproducing the run.
    (ROOT / "batch_metadata").mkdir(exist_ok=True)
    for chunk in range(scenario["n_batches"]):
        name = f"batch_{chunk:03d}.json"
        source = ROOT / "points" / name
        destination = ROOT / "batch_metadata" / name
        if source.exists():
            shutil.copyfile(source, destination)
        if not destination.is_file():
            raise FileNotFoundError(destination)
        names.append("batch_metadata/" + name)
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate archive members")
    records = {}
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    write(ROOT / "checksums.json", dict(algorithm="SHA-256", files=records,
        note="Payload file hashes; this index and the ZIP checksum are kept outside their own hash scope."))
    names.append("checksums.json")
    temporary = ROOT / (PREFIX + ".tmp.zip")
    destination = ROOT / (PREFIX + ".zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, PREFIX + "/" + name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        expected_members = {PREFIX + "/" + name for name in names}
        if set(archive.namelist()) != expected_members:
            raise RuntimeError("ZIP inventory differs from the planned payload")
        for name in names:
            member = archive.read(PREFIX + "/" + name)
            if hashlib.sha256(member).hexdigest() != digest(ROOT / name):
                raise RuntimeError("ZIP member digest differs: " + name)
    if temporary.stat().st_size >= 95 * 1024**2:
        raise RuntimeError("Archive exceeds the intended direct-download size budget")
    temporary.replace(destination)
    result = dict(status="passed", archive=destination.name, bytes=destination.stat().st_size,
        sha256=digest(destination), member_count=len(names), root_folder=PREFIX,
        members=sorted(names), crc_checked=True, all_member_hashes_checked=True,
        checkpoint_npz_omitted="All 100000 rows are preserved in replicates.npz; no duplicate checkpoint arrays in ZIP",
        production_commit=scenario["production_commit"])
    write(ROOT / "bundle_verification.json", result)
    (ROOT / (destination.name + ".sha256")).write_text(
        result["sha256"] + "  " + destination.name + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "members"}, indent=2))


if __name__ == "__main__":
    main()
