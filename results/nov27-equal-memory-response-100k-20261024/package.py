"""Package the audited A/B comparison without running any numerical experiment.

The main ZIP contains all four concentrations, LC, decisions, complete auxiliary
vectors, statuses, observations and reproducible model/plotting code. Full native
17-parameter free vectors are a lossless supplementary NPZ delivered separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
PREFIX = "XeMat_Nov27_equal_memory_response_AB_100k"
FILES = [
    "scenario_A.json", "scenario_arrays_A.npz", "scenario_B.json", "scenario_arrays_B.npz",
    "scenario_B.lock.json", "runner_lock.json", "observations.npz", "preparation_verification.json",
    "replicates.npz", "replicates_schema.md", "summary.csv", "paired_comparison.csv",
    "summary.json", "diagnostics.json", "report.md", "false_detection_comparison.pdf",
    "false_detection_comparison.png", "figures.tex", "results_table.tex", "plot_comparison.py", "prepare_scenario.py",
    "study.py", "analyze.py", "verify_study.py", "verification.json", "figure_verification.json",
    "requirements.txt", "package.py", "prepare_scenario.log", "run.log",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    scenario = read(ROOT / "scenario_B.json")
    summary = read(ROOT / "summary.json")
    audit = read(ROOT / "verification.json")
    if audit.get("status") != "passed":
        raise RuntimeError("Completed numerical audit required")
    if not audit.get("full_free_parameters_checked") or not audit.get("original_A_arrays_compared"):
        raise RuntimeError("Complete original-A and full-free-vector checks required for publication")
    for name, expected in audit["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Numerically audited file changed: " + name)
    for name, expected in read(ROOT / "scenario_B.lock.json").items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Frozen scenario changed: " + name)
    for name, expected in scenario["production_sha256"].items():
        if sha(ROOT / "kernel_snapshot" / name) != expected:
            raise RuntimeError("Unchanged kernel snapshot differs: " + name)
    figure_check = read(ROOT / "figure_verification.json")
    if figure_check.get("status") != "passed":
        raise RuntimeError("Figure verification required")
    for name, expected in figure_check["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Audited figure changed: " + name)
    for name, field in (("replicates.npz", "replicates_sha256"),
                        ("full_free_parameters.npz", "full_free_parameters_sha256"),
                        ("summary.csv", "summary_csv_sha256"),
                        ("paired_comparison.csv", "paired_csv_sha256")):
        if sha(ROOT / name) != summary[field]:
            raise RuntimeError("Reported file digest differs: " + name)
    supplementary = ROOT / "full_free_parameters.npz"
    if supplementary.stat().st_size >= 95 * 1024**2:
        raise RuntimeError("Supplement exceeds direct-download file budget")
    write(ROOT / "supplement.json", dict(file=supplementary.name,
        bytes=supplementary.stat().st_size, sha256=sha(supplementary),
        included_in_main_zip=False,
        description="All four A/B × LS/Poisson native full free-parameter arrays, 100000x17 each; no rounding or dropped rows. Four Xe concentrations and all auxiliary vectors are already in the main ZIP.",
        use="Place full_free_parameters.npz beside replicates.npz to enable optional complete native-vector verification."))
    names = list(FILES) + ["supplement.json"]
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "inputs").glob("*.PHD")))
    names.extend("kernel_snapshot/" + name for name in scenario["production_sha256"])
    names.append("kernel_snapshot/snapshot_version.json")
    metadata_folder = ROOT / "batch_metadata_B"
    metadata_folder.mkdir(exist_ok=True)
    for chunk in range(100):
        name = f"batch_{chunk:03d}.json"
        source = ROOT / "points" / name
        destination = metadata_folder / name
        if source.exists():
            shutil.copyfile(source, destination)
        if not destination.is_file():
            raise FileNotFoundError(destination)
        names.append("batch_metadata_B/" + name)
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate archive paths")
    records = {name:dict(bytes=(ROOT/name).stat().st_size, sha256=sha(ROOT/name))
               for name in sorted(names)}
    write(ROOT / "checksums.json", dict(algorithm="SHA-256", files=records,
        supplementary={supplementary.name:read(ROOT / "supplement.json")}))
    names.append("checksums.json")
    destination = ROOT / (PREFIX + ".zip")
    temporary = ROOT / (PREFIX + ".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name in sorted(names):
            archive.write(ROOT / name, PREFIX + "/" + name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
        if set(archive.namelist()) != {PREFIX + "/" + name for name in names}:
            raise RuntimeError("Archive inventory differs")
        for name in names:
            if hashlib.sha256(archive.read(PREFIX + "/" + name)).hexdigest() != sha(ROOT / name):
                raise RuntimeError("Archive member digest differs: " + name)
    if temporary.stat().st_size >= 95 * 1024**2:
        raise RuntimeError("Main ZIP exceeds direct-download size budget")
    temporary.replace(destination)
    result = dict(status="passed", archive=destination.name, root_folder=PREFIX,
        bytes=destination.stat().st_size, sha256=sha(destination), member_count=len(names),
        members=sorted(names), crc_checked=True, all_member_hashes_checked=True,
        supplementary=read(ROOT / "supplement.json"), production_commit=scenario["production_commit"])
    write(ROOT / "bundle_verification.json", result)
    for path in (destination, supplementary):
        (ROOT / (path.name + ".sha256")).write_text(sha(path) + "  " + path.name + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "members"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
