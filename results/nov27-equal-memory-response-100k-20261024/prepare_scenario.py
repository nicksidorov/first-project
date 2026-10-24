"""Freeze model B while retaining all previously observed Monte Carlo rows.

The four xenon columns of the BASE memory response are replaced by the
corresponding fresh-response columns. Decay/daughter corrections are then
applied separately for SAMPLE and GASBK using their existing matrices.
No solver, detection-limit routine, or random-number generator is called.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("XEMAT_BASELINE_A", "/workspace/scratch/nov27-strong-memory-lc-100k-20261024"))
REPO = ROOT / "kernel_snapshot" if (ROOT / "kernel_snapshot").is_dir() else SOURCE / "kernel_snapshot"
sys.path.insert(0, str(REPO))

import numpy as np
from xemat.io import jsonable
from xemat.model import JointModel, build_joint_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("scenario_B.json", "scenario_arrays_B.npz", "observations.npz"):
        if (ROOT / name).exists():
            raise FileExistsError("Frozen preparation already exists: " + name)
    scenario_A = read(SOURCE / "scenario.json")
    source_summary = read(SOURCE / "summary.json")
    source_replicates = SOURCE / "replicates.npz"
    source_sha = sha(source_replicates)
    assert source_sha == source_summary["replicates_sha256"]
    assert scenario_A["production_commit"] == "69ff5a62ccaaddeb6372f7f003109ce5ff6cf333"
    assert scenario_A["n"] == 100000
    for name, digest in read(SOURCE / "scenario.lock.json").items():
        assert sha(SOURCE / name) == digest, name
    for name, digest in scenario_A["production_sha256"].items():
        assert sha(SOURCE / "kernel_snapshot" / name) == digest, name
        destination = ROOT / "kernel_snapshot" / name
        destination.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(SOURCE / "kernel_snapshot" / name, destination)
        assert sha(destination) == digest
    shutil.copyfile(SOURCE / "kernel_snapshot/snapshot_version.json",
                    ROOT / "kernel_snapshot/snapshot_version.json")
    (ROOT / "inputs").mkdir(exist_ok=True)
    for record in scenario_A["input_files"].values():
        source = SOURCE / record["path"]
        destination = ROOT / "inputs" / source.name
        assert sha(source) == record["sha256"]
        shutil.copyfile(source, destination)
    shutil.copyfile(SOURCE / "scenario.json", ROOT / "scenario_A.json")
    shutil.copyfile(SOURCE / "scenario_arrays.npz", ROOT / "scenario_arrays_A.npz")

    A = JointModel(**scenario_A["model"])
    old = scenario_A["calibration_matrices"]
    matrices = {name: np.asarray(old[name], dtype=float) for name in
                ("R_c0", "R_g0", "R_c", "R_g1", "R_g2", "C_s_c", "C_s_g", "C_g_g", "F")}
    Rg0_B = matrices["R_g0"].copy()
    Rg0_B[:, 1:5] = matrices["R_c0"][:, 1:5]
    np.testing.assert_array_equal(Rg0_B[:, 0], matrices["R_g0"][:, 0])
    Rg1_B = Rg0_B @ matrices["C_s_g"]
    Rg2_B = Rg0_B @ matrices["C_g_g"]
    M_B = build_joint_matrix(matrices["R_c"], Rg1_B, Rg2_B, matrices["F"],
                            kappa_s=old["kappa_s"], kappa_g=old["kappa_g"])
    B = JointModel(M_B, A.x, A.b, A.n_sample_parameters,
                   A.T_c, A.coordinates, A.n_detector_parameters)
    S_A, S_B = A.physical_matrix, B.physical_matrix
    assert S_A.shape == S_B.shape == (21, 17)
    assert np.linalg.matrix_rank(S_B / np.linalg.norm(S_B, axis=0)) == 17
    # These are exact identities for the unchanged fresh, radon-memory,
    # detector-background blocks and coordinate transformation.
    unchanged_columns = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]
    np.testing.assert_array_equal(M_B[:, unchanged_columns], A.M[:, unchanged_columns])
    np.testing.assert_array_equal(B.T_c, A.T_c)
    np.testing.assert_array_equal(B.x, A.x)
    np.testing.assert_array_equal(B.b, A.b)
    np.testing.assert_array_equal(M_B[14:], A.M[14:])

    # Explicit Xe133m chain check: its BASE mother column is already equal
    # in this calibration, but the changed Xe133 base changes the daughter
    # contribution in both exposure-corrected Xe133m columns.
    assert np.array_equal(matrices["R_g0"][:, 4], matrices["R_c0"][:, 4])
    daughter_checks = {}
    for name, C, corrected in (("sample_memory", matrices["C_s_g"], Rg1_B),
                                ("gasbk", matrices["C_g_g"], Rg2_B)):
        coefficient = float(C[2, 4])
        explicit = Rg0_B[:, 4] + coefficient * Rg0_B[:, 2]
        np.testing.assert_allclose(corrected[:, 4], explicit, rtol=1e-15, atol=1e-15)
        old_corrected = matrices["R_g1" if name == "sample_memory" else "R_g2"]
        expected_change = coefficient * (Rg0_B[:, 2] - matrices["R_g0"][:, 2])
        np.testing.assert_allclose(corrected[:, 4] - old_corrected[:, 4], expected_change,
                                   rtol=1e-12, atol=1e-15)
        daughter_checks[name] = dict(coefficient=coefficient,
            formula="r_g,133m(T)=r_g,133m^(0)+C[2,4]*r_g,133^(0)",
            corrected_Xe133m_A=old_corrected[:, 4], corrected_Xe133m_B=corrected[:, 4],
            expected_change_due_to_daughter=expected_change)

    with np.load(SOURCE / "scenario_arrays.npz", allow_pickle=False) as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    truth = arrays["truth_physical"].copy()
    mu0_A = arrays["mu0"].copy()
    np.testing.assert_allclose(mu0_A, S_A @ truth + A.b, rtol=1e-14, atol=1e-12)
    mu_B_at_A_truth = S_B @ truth + B.b
    assert np.any(np.abs(mu_B_at_A_truth - mu0_A) > 1e-10)
    arrays.update(M=M_B, M_A=A.M, M_B=M_B, physical_matrix=S_B,
        physical_matrix_A=S_A, physical_matrix_B=S_B, mu0=mu0_A, mu0_A=mu0_A,
        mu_model_B_at_A_truth=mu_B_at_A_truth,
        mean_model_mismatch_B_minus_generator_A=mu_B_at_A_truth-mu0_A,
        R_c0=matrices["R_c0"], R_g0_A=matrices["R_g0"], R_g0_B=Rg0_B,
        R_c=matrices["R_c"], R_g1_A=matrices["R_g1"], R_g1_B=Rg1_B,
        R_g2_A=matrices["R_g2"], R_g2_B=Rg2_B,
        C_s_g=matrices["C_s_g"], C_g_g=matrices["C_g_g"], F=matrices["F"],
        calibration_M=M_B, calibration_R_g0=Rg0_B,
        calibration_R_g1=Rg1_B, calibration_R_g2=Rg2_B,
        interpretation_mu0=np.array("Original A generating mean; NOT model B mean at A truth"),
        generating_model_variant=np.array("A"), processing_model_variant=np.array("B"))
    np.savez_compressed(ROOT / "scenario_arrays_B.npz", **arrays)
    with np.load(source_replicates, allow_pickle=False) as raw:
        names = ("x", "replicate_id", "chunk", "row_in_batch", "seed_table",
                 "isotope_order", "observation_order", "observation_units")
        observations = {name: raw[name].copy() for name in names}
    assert observations["x"].shape == (100000, 21)
    np.testing.assert_array_equal(observations["replicate_id"], np.arange(100000))
    observations.update(source_replicates_sha256=np.array(source_sha),
                        counts_reused_without_generation=np.array(True))
    np.savez_compressed(ROOT / "observations.npz", **observations)

    changed_base_entries = []
    for row, col in np.argwhere(Rg0_B != matrices["R_g0"]):
        changed_base_entries.append(dict(row_index=int(row), roi=int(scenario_A["roi_order"][row]),
            column_index=int(col), isotope=old["isotope_names"][col],
            A=float(matrices["R_g0"][row, col]), B=float(Rg0_B[row, col])))
    scenario_B = copy.deepcopy(scenario_A)
    calibration_B = scenario_B["calibration_matrices"]
    for name, value in (("M", M_B), ("R_g0", Rg0_B), ("R_g1", Rg1_B), ("R_g2", Rg2_B)):
        calibration_B[name] = value
    scenario_B.update(schema_version=2,
        scenario_id="nov27_2025_strong_memory_detector221_equal_base_memory_response_B",
        variant="B", variant_label="Equal xenon base responses in the processing model",
        created_utc=datetime.now(timezone.utc).isoformat(), model=jsonable(B),
        generation_model_variant="A", no_new_observation_generation=True,
        observations_file="observations.npz", observations_sha256=sha(ROOT / "observations.npz"),
        source_replicates_sha256=source_sha,
        scenario_arrays_sha256=sha(ROOT / "scenario_arrays_B.npz"),
        preparation_script_sha256=sha(Path(__file__)),
        source_baseline_A=dict(directory=str(SOURCE),
            published_commit="72b360874b1596a8ab85520268358612ec23206e",
            replicates_file="replicates.npz", replicates_sha256=source_sha,
            scenario_sha256=sha(SOURCE / "scenario.json"),
            scenario_arrays_sha256=sha(SOURCE / "scenario_arrays.npz"),
            summary_sha256=sha(SOURCE / "summary.json"),
            direct_source_url="https://raw.githubusercontent.com/nicksidorov/first-project/72b360874b1596a8ab85520268358612ec23206e/results/nov27-strong-memory-lc-100k-20261024/replicates.npz"),
        modification=dict(base_memory_matrix="R_g0", replaced_columns_zero_based=[1,2,3,4],
            rule="R_g0_B[:,1:5]=R_c0[:,1:5]; R_g0_B[:,0]=R_g0_A[:,0]",
            corrected_memory_sample="R_g1_B=R_g0_B@C_s_g",
            corrected_memory_gasbk="R_g2_B=R_g0_B@C_g_g",
            sample_memory_block="R_g1_B@F", no_column_renormalization=True,
            unchanged=["R_c0", "R_c", "C_s_c", "C_s_g", "C_g_g", "F", "H", "H_g1", "H_g2",
                "Gamma", "G_p", "G_s", "T_c", "T_V", "b", "kappa_s", "kappa_g", "times", "volumes",
                "efficiencies", "branching", "detector_background_model", "solver_settings", "WLS_weights"],
            changed_base_entries=changed_base_entries,
            Xe133m_base_column_already_identical=True,
            Xe133m_daughter_checks=daughter_checks),
        generating_truth_note="Copied fixed physical truth and mu0 from A. Data were generated with A's different responses; B is only the processing model. mu0 must NOT be recomputed as S_B@truth.",
        model_B_at_A_truth_is_not_generator=mu_B_at_A_truth,
        lc_rule="For B refit both free and corresponding nonnegative auxiliary models using B; one auxiliary fit per method shared by all four target removals. Calculate B's own LC with unchanged full-model procedure. Never reuse A LC for B; no c multiplier and no null-constrained fit.",
        source_fit_count_this_experiment=0, newly_generated_observation_vectors=0,
        reused_observation_vectors=100000,
        calibration_warnings=scenario_A["calibration_warnings"] + [
            "Experimental B processing model: base xenon memory response replaced by fresh base response, before unchanged decay/daughter corrections. Data generation remains model A."])
    scenario_B["metadata"]["processing_variant"] = "B: equal xenon base responses; radon unchanged"
    scenario_B["metadata"]["base_response_convention"] = "R_g0_B xenon columns copied from R_c0; multiply original C_s_g/C_g_g; no renormalization; original F retained"
    write(ROOT / "scenario_B.json", scenario_B)
    files = ["prepare_scenario.py", "scenario_A.json", "scenario_arrays_A.npz",
             "scenario_B.json", "scenario_arrays_B.npz", "observations.npz"]
    write(ROOT / "scenario_B.lock.json", {name: sha(ROOT / name) for name in files})
    audit = dict(status="passed", no_random_number_generation=True, no_fits=True,
        source_observations_reused=100000, source_replicates_sha256=source_sha,
        unchanged_radon_base_column=True, unchanged_fresh_and_detector_model_columns=True,
        no_column_renormalization=True, model_rank_B=17,
        changed_base_entries=changed_base_entries, daughter_checks=daughter_checks,
        maximum_absolute_mean_mismatch=float(np.max(np.abs(mu_B_at_A_truth-mu0_A))),
        mean_mismatch_B_at_A_truth_minus_generating_A=(mu_B_at_A_truth-mu0_A),
        generating_mean_A_preserved=True, files_sha256={name:sha(ROOT/name) for name in files})
    write(ROOT / "preparation_verification.json", audit)
    print(json.dumps(jsonable(dict(status="passed", source_observations_reused=100000,
        changed_base_entries=changed_base_entries,
        daughter_coefficients={name: value["coefficient"] for name,value in daughter_checks.items()},
        maximum_absolute_mean_mismatch=audit["maximum_absolute_mean_mismatch"],
        scenario_B_sha256=sha(ROOT / "scenario_B.json"),
        observations_sha256=sha(ROOT / "observations.npz"))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
