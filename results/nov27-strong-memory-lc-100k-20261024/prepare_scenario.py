"""Freeze the strong-GASBK null scenario using the unchanged XeMat kernel.

Exactly one nonnegative Poisson fit is made to the original three spectra.
Only its four fresh xenon concentrations are then set to zero.  This is a
fixed plug-in generating scenario, not a fit under the null hypothesis.
This script does not calculate LC, LD, MDC, power, or calibrated thresholds.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = ROOT / "kernel_snapshot" if (ROOT / "kernel_snapshot").is_dir() else Path("/workspace/first-project")
REPO = Path(os.environ.get("XEMAT_REPO", str(DEFAULT_REPO)))
sys.path.insert(0, str(REPO))

import numpy as np
import scipy
from constants import FIXED_ROI_TABLE, HALF_LIFE_DAYS, ROI_INFO, XENON_ISOTOPES
from ims_parser import convert_roi_energy_to_channels, fit_energy_calibration
from process_measurement import _constants_input, _read_spectra
from xemat.diagnostics import compute_fit_quality
from xemat.io import jsonable
from xemat.solvers import fit

ISOTOPES = ["Rn222", "Xe135", "Xe133", "Xe131m", "Xe133m"]
ROIS = [1, 2, 3, 5, 6, 7, 8]
EXPECTED_HEAD = "69ff5a62ccaaddeb6372f7f003109ce5ff6cf333"
SOURCE_PATHS = {
    "SAMPLE": Path("/workspace/attachments/99dec4da-7b31-4526-93e5-cc537f33073f/SPIKE20251127072232736.PHD"),
    "GASBK": Path("/workspace/attachments/0b8a1615-e19e-4146-bfcd-3fd0c5f6b714/GASBK20251126153029165.PHD"),
    "DETBK": Path("/workspace/scratch/real-spectra-3/DETBK20250610213044796.PHD"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identifiers(path):
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    result = {}
    for line in lines:
        for name in ("MSG_ID", "DATA_TYPE"):
            if line.startswith(name + " "):
                result[name.lower()] = line.split(maxsplit=1)[1]
    i = next(i for i, line in enumerate(lines) if line.startswith("#Header"))
    result.update(header=lines[i+1], sample_identifier=lines[i+2],
                  linked_measurements=lines[i+3], message_time=lines[i+4])
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "scenario.json").exists() or (ROOT / "scenario_arrays.npz").exists():
        raise FileExistsError("Scenario already frozen; do not overwrite it")
    if (REPO / "snapshot_version.json").exists():
        head = json.loads((REPO / "snapshot_version.json").read_text())["production_commit"]
    else:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    assert head == EXPECTED_HEAD
    source_names = ["constants.py", "ims_parser.py", "process_measurement.py"]
    source_names += [str(p.relative_to(REPO)) for p in sorted((REPO / "xemat").glob("*.py"))]
    production_hashes = {name: sha(REPO / name) for name in source_names}
    snapshot = ROOT / "kernel_snapshot"
    for name in source_names:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (REPO / name).resolve() != destination.resolve():
            shutil.copyfile(REPO / name, destination)
        assert sha(destination) == production_hashes[name]
    write_json(snapshot / "snapshot_version.json", dict(production_commit=head, sha256=production_hashes))
    (ROOT / "inputs").mkdir(exist_ok=True)
    paths = []
    provenance = {}
    for role, source in SOURCE_PATHS.items():
        destination = ROOT / "inputs" / source.name
        if source.exists():
            shutil.copyfile(source, destination)
        elif not destination.exists():
            raise FileNotFoundError(source)
        paths.append(str(destination))
        provenance[role] = dict(path=str(destination.relative_to(ROOT)),
            original_path=str(source), sha256=sha(destination), bytes=destination.stat().st_size,
            **identifiers(destination))

    # Reuse the exact application parser, fixed ROI extraction, and matrix builder.
    # _constants_input stops before any fitting or detection-limit calculations.
    model, matrices, metadata = _constants_input(paths, use_fixed_roi=True,
        radon_mode="legacy", radon_h=None, use_fixed_efficiencies=True)
    parsed = _read_spectra(paths)
    assert all(data["detector_id"] == 221 for data in parsed.values())
    for label in ("sample_efficiencies", "gasbk_efficiencies"):
        np.testing.assert_array_equal(metadata[label], [.241169, .367784, .233937, .233937])
    np.testing.assert_array_equal(model.x[14:], [522, 854, 494, 66, 64, 89, 97])
    S = model.physical_matrix
    assert S.shape == (21, 17)
    assert np.linalg.matrix_rank(S / np.linalg.norm(S, axis=0)) == 17
    assert np.all(model.b == 0)

    # The sole source-data fit. No fitted sample is forced to satisfy a null.
    original_fit = fit(model, "poisson_nonnegative")
    original = original_fit.physical_parameters
    if not original_fit.success or original is None or not np.all(np.isfinite(original)):
        raise RuntimeError("Source nonnegative Poisson fit failed: " + original_fit.message)
    if np.any(original < 0):
        raise ValueError("Source physical nonnegative estimate contains a negative entry")
    truth = original.copy()
    truth[1:5] = 0.0
    mu0 = S @ truth + model.b
    if not np.all(mu0 > 0):
        raise ValueError("Generating mean must be positive for this scenario")
    fresh_radon = S[:, 0] * truth[0]
    memory = S[:, 5:10] @ truth[5:10]
    detector = S[:, 10:] @ truth[10:]
    np.testing.assert_allclose(mu0, fresh_radon + memory + detector + model.b,
                               rtol=1e-14, atol=1e-12)
    np.testing.assert_array_equal(truth[np.r_[0,5:17]], original[np.r_[0,5:17]])
    np.testing.assert_array_equal(truth[1:5], np.zeros(4))

    physical_names = ["a_V:" + name for name in ISOTOPES]
    physical_names += ["gamma:" + name for name in ISOTOPES]
    physical_names += [f"zeta:ROI{number}" for number in ROIS]
    physical_units = ["Bq/m3 (auxiliary radon coordinate, not atmospheric Rn concentration)"]
    physical_units += ["Bq/m3"] * 4
    physical_units += ["reference-ROI expected counts in GASBK (before daughter correction)"] * 5
    physical_units += ["expected counts in DETBK"] * 7
    observations = [f"{block}:ROI{roi}" for block in ("SAMPLE", "GASBK", "DETBK") for roi in ROIS]
    arrays = dict(M=model.M, T_c=model.T_c, T_V=model.T_V, b=model.b,
        x_observed=model.x.astype(np.int64), physical_matrix=S,
        original_nonnegative_physical=original, original_nonnegative_theta=original_fit.theta,
        original_nonnegative_mu=original_fit.mu, truth_physical=truth,
        truth_theta=model.from_physical(truth), mu0=mu0,
        mu0_fresh_radon=fresh_radon, mu0_memory=memory,
        mu0_detector_background=detector, mu0_fixed_offset=model.b,
        mu0_fresh_xenon=np.zeros(21), physical_parameter_names=np.array(physical_names),
        physical_parameter_units=np.array(physical_units), observation_names=np.array(observations),
        observation_units=np.array("counts"), xenon_order=np.array(ISOTOPES[1:]),
        output_concentration_units=np.array("mBq/m3"), roi_order=np.array(ROIS),
        sample_x_observed=model.x[:7].astype(np.int64),
        gasbk_x_observed=model.x[7:14].astype(np.int64),
        detbk_x_observed=model.x[14:].astype(np.int64))
    spectra = {}
    for role, key in (("SAMPLE", "sample"), ("GASBK", "background"), ("DETBK", "detector")):
        data = parsed[key]
        gfunc, bfunc = fit_energy_calibration(data["g_energy"], data["b_energy"])
        roi_energy = np.array([FIXED_ROI_TABLE[number] for number in ROIS])
        roi_channels = np.clip(np.round(convert_roi_energy_to_channels(roi_energy,gfunc,bfunc)),0,255).astype(int)
        for field in ("histogram", "g_spectrum", "b_spectrum", "g_energy", "b_energy"):
            arrays[f"{role.lower()}_{field}"] = np.asarray(data[field])
        arrays[f"{role.lower()}_roi_channels"] = roi_channels
        spectra[role] = dict(start_time=data["start_time"],
            end_time=data["start_time"] + timedelta(seconds=data["real_time"]),
            real_time_seconds=data["real_time"], live_time_seconds=data["live_time"],
            collection_start=data["collection_start"], collection_end=data["collection_end"],
            reported_air_volume_m3=data["air_volume"], xenon_volume_cm3=data["xenon_volume"],
            detector_id=data["detector_id"], detector_name=data["detector_name"],
            roi_channels_inclusive_beta_min_max_gamma_min_max=roi_channels,
            file_roi_numbers=data["roi_numbers"], file_roi_limits_keV=data["roi_limits"],
            file_efficiencies_not_used=data["efficiency"], file_roi_to_isotope=data.get("roi_to_isotope", {}),
            histogram_shape=np.asarray(data["histogram"]).shape,
            histogram_axis_order="row=gamma channel; column=beta channel; as stored by existing parser")
    for name, value in matrices.items():
        if isinstance(value, np.ndarray):
            arrays["calibration_" + name] = value
    np.savez_compressed(ROOT / "scenario_arrays.npz", **arrays)

    sample, gasbk, detbk = parsed["sample"], parsed["background"], parsed["detector"]
    fit_diagnostics = {name: getattr(original_fit, name) for name in
        ("method", "success", "message", "warnings", "iterations", "objective", "rank", "condition_number", "coordinates")}
    fit_diagnostics["physical_covariance_available"] = original_fit.physical_covariance is not None
    fit_diagnostics["fit_quality"] = jsonable(compute_fit_quality(model, original_fit))
    scenario = dict(schema_version=1, scenario_id="nov27_2025_strong_memory_detector221",
        created_utc=datetime.now(timezone.utc).isoformat(), n=100000, N=100000, n_replicates=100000, alpha=.05,
        master_seed=2026102801, n_batches=100, batch_size=1000,
        seed_rule="numpy.random.default_rng([master_seed, chunk]); chunk=0..99; poisson(mu0, size=(1000,21))",
        bit_generator="PCG64", families={"wls":"nnls", "poisson":"poisson_nonnegative"},
        methods=["wls", "poisson"], isotopes=ISOTOPES[1:], target_indices=[1,2,3,4],
        observation_order=observations, observation_blocks=["SAMPLE", "GASBK", "DETBK"],
        roi_order=ROIS, observation_units="counts", output_concentration_units="mBq/m3",
        physical_coordinates="[a_V; gamma; zeta]", input_coordinates=model.coordinates,
        physical_parameter_names=physical_names, physical_parameter_units=physical_units,
        input_parameter_names=matrices["parameter_names"],
        model=jsonable(model), metadata=metadata, calibration_matrices=matrices,
        truth_physical=truth, original_nonnegative_physical=original, mu0=mu0,
        input_files=provenance, measurement_metadata=spectra,
        use_fixed_roi=True, fixed_roi_energy_limits_keV={number:FIXED_ROI_TABLE[number] for number in ROIS},
        roi_energy_columns=["beta_min", "beta_max", "gamma_min", "gamma_max"],
        use_fixed_efficiencies=True,
        efficiencies=dict(isotopes=ISOTOPES, sample=[.3]+metadata["sample_efficiencies"],
                          gasbk=[.3]+metadata["gasbk_efficiencies"], units="dimensionless fraction; branching excluded"),
        branching=dict(zip(ISOTOPES, [ROI_INFO[name]["br"] for name in ISOTOPES])),
        sensitivities=dict(sample=matrices["I_c"], gasbk=matrices["I_g"], formula="I_j=epsilon_j*branching_j"),
        half_life_days=HALF_LIFE_DAYS, decay_constants_per_second=XENON_ISOTOPES,
        effective_air_volume_m3=metadata["effective_air_volume_m3"],
        xenon_volume_cm3=metadata["xenon_volume_cm3"],
        volume_formula="V0=xenon_volume_cm3/0.087; native a_V in Bq/m3; displayed Xe concentrations multiply by 1000",
        time_parameters_seconds=metadata["time_parameters"],
        measurement_intervals_seconds=dict(
            sample_start_minus_gasbk_start=(sample["start_time"]-gasbk["start_time"]).total_seconds(),
            sample_start_minus_gasbk_end=(sample["start_time"]-gasbk["start_time"]).total_seconds()-gasbk["real_time"],
            sample_start_minus_detbk_start=(sample["start_time"]-detbk["start_time"]).total_seconds(),
            gasbk_start_minus_detbk_start=(gasbk["start_time"]-detbk["start_time"]).total_seconds()),
        truth_construction=dict(method="poisson_nonnegative", source_fit_count=1,
            description="One unchanged joint nonnegative Poisson ML fit to original SAMPLE-role SPIKE, GASBK, DETBK. Set only fresh Xe indices 1:5 of its physical estimate to exact zero; retain fresh Rn, all five memory coordinates, and all seven fitted DETBK means. No null refit.",
            fresh_xenon_zero_indices=[1,2,3,4], unchanged_indices=[0]+list(range(5,17)),
            detector_background="Same DETBK source file as previous study; generating zeta is the jointly fitted value, not a new measured background and not the unadjusted raw counts",
            uncertainty_of_plugin_truth_resampled=False, original_fit_diagnostics=fit_diagnostics),
        generator="All 21 SAMPLE/GASBK/DETBK ROI counts are independent Poisson(mu0), same realization used by both methods",
        weights=dict(wls="W=diag(1/where(x==0,1,x)); recomputed from every realization, all 21 original ROI counts", nnls="Identical criterion and weights as WLS; nonnegativity on all physical [a_V; gamma; zeta]", zero_counts="Only the WLS variance of a zero count is replaced by 1; x remains zero; Poisson uses original zeros"),
        optimizer=dict(entrypoint="xemat.solvers.fit", max_iter=2000, tol=1e-10, settings="Existing defaults; no overrides or algorithm changes"),
        lc_rule="For each method one corresponding nonnegative auxiliary fit per realization; remove each tested fresh Xe contribution from its expected mean in turn, retain all other fitted coordinates, use existing full-model covariance to calculate LC at alpha=.05. No null refit. Detect iff free a_V_i > LC_i. Native Bq/m3; both multiply by 1000 for output. No c multiplier.",
        missing_policy="Keep every generated realization and failure reason; no replacement or selection. A missing free-fit covariance alone does not invalidate an available free estimate and LC.",
        calibration_warnings=matrices["warnings"], systematics=False, threshold_multiplier=1.0,
        no_power_curves=True, production_commit=head, production_sha256=production_hashes,
        scenario_arrays_sha256=sha(ROOT / "scenario_arrays.npz"),
        preparation_script_sha256=sha(Path(__file__)),
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        memory_units_note="gamma is a GASBK reference-ROI expected-count coordinate, not concentration; gamma is retained without volume conversion",
        time_zone_note="File timestamps are naive as parsed; no timezone conversion is applied",
        seed_table=[[2026102801, chunk] for chunk in range(100)])
    write_json(ROOT / "scenario.json", scenario)
    write_json(ROOT / "scenario.lock.json", {name:sha(ROOT/name) for name in
        ("scenario.json", "scenario_arrays.npz", "prepare_scenario.py")})
    print(json.dumps(jsonable(dict(source_fit=fit_diagnostics, x=model.x, original=original,
        truth=truth, mu0=mu0, volume=metadata["effective_air_volume_m3"],
        times=metadata["time_parameters"], scenario_sha256=sha(ROOT / "scenario.json"))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
