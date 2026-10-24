"""Independent audit of the frozen strong-memory false-detection study.

No fits or new experiments are performed. Recreating the already saved random
counts verifies the seed convention; eight predetermined rows check LC by QR.
The ordinary portable audit uses replicates.npz and batch JSON metadata. Add
--source-batches to compare every merged field with the original batch NPZs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parent
FAMILIES = {"wls": "LS", "poisson": "Poisson ML"}
ISOTOPES = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
FIXED_IDS = [0, 999, 12345, 24999, 49999, 65432, 89999, 99999]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(a, b, message=""):
    np.testing.assert_array_equal(a, b, err_msg=message)


def close(a, b, message="", rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=message,
                               equal_nan=True)


def clopper_pearson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1))
    upper = 1.0 if k == n else float(beta.ppf(.975, k+1, n-k))
    return lower, upper


def check_scenario(root):
    scenario = read(root / "scenario.json")
    lock = read(root / "scenario.lock.json")
    for name, digest in lock.items():
        require(sha(root / name) == digest, "Scenario lock: " + name)
    runner = read(root / "runner_lock.json")
    for key, name in (("study_sha256", "study.py"),
                      ("scenario_sha256", "scenario.json"),
                      ("scenario_arrays_sha256", "scenario_arrays.npz")):
        require(runner[key] == sha(root/name), "Runner lock: " + name)
    require(runner["no_null_fit"] and runner["no_replacement"] and
            runner["no_threshold_multiplier"], "Frozen runner scope")
    require(scenario["production_commit"] ==
            "69ff5a62ccaaddeb6372f7f003109ce5ff6cf333", "Code version")
    snapshot = root / "kernel_snapshot"
    for name, digest in scenario["production_sha256"].items():
        require(sha(snapshot/name) == digest, "Kernel snapshot: " + name)
    for name, digest in runner["core_sha256"].items():
        require(sha(snapshot/name) == digest, "Runner kernel: " + name)
    for item in scenario["input_files"].values():
        require(sha(root/item["path"]) == item["sha256"], "Input PHD digest")
    require(scenario["input_files"]["DETBK"]["sha256"] ==
            "ca567f1b071fa29bba04b1c3ce9cfdc96ac7e5aa9bc3ec75cfb44b0ca39be950",
            "Same detector 221 background source as preceding study")
    require(scenario["use_fixed_roi"] and scenario["use_fixed_efficiencies"],
            "Fixed ROI and efficiencies")
    expected_eff = [.3, .241169, .367784, .233937, .233937]
    equal(scenario["efficiencies"]["sample"], expected_eff)
    equal(scenario["efficiencies"]["gasbk"], expected_eff)
    expected_br = dict(zip(["Rn222"] + ISOTOPES, [.37, .902, .37, .54, .5577]))
    require(scenario["branching"] == expected_br, "Branching values")
    require(scenario["n"] == 100000 and scenario["batch_size"] == 1000 and
            scenario["n_batches"] == 100, "Frozen sample size")
    require(scenario["alpha"] == .05 and scenario["master_seed"] == 2026102801,
            "Alpha and seed")
    require(scenario["truth_construction"]["source_fit_count"] == 1,
            "One source-data nonnegative fit")
    sys.path.insert(0, str(snapshot))
    from xemat.model import JointModel
    from process_measurement import _constants_input, _read_spectra
    model = JointModel(**scenario["model"])
    a = dict(np.load(root/"scenario_arrays.npz", allow_pickle=False))
    equal(model.M, a["M"])
    equal(model.T_c, a["T_c"])
    equal(model.b, a["b"])
    equal(model.x, a["x_observed"])
    equal(model.physical_matrix, a["physical_matrix"])
    original, truth = a["original_nonnegative_physical"], a["truth_physical"]
    require(original.shape == truth.shape == (17,), "Physical parameter dimensions")
    require(np.all(original >= 0), "Nonnegative source estimate")
    equal(truth[1:5], np.zeros(4))
    keep = np.r_[0, 5:17]
    equal(truth[keep], original[keep])
    equal(truth, scenario["truth_physical"])
    equal(original, scenario["original_nonnegative_physical"])
    equal(a["mu0"], model.physical_matrix @ truth + model.b)
    require(np.all(a["mu0"] > 0), "Positive generating mean")
    S = a["physical_matrix"]
    equal(a["mu0_fresh_radon"], S[:, 0] * truth[0])
    equal(a["mu0_memory"], S[:, 5:10] @ truth[5:10])
    equal(a["mu0_detector_background"], S[:, 10:] @ truth[10:])
    close(a["mu0"], a["mu0_fresh_radon"] + a["mu0_memory"] +
          a["mu0_detector_background"] + a["b"], rtol=1e-14)
    equal(a["mu0_fresh_xenon"], np.zeros(21))
    equal(S[14:, 10:], np.eye(7))
    equal(S[14:, :10], np.zeros((7,10)))
    equal(S[7:, :5], np.zeros((14,5)))
    equal(model.x[14:], [522, 854, 494, 66, 64, 89, 97])
    # Rebuild only parsing and calibration; there are no fit/detection calls here.
    paths = [str(root/scenario["input_files"][role]["path"])
             for role in ("SAMPLE", "GASBK", "DETBK")]
    rebuilt, extras, metadata = _constants_input(paths, use_fixed_roi=True,
        radon_mode="legacy", radon_h=None, use_fixed_efficiencies=True)
    equal(model.M, rebuilt.M)
    equal(model.T_c, rebuilt.T_c)
    equal(model.x, rebuilt.x)
    equal(model.b, rebuilt.b)
    parsed = _read_spectra(paths)
    for role, key in (("sample", "sample"), ("gasbk", "background"),
                      ("detbk", "detector")):
        for field in ("histogram", "g_spectrum", "b_spectrum", "g_energy", "b_energy"):
            equal(a[f"{role}_{field}"], parsed[key][field])
    return scenario, a


def check_method(family, arrays):
    free = arrays[family + "_free_physical"]
    pilot = arrays[family + "_auxiliary_physical"]
    fstatus = arrays[family + "_free_status"]
    pstatus = arrays[family + "_auxiliary_status"]
    lcstatus = arrays[family + "_LC_status"]
    estimate = arrays[family + "_concentrations"]
    lc = arrays[family + "_LC"]
    decision = arrays[family + "_decision"]
    n = len(free)
    require(free.shape == pilot.shape == (n,17), family + " full estimates shape")
    require(estimate.shape == lc.shape == lcstatus.shape == decision.shape == (n,4),
            family + " target arrays shape")
    require(set(np.unique(fstatus)) <= {-2,-1,0,1,2}, "Free status values")
    require(set(np.unique(pstatus)) <= {-2,-1,0,1,2}, "Auxiliary status values")
    require(set(np.unique(lcstatus)) <= {-2,-1,0,1}, "LC status values")
    require(set(np.unique(decision)) <= {-1,0,1}, "Decision values")
    require(np.all(np.isfinite(free[fstatus == 1])), "Finite successful free estimates")
    require(np.all(np.isfinite(pilot[pstatus == 1])), "Finite successful auxiliary estimates")
    require(np.all(pilot[pstatus == 1] >= 0), "Nonnegative physical auxiliary estimates")
    equal(estimate, free[:,1:5]*1000)
    require(np.all(np.isfinite(lc[lcstatus == 1])) and np.all(lc[lcstatus == 1] >= 0),
            "Available LC values")
    valid = (fstatus[:,None] == 1) & (pstatus[:,None] == 1) & (lcstatus == 1)
    equal(decision >= 0, valid, "Available decisions match fitting and LC statuses")
    equal(decision[valid], (estimate > lc)[valid], "Strict free-estimate > LC rule")
    covariance_available = arrays[family + "_free_covariance_available"]
    return dict(n_generated=n, n_valid=valid.sum(axis=0).tolist(),
        n_false=(decision == 1).sum(axis=0).tolist(),
        n_fail=(~valid).sum(axis=0).tolist(),
        negative_free_counts=((estimate < 0) & (fstatus[:,None] == 1)).sum(axis=0).tolist(),
        successful_free_covariance_unavailable=int(np.sum((fstatus == 1) & ~covariance_available)),
        covariance_unavailable_but_decision_available=(valid & ~covariance_available[:,None]).sum(axis=0).tolist(),
        covariance_unavailable_but_any_decision_available=int(np.sum(valid.any(axis=1) & ~covariance_available)),
        free_status_counts={str(k): int(v) for k,v in Counter(fstatus.tolist()).items()},
        auxiliary_status_counts={str(k): int(v) for k,v in Counter(pstatus.tolist()).items()})


def qr_lc_checks(scenario_arrays, replicates):
    S, b = scenario_arrays["physical_matrix"], scenario_arrays["b"]
    checks = []
    for identifier in FIXED_IDS:
        for family in FAMILIES:
            if replicates[family+"_auxiliary_status"][identifier] != 1:
                continue
            pilot = replicates[family+"_auxiliary_physical"][identifier]
            for column, target in enumerate([1,2,3,4]):
                if replicates[family+"_LC_status"][identifier,column] != 1:
                    continue
                nuisance = pilot.copy()
                nuisance[target] = 0
                mu = S @ nuisance + b
                require(np.all(mu > 0), "QR spotcheck positive null mean")
                A = S / np.sqrt(mu)[:,None]
                scales = np.linalg.norm(A, axis=0)
                _, R = np.linalg.qr(A/scales, mode="reduced")
                inverse = solve_triangular(R, np.eye(17))
                sigma = np.linalg.norm(inverse[target]) / scales[target]
                independent = 1000 * norm.ppf(.95) * sigma
                saved = replicates[family+"_LC"][identifier,column]
                close(independent, saved, "Independent scaled-QR LC", rtol=2e-10, atol=1e-12)
                checks.append(dict(replicate_id=identifier, method=family,
                    isotope=ISOTOPES[column], recorded_LC_mBq_m3=float(saved),
                    independent_QR_LC_mBq_m3=float(independent),
                    relative_difference=float((independent-saved)/saved)))
    return checks


def check_summary(root, arrays):
    with (root/"summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 8, "Exactly eight summary rows")
    lookup = {(row["method"],row["isotope"]):row for row in rows}
    require(len(lookup) == 8, "Unique method/isotope summary rows")
    compared = 0
    for family, label in FAMILIES.items():
        fstatus = arrays[family+"_free_status"]
        for col, isotope in enumerate(ISOTOPES):
            row = lookup[label,isotope]
            d = arrays[family+"_decision"][:,col]
            n, nv, k = len(d), int(np.sum(d >= 0)), int(np.sum(d == 1))
            nf = n-nv
            lo,hi = clopper_pearson(k,nv)
            estimates = arrays[family+"_concentrations"][:,col]
            estimates = estimates[(fstatus == 1) & np.isfinite(estimates)]
            lc = arrays[family+"_LC"][:,col]
            lc = lc[(arrays[family+"_LC_status"][:,col] == 1) & np.isfinite(lc)]
            qs = np.quantile(lc,[.025,.5,.975]) if len(lc) else [np.nan]*3
            expected = dict(n_generated=n,n_valid=nv,n_false=k,n_fail=nf,
                frequency=k/nv if nv else np.nan,ci95_low=lo,ci95_high=hi,
                frequency_all_min=k/n,frequency_all_max=(k+nf)/n,
                mean_estimate_mBq_m3=np.mean(estimates) if len(estimates) else np.nan,
                sd_estimate_mBq_m3=np.std(estimates,ddof=1) if len(estimates)>1 else np.nan,
                lc_q025_mBq_m3=qs[0],lc_median_mBq_m3=qs[1],lc_q975_mBq_m3=qs[2],
                n_available_estimate=len(estimates),n_available_lc=len(lc),
                n_covariance_unavailable_decision=int(np.sum(
                    (d >= 0) & ~arrays[family+"_free_covariance_available"])))
            for name,value in expected.items():
                actual = float(row[name]) if row[name] else np.nan
                close(actual,value, f"CSV {label}/{isotope}/{name}",rtol=2e-11,atol=2e-13)
                compared += 1
    return compared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT)
    parser.add_argument("--source-batches",action="store_true",
                        help="Also compare original points/batch_NNN.npz against merged arrays")
    args=parser.parse_args()
    root=args.study.resolve()
    scenario,scenarios=check_scenario(root)
    arrays=dict(np.load(root/"replicates.npz",allow_pickle=False))
    n=scenario["n"]
    equal(arrays["replicate_id"],np.arange(n))
    equal(arrays["chunk"],np.repeat(np.arange(100),1000))
    equal(arrays["row_in_batch"],np.tile(np.arange(1000),100))
    equal(arrays["isotope_order"],ISOTOPES)
    require(str(arrays["concentration_units"].item())=="mBq/m3", "Output units")
    require(arrays["x"].shape==(n,21),"All shared observations retained")
    equal(arrays["seed_table"],[[scenario["master_seed"],chunk] for chunk in range(100)])
    require(str(arrays["scenario_sha256"].item())==sha(root/"scenario.json"),"Merged scenario hash")
    require(str(arrays["runner_lock_sha256"].item())==sha(root/"runner_lock.json"),"Merged runner hash")
    catalogue=[json.loads(str(item)) for item in arrays["diagnostic_catalogue_json"]]
    catalogue_map={json.dumps(item,ensure_ascii=False,sort_keys=True):i
                   for i,item in enumerate(catalogue)}
    require(len(catalogue_map)==len(catalogue),"Unique global diagnostic entries")
    diagnostics=read(root/"diagnostics.json")
    require(diagnostics["diagnostic_catalogue"]==catalogue,"Diagnostic JSON matches NPZ catalog")
    batch_records=[]
    expected_errors=[]
    for chunk in range(100):
        selected=slice(chunk*1000,(chunk+1)*1000)
        seed=[scenario["master_seed"],chunk]
        generated=np.random.default_rng(seed).poisson(scenarios["mu0"],size=(1000,21))
        equal(arrays["x"][selected],generated,"Exact saved draw regeneration")
        metadata_path=root/"batch_metadata"/f"batch_{chunk:03d}.json"
        if not metadata_path.exists():
            metadata_path=root/"points"/f"batch_{chunk:03d}.json"
        meta=read(metadata_path)
        require(meta["seed"]==seed and meta["n_generated"]==1000,"Batch seed/count")
        require(meta["runner_lock_sha256"]==sha(root/"runner_lock.json"),"Batch runner lock")
        mapping=np.array([catalogue_map[json.dumps(item,ensure_ascii=False,sort_keys=True)]
                          for item in meta["diagnostic_catalogue"]],dtype=np.int32)
        expected_errors.extend(dict(chunk=chunk,**error) for error in meta["errors"])
        for family in FAMILIES:
            for phase in ("free","auxiliary"):
                codes=arrays[f"{family}_{phase}_diagnostic_code"][selected]
                require(np.all((codes>=0)&(codes<len(catalogue))),"Fit diagnostic catalog bounds")
            codes=arrays[family+"_LC_diagnostic_code"][selected]
            require(np.all((codes>=0)&(codes<len(catalogue))),"LC diagnostic catalog bounds")
        if args.source_batches:
            path=root/"points"/f"batch_{chunk:03d}.npz"
            require(sha(path)==meta["npz_sha256"],"Original source batch SHA")
            source=dict(np.load(path,allow_pickle=False))
            for key,value in source.items():
                if key=="seed":
                    equal(value,seed)
                elif value.ndim and value.shape[0]==1000:
                    if key.endswith("_diagnostic_code"):
                        value=mapping[value]
                    equal(arrays[key][selected],value,"Exact merged source field: "+key)
                else:
                    equal(arrays[key],value,"Exact merged invariant: "+key)
        batch_records.append(dict(chunk=chunk,npz_sha256=meta["npz_sha256"],
                                  metadata_sha256=sha(metadata_path),n_error_records=len(meta["errors"])))
    require(diagnostics["errors"]==expected_errors,"All failure/recovery error records retained")
    for family in FAMILIES:
        for phase in ("free","auxiliary"):
            prefix=family+"_"+phase
            saved=diagnostics["fit_diagnostics"][family][phase]
            counts=Counter(arrays[prefix+"_diagnostic_code"].tolist())
            messages,warnings,python_warnings=Counter(),Counter(),Counter()
            for code,amount in counts.items():
                diagnostic=catalogue[code]
                selected=arrays[prefix+"_diagnostic_code"]==code
                status_values=set(arrays[prefix+"_status"][selected].tolist())
                if "success" in diagnostic:
                    expected_status={1,2} if diagnostic["success"] else {0}
                    require(status_values<=expected_status,
                            "Solver success flag matches saved fit status")
                elif "exception" in diagnostic:
                    require(status_values=={-1},"Exception diagnostic matches fit status")
                messages[str(diagnostic.get("message",diagnostic.get("exception","No message")))]+=amount
                for value in diagnostic.get("warnings",[]):
                    warnings[str(value)]+=amount
                for value in diagnostic.get("python_warnings",[]):
                    python_warnings[str(value)]+=amount
            require(saved["messages"]==dict(messages),"Optimizer message counts")
            require(saved["warnings"]==dict(warnings),"Optimizer warning counts")
            require(saved["python_warnings"]==dict(python_warnings),"Python warning counts")
            expected_status={str(k):v for k,v in Counter(arrays[prefix+"_status"].tolist()).items()}
            require(saved["status_counts"]==expected_status,"Optimizer status counts")
    method_checks={family:check_method(family,arrays) for family in FAMILIES}
    qr=qr_lc_checks(scenarios,arrays)
    csv_values=check_summary(root,arrays)
    result=dict(status="passed",verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)), n=100000, exact_seed_count_vectors_verified=n,
        all_source_batch_values_compared=bool(args.source_batches),
        source_batches=batch_records, methods=method_checks,
        numeric_summary_values_verified=csv_values,
        independent_lc_method="Column-scaled QR and triangular inverse, no XeMat covariance call",
        independent_lc_predetermined_ids=FIXED_IDS, independent_lc_checks=qr,
        no_refitting=True,no_new_experiments=True,
        files_sha256={name:sha(root/name) for name in
            ("scenario.json","scenario_arrays.npz","runner_lock.json","replicates.npz","summary.csv")})
    (root/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","n","numeric_summary_values_verified","methods")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
