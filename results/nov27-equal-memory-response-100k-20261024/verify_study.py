"""Independent audit of the paired original/equal-response analysis.

No observations, fits, or additional scientific experiments are produced.
The B matrix is assembled independently from original A calibration arrays.
LC checks use scaled QR on eight predetermined saved rows, not XeMat's SVD.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.linalg import block_diag, solve_triangular
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parent
SOURCE_A = Path("/workspace/scratch/nov27-strong-memory-lc-100k-20261024")
SOURCE_A_SHA = "ede7d13cb1c7d3893550088b7b5ddc1232660f8831a780ac637cd05405da7804"
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
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol,
                               err_msg=message, equal_nan=True)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def cp(k, n):
    if not n:
        return np.nan, np.nan
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))


def check_scenario(root):
    A, B = read(root/"scenario_A.json"), read(root/"scenario_B.json")
    a, b = load(root/"scenario_arrays_A.npz"), load(root/"scenario_arrays_B.npz")
    for name, digest in read(root/"scenario_B.lock.json").items():
        require(sha(root/name) == digest, "Frozen scenario changed: " + name)
    for name, digest in A["production_sha256"].items():
        require(sha(root/"kernel_snapshot"/name) == digest, "Unchanged kernel: " + name)
    require(A["production_sha256"] == B["production_sha256"], "Same production kernel")
    require(A["production_commit"] == B["production_commit"] ==
            "69ff5a62ccaaddeb6372f7f003109ce5ff6cf333", "Production version")
    require(B["source_replicates_sha256"] == SOURCE_A_SHA, "Original ensemble provenance")
    require(B["source_fit_count_this_experiment"] == B["newly_generated_observation_vectors"] == 0,
            "No generation or source fit in this experiment")
    require(B["reused_observation_vectors"] == B["n"] == 100000, "All saved observations reused")
    require(B["variant"] == "B" and B["generation_model_variant"] == "A", "Variant roles")
    require(B["threshold_multiplier"] == 1 and B["alpha"] == .05, "Unmodified LC level")
    for key in ("truth_physical", "original_nonnegative_physical", "mu0", "weights", "optimizer",
                "efficiencies", "branching", "time_parameters_seconds", "measurement_intervals_seconds",
                "effective_air_volume_m3", "xenon_volume_cm3", "roi_order", "systematics"):
        require(A[key] == B[key], "Same scenario field: " + key)
    for item in A["input_files"].values():
        require(sha(root/item["path"]) == item["sha256"], "Original PHD unchanged")
    expected_eff = [.3, .241169, .367784, .233937, .233937]
    equal(A["efficiencies"]["sample"], expected_eff)
    equal(A["efficiencies"]["gasbk"], expected_eff)
    require(A["branching"] == dict(zip(["Rn222"]+ISOTOPES,[.37,.902,.37,.54,.5577])), "Branching")

    c = {key: np.asarray(value) for key, value in A["calibration_matrices"].items()}
    # Independent column assembly rather than calling prepare_scenario or the
    # production joint-matrix builder.
    Rg0 = np.column_stack([c["R_g0"][:, 0], c["R_c0"][:, 1:5]])
    Rg1, Rg2 = Rg0 @ c["C_s_g"], Rg0 @ c["C_g_g"]
    zeros, identity = np.zeros((7,5)), np.eye(7)
    M = np.block([[c["R_c"], Rg1 @ c["F"], float(c["kappa_s"]) * identity],
                  [zeros, Rg2, float(c["kappa_g"]) * identity],
                  [zeros, zeros, identity]])
    equal(M, b["M_B"], "Independent matrix B")
    equal(M, b["M"])
    equal(M, B["model"]["M"])
    equal(b["M_A"], a["M"])
    equal(b["R_g0_B"], Rg0)
    equal(b["R_g1_B"], Rg1)
    equal(b["R_g2_B"], Rg2)
    Q = block_diag(np.asarray(A["model"]["T_c"]), np.eye(12))
    S = np.linalg.solve(Q.T, M.T).T
    close(S, b["physical_matrix_B"], "Independent physical transform", rtol=1e-14)
    equal(b["physical_matrix"], b["physical_matrix_B"])
    equal(b["physical_matrix_A"], a["physical_matrix"])
    keep = [0,1,2,3,4,5,10,11,12,13,14,15,16]
    equal(M[:, keep], a["M"][:, keep], "Fresh/Rn-memory/DETBK columns unchanged")
    equal(Rg0[:, 0], c["R_g0"][:, 0], "Radon base column unchanged")
    for key in ("R_c0", "R_c", "C_s_c", "C_s_g", "C_g_g", "F", "H", "H_g1", "H_g2",
                "Gamma", "G_p", "G_s", "T_c", "T_V", "kappa_s", "kappa_g"):
        equal(B["calibration_matrices"][key], A["calibration_matrices"][key], key)
    for key in ("T_c", "T_V", "b", "x_observed", "truth_physical", "original_nonnegative_physical",
                "mu0", "mu0_fresh_radon", "mu0_memory", "mu0_detector_background"):
        equal(a[key], b[key], "A generating scenario preserved: " + key)
    equal(a["truth_physical"][1:5], np.zeros(4))
    close(a["mu0"], a["physical_matrix"] @ a["truth_physical"] + a["b"])
    close(b["mu_model_B_at_A_truth"], S @ a["truth_physical"] + a["b"])
    require(not np.allclose(b["mu0"], S @ a["truth_physical"] + a["b"]),
            "Intentional B misspecification must not become a new generating mean")
    daughter = []
    for label, C, R in (("SAMPLE_memory",c["C_s_g"],Rg1),("GASBK",c["C_g_g"],Rg2)):
        expected = Rg0[:,4] + C[2,4] * Rg0[:,2]
        close(R[:,4], expected, "Xe133m mother + daughter", rtol=1e-15, atol=1e-15)
        daughter.append(dict(block=label, coefficient=float(C[2,4]), mother_plus_daughter_verified=True))

    observations = load(root/"observations.npz")
    require(str(observations["source_replicates_sha256"].item()) == SOURCE_A_SHA, "Observation source SHA")
    require(observations["x"].shape == (100000,21) and
            np.issubdtype(observations["x"].dtype,np.integer), "Full original integer observations")
    equal(observations["replicate_id"], np.arange(100000))
    equal(observations["seed_table"], [[2026102801,i] for i in range(100)])
    require(sha(root/"observations.npz") == B["observations_sha256"], "Observation array hash")
    runner = read(root/"runner_lock.json")
    for key, name in (("study_sha256","study.py"),("scenario_sha256","scenario_B.json"),
                      ("scenario_arrays_sha256","scenario_arrays_B.npz"),("observations_sha256","observations.npz")):
        require(runner[key] == sha(root/name), "Frozen runner: " + name)
    require(runner["no_new_observations"] and runner["no_null_fit"] and runner["no_replacement"] and
            runner["no_threshold_multiplier"], "Frozen experiment scope")
    for name, digest in runner["core_sha256"].items():
        require(sha(root/"kernel_snapshot"/name) == digest, "Runner kernel")
    tree = ast.parse((root/"study.py").read_text())
    require(not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and
                    node.func.attr in ("poisson","default_rng","seed") for node in ast.walk(tree)),
            "No random generation calls in the B runner")
    return {"A":A,"B":B}, {"A":a,"B":b}, observations, daughter


def check_method(family, arrays):
    free = arrays.get(family+"_free_physical")
    pilot = arrays[family+"_auxiliary_physical"]
    fs, ps = arrays[family+"_free_status"], arrays[family+"_auxiliary_status"]
    ls = arrays[family+"_LC_status"]
    estimate, lc, decision = (arrays[family+suffix] for suffix in ("_concentrations","_LC","_decision"))
    n = len(pilot)
    require(pilot.shape == (n,17), "Full auxiliary physical vectors retained")
    require(estimate.shape == lc.shape == ls.shape == decision.shape == (n,4), "All target arrays")
    require(set(np.unique(fs)) <= {-2,-1,0,1,2} and set(np.unique(ps)) <= {-2,-1,0,1,2}, "Fit codes")
    require(set(np.unique(ls)) <= {-2,-1,0,1} and set(np.unique(decision)) <= {-1,0,1}, "LC/decision codes")
    require(np.all(np.isfinite(estimate[fs == 1])) and np.all(np.isfinite(pilot[ps == 1])), "Successful finite estimates")
    require(np.all(pilot[ps == 1] >= 0), "Nonnegative physical pilot")
    if free is not None:
        require(free.shape == (n,17) and np.all(np.isfinite(free[fs == 1])), "Full free physical vectors")
        equal(estimate, free[:,1:5]*1000)
    require(np.all(np.isfinite(lc[ls == 1])) and np.all(lc[ls == 1] >= 0), "Available LC")
    valid = (fs[:,None] == 1) & (ps[:,None] == 1) & (ls == 1)
    equal(decision >= 0, valid)
    equal(decision[valid], (estimate > lc)[valid])
    no_cov = ~arrays[family+"_free_covariance_available"]
    return dict(n_observations=n,full_free_parameters_checked=free is not None,n_valid=valid.sum(axis=0).tolist(),
        n_false=(decision == 1).sum(axis=0).tolist(),n_fail=(~valid).sum(axis=0).tolist(),
        negative_free_counts=((estimate < 0) & (fs[:,None] == 1)).sum(axis=0).tolist(),
        successful_free_covariance_unavailable=int(np.sum((fs == 1)&no_cov)),
        covariance_unavailable_decision=(valid&no_cov[:,None]).sum(axis=0).tolist(),
        free_status_counts={str(k):int(v) for k,v in Counter(fs.tolist()).items()},
        auxiliary_status_counts={str(k):int(v) for k,v in Counter(ps.tolist()).items()})


def qr_lc_checks(scenario_arrays, replicates, variant):
    S, b = scenario_arrays["physical_matrix"], scenario_arrays["b"]
    checks = []
    for identifier in FIXED_IDS:
        if identifier >= len(replicates["wls_LC"]):
            continue
        for family in FAMILIES:
            if replicates[family+"_auxiliary_status"][identifier] != 1:
                continue
            for col, target in enumerate([1,2,3,4]):
                if replicates[family+"_LC_status"][identifier,col] != 1:
                    continue
                z = replicates[family+"_auxiliary_physical"][identifier].copy()
                z[target] = 0
                mu = S @ z + b
                require(np.all(mu > 0), "Positive target-removed mean")
                A = S / np.sqrt(mu)[:,None]
                scales = np.linalg.norm(A,axis=0)
                _,R = np.linalg.qr(A/scales,mode="reduced")
                inv = solve_triangular(R,np.eye(17))
                independent = 1000*norm.ppf(.95)*np.linalg.norm(inv[target])/scales[target]
                saved = replicates[family+"_LC"][identifier,col]
                close(independent,saved,"Independent scaled-QR own-model LC",rtol=2e-10,atol=1e-12)
                checks.append(dict(variant=variant,replicate_id=identifier,method=family,isotope=ISOTOPES[col],
                    saved=float(saved),independent=float(independent),relative_difference=float((independent-saved)/saved)))
    return checks


def view_variant(arrays, variant):
    prefix = variant + "_"
    return {key[len(prefix):]: value for key,value in arrays.items()
            if key.startswith(prefix) and any(key.startswith(prefix+f+"_") for f in FAMILIES)}


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def compare_fields(row, expected, label):
    for key,value in expected.items():
        actual = float(row[key]) if row[key] else np.nan
        close(actual,value,label+"/"+key,rtol=2e-11,atol=2e-13)
    return len(expected)


def check_summary(root, arrays):
    rows = csv_rows(root/"summary.csv")
    require(len(rows) == 16, "Sixteen individual summary rows")
    lookup = {(r["variant"],r["method"],r["isotope"]):r for r in rows}
    require(len(lookup) == 16,"Unique variant/method/isotope rows")
    compared = 0
    for variant in ("A","B"):
        v = view_variant(arrays,variant)
        for family,label in FAMILIES.items():
            for j,isotope in enumerate(ISOTOPES):
                d=v[family+"_decision"][:,j]
                n,nv,k=len(d),int(np.sum(d>=0)),int(np.sum(d==1))
                lo,hi=cp(k,nv)
                estimates=v[family+"_concentrations"][:,j]
                estimates=estimates[(v[family+"_free_status"]==1)&np.isfinite(estimates)]
                lc=v[family+"_LC"][:,j]
                lc=lc[(v[family+"_LC_status"][:,j]==1)&np.isfinite(lc)]
                q=np.quantile(lc,[.025,.5,.975]) if len(lc) else [np.nan]*3
                expected=dict(n_generated=n,n_observations=n,n_new_generated=0,n_reused=n,
                    n_valid=nv,n_false=k,n_fail=n-nv,frequency=k/nv if nv else np.nan,
                    ci95_low=lo,ci95_high=hi,frequency_all_min=k/n,frequency_all_max=(k+n-nv)/n,
                    mean_estimate_mBq_m3=float(np.mean(estimates)) if len(estimates) else np.nan,
                    sd_estimate_mBq_m3=float(np.std(estimates,ddof=1)) if len(estimates)>1 else np.nan,
                    lc_q025_mBq_m3=q[0],lc_median_mBq_m3=q[1],lc_q975_mBq_m3=q[2],
                    n_available_estimate=len(estimates),n_available_lc=len(lc),
                    n_negative_estimate=int(np.sum(estimates<0)),
                    n_covariance_unavailable_decision=int(np.sum((d>=0)&~v[family+"_free_covariance_available"])))
                compared += compare_fields(lookup[variant,label,isotope],expected,variant+"/"+label+"/"+isotope)
    return compared


def check_paired(root, arrays):
    rows=csv_rows(root/"paired_comparison.csv")
    require(len(rows)==8,"Eight paired comparison rows")
    lookup={(r["method"],r["isotope"]):r for r in rows}
    require(len(lookup)==8,"Unique paired rows")
    A,B=view_variant(arrays,"A"),view_variant(arrays,"B")
    compared=0
    for family,label in FAMILIES.items():
        for j,isotope in enumerate(ISOTOPES):
            da,db=A[family+"_decision"][:,j],B[family+"_decision"][:,j]
            va,vb=da>=0,db>=0
            both=va&vb
            counts=np.bincount((2*da[both]+db[both]).astype(int),minlength=4)
            n00,n01,n10,n11=[int(v) for v in counts]
            common=int(both.sum())
            ka,kb=n10+n11,n01+n11
            ea,eb=A[family+"_concentrations"][:,j],B[family+"_concentrations"][:,j]
            la,lb=A[family+"_LC"][:,j],B[family+"_LC"][:,j]
            valid_est=(A[family+"_free_status"]==1)&(B[family+"_free_status"]==1)&np.isfinite(ea)&np.isfinite(eb)
            valid_lc=(A[family+"_LC_status"][:,j]==1)&(B[family+"_LC_status"][:,j]==1)&np.isfinite(la)&np.isfinite(lb)
            de,dl=eb[valid_est]-ea[valid_est],lb[valid_lc]-la[valid_lc]
            expected=dict(n_observations=len(da),n_common_valid_decisions=common,
                n_A_valid=int(va.sum()),n_B_valid=int(vb.sum()),
                n_A_only_valid=int(np.sum(va&~vb)),n_B_only_valid=int(np.sum(~va&vb)),n_neither_valid=int(np.sum(~va&~vb)),
                n00=n00,n01=n01,n10=n10,n11=n11,n_A_false_common=ka,n_B_false_common=kb,
                frequency_A_common=ka/common if common else np.nan,frequency_B_common=kb/common if common else np.nan,
                delta_frequency_percentage_points=100*(n01-n10)/common if common else np.nan,
                n_paired_estimates=len(de),mean_delta_estimate_B_minus_A_mBq_m3=float(np.mean(de)) if len(de) else np.nan,
                sd_delta_estimate_B_minus_A_mBq_m3=float(np.std(de,ddof=1)) if len(de)>1 else np.nan,
                n_paired_lc=len(dl),mean_delta_lc_B_minus_A_mBq_m3=float(np.mean(dl)) if len(dl) else np.nan,
                sd_delta_lc_B_minus_A_mBq_m3=float(np.std(dl,ddof=1)) if len(dl)>1 else np.nan)
            compared+=compare_fields(lookup[label,isotope],expected,"Paired/"+label+"/"+isotope)
    return compared


def check_diagnostics(values, catalogue, saved):
    require(saved["diagnostic_catalogue"]==catalogue,"NPZ and JSON catalogues agree")
    for family in FAMILIES:
        for phase in ("free","auxiliary"):
            prefix=family+"_"+phase
            codes=values[prefix+"_diagnostic_code"]
            require(np.all((codes>=0)&(codes<len(catalogue))),"Fit code bounds")
            counts=Counter(codes.tolist())
            messages,warnings,pywarnings=Counter(),Counter(),Counter()
            for code,amount in counts.items():
                diagnostic=catalogue[code]
                statuses=set(values[prefix+"_status"][codes==code].tolist())
                if "success" in diagnostic:
                    require(statuses <= ({1,2} if diagnostic["success"] else {0}),"Fit success/status agreement")
                elif "exception" in diagnostic:
                    require(statuses=={-1},"Exception/status agreement")
                messages[diagnostic.get("message",diagnostic.get("exception","No message"))]+=amount
                for warning in diagnostic.get("warnings",[]):warnings[warning]+=amount
                for warning in diagnostic.get("python_warnings",[]):pywarnings[warning]+=amount
            item=saved["fit_diagnostics"][family][phase]
            require(item["messages"]==dict(messages),"Optimizer message totals")
            require(item["warnings"]==dict(warnings),"Optimizer warning totals")
            require(item["python_warnings"]==dict(pywarnings),"Python warning totals")
            require(item["status_counts"]=={str(k):int(v) for k,v in Counter(values[prefix+"_status"].tolist()).items()},"Fit status totals")
        codes=values[family+"_LC_diagnostic_code"]
        require(np.all((codes>=0)&(codes<len(catalogue))),"LC diagnostic bounds")


def check_sources(root, arrays, scenarios, diagnostics, source_batches, source_A, full_free):
    n=100000
    records=[]
    B=view_variant(arrays,"B")
    catalogue=[json.loads(str(value)) for value in arrays["B_diagnostic_catalogue_json"]]
    catmap={json.dumps(value,sort_keys=True,ensure_ascii=False):i for i,value in enumerate(catalogue)}
    require(len(catmap)==len(catalogue),"Unique B diagnostic catalogue")
    errors=[]
    for chunk in range(100):
        selected=slice(chunk*1000,(chunk+1)*1000)
        candidates=[root/"batch_metadata_B"/f"batch_{chunk:03d}.json",
                    root/"batch_metadata"/f"batch_{chunk:03d}.json",root/"points"/f"batch_{chunk:03d}.json"]
        path=next((p for p in candidates if p.exists()),candidates[0])
        meta=read(path)
        require(meta["chunk"]==chunk and meta["variant"]=="B" and meta["n_reused"]==1000 and
                meta["n_generated"]==0,"B batch identities/no generated counts")
        equal(meta["seed"],arrays["seed_table"][chunk])
        require(meta["runner_lock_sha256"]==sha(root/"runner_lock.json"),"B batch runner hash")
        require(meta["source_replicates_sha256"]==SOURCE_A_SHA,"B original observation source")
        mapping=np.array([catmap[json.dumps(value,sort_keys=True,ensure_ascii=False)]
                          for value in meta["diagnostic_catalogue"]],dtype=int)
        errors.extend(dict(chunk=chunk,**e) for e in meta["errors"])
        if source_batches:
            require(full_free,"Full free supplement required for every raw source field")
            npz=root/"points"/f"batch_{chunk:03d}.npz"
            require(sha(npz)==meta["npz_sha256"],"B raw checkpoint hash")
            source=load(npz)
            for key,value in source.items():
                if key=="seed":equal(value,arrays["seed_table"][chunk])
                elif key in ("x","replicate_id","chunk","row_in_batch"):
                    equal(value,arrays[key][selected],"B shared rows bit-exact: "+key)
                elif value.ndim and value.shape[0]==1000:
                    if key.endswith("_diagnostic_code"):value=mapping[value]
                    equal(value,B[key][selected],"B fitted rows bit-exact: "+key)
                else:equal(value,arrays[key],"B invariant field: "+key)
        records.append(dict(chunk=chunk,npz_sha256=meta["npz_sha256"],metadata_sha256=sha(path),n_error_records=len(meta["errors"])))
    require(diagnostics["B"]["errors"]==errors,"Every B failure/recovery retained")
    source_A_checked=False
    if source_A is not None:
        path=source_A/"replicates.npz"
        require(sha(path)==SOURCE_A_SHA,"Original A file byte digest")
        original=load(path)
        for key,value in original.items():
            if any(key.startswith(f+"_") for f in FAMILIES):
                target="A_"+key
                if key.endswith("_free_physical") and not full_free:continue
                equal(value,arrays[target],"A saved values reused bit-exact: "+key)
            elif key in arrays:
                equal(value,arrays[key],"A shared values bit-exact: "+key)
        equal(original["diagnostic_catalogue_json"],arrays["A_diagnostic_catalogue_json"])
        equal(original["status_legends_json"],arrays["A_status_legends_json"])
        require(diagnostics["A"]["errors"]==read(source_A/"diagnostics.json")["errors"],"A failures unchanged")
        source_A_checked=True
    # These old failed rows must remain even in the portable main archive.
    equal(np.flatnonzero(arrays["A_poisson_auxiliary_status"]!=1),[46012,46549,82763])
    return records,source_A_checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT)
    parser.add_argument("--source-batches",action="store_true",help="Compare all original B checkpoint arrays; needs full-free supplement")
    parser.add_argument("--require-full-free",action="store_true",help="Fail if optional full_free_parameters.npz is absent")
    parser.add_argument("--source-A",type=Path,default=None,help="Original A study for bit-exact source comparison")
    parser.add_argument("--no-local-source-A",action="store_true",help="Exercise portable verification without local original A files")
    args=parser.parse_args()
    root=args.study.resolve()
    scenarios,scenario_arrays,observations,daughter=check_scenario(root)
    filename="replicates.npz"
    arrays=load(root/filename)
    supplement=root/"full_free_parameters.npz"
    full_free=supplement.exists()
    require(full_free or not (args.require_full_free or args.source_batches),"Full free supplement required")
    summary=read(root/"summary.json")
    require(summary["replicates_sha256"]==sha(root/filename),"Main merged data hash")
    require(str(arrays["full_free_parameters_sha256"].item())==summary["full_free_parameters_sha256"],
            "Main file references the exact full-free supplement")
    if full_free:
        require(summary["full_free_parameters_sha256"]==sha(supplement),"Full free supplement hash")
        for key,value in load(supplement).items():
            if key in arrays:equal(value,arrays[key],"Supplement shared metadata: "+key)
            else:arrays[key]=value
    for key in ("x","replicate_id","chunk","row_in_batch","seed_table"):
        equal(arrays[key],observations[key],"Exact reuse of saved A observations: "+key)
    equal(arrays["isotope_order"],ISOTOPES)
    require(str(arrays["concentration_units"].item())=="mBq/m3","Concentration units")
    require(int(arrays["newly_generated_observation_vectors"].item())==0,"No new observations")
    require(str(arrays["generation_model_variant"].item())=="A","A-generated observations")
    require(str(arrays["A_source_replicates_sha256"].item())==SOURCE_A_SHA,"Embedded A source SHA")
    for variant in ("A","B"):
        require(str(arrays[variant+"_scenario_sha256"].item())==sha(root/f"scenario_{variant}.json"),"Embedded scenario digest")
    require(str(arrays["B_runner_lock_sha256"].item())==sha(root/"runner_lock.json"),"Embedded B runner digest")
    diagnostics=read(root/"diagnostics.json")
    source_A=args.source_A
    if source_A is None and not args.no_local_source_A and (SOURCE_A/"replicates.npz").is_file():source_A=SOURCE_A
    records,A_checked=check_sources(root,arrays,scenarios,diagnostics,args.source_batches,source_A,full_free)
    methods={};qr=[]
    for variant in ("A","B"):
        values=view_variant(arrays,variant)
        catalog=[json.loads(str(v)) for v in arrays[variant+"_diagnostic_catalogue_json"]]
        check_diagnostics(values,catalog,diagnostics[variant])
        methods[variant]={family:check_method(family,values) for family in FAMILIES}
        qr.extend(qr_lc_checks(scenario_arrays[variant],values,variant))
    summary_count=check_summary(root,arrays)
    paired_count=check_paired(root,arrays)
    names=["scenario_A.json","scenario_B.json","scenario_arrays_A.npz","scenario_arrays_B.npz",
           "observations.npz","runner_lock.json",filename,"summary.csv","paired_comparison.csv",
           "summary.json","diagnostics.json","analyze.py"]
    if full_free:names.append("full_free_parameters.npz")
    result=dict(status="passed",verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(Path(__file__)),n_observations=100000,no_new_random_draws=True,no_fits=True,
        all_saved_observations_exact=True,all_B_source_arrays_compared=bool(args.source_batches),
        original_A_arrays_compared=A_checked,full_free_parameters_checked=full_free,
        source_batches_B=records,methods=methods,summary_numeric_fields_verified=summary_count,
        paired_numeric_fields_verified=paired_count,Xe133m_daughter_checks=daughter,
        independent_QR_LC_checks=qr,independent_QR_predetermined_ids=FIXED_IDS,
        files_sha256={name:sha(root/name) for name in names})
    (root/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","n_observations","full_free_parameters_checked",
        "original_A_arrays_compared","summary_numeric_fields_verified","paired_numeric_fields_verified","methods")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
