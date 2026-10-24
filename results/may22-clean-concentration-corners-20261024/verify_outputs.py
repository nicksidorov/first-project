"""Audit the saved XeMat corner-plot package without simulation or fitting.

This checker never evaluates model_covariance or any detection threshold.
The already saved 17-parameter covariance is checked by an information-matrix
residual, without inverse, solve, SVD, or a second reference estimate.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from unittest.mock import patch

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
from PIL import Image

ORDER = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
METHODS = {"ls": "wls", "poisson_ml": "poisson"}
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, name, rtol=2e-12, atol=2e-15):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)


def equal(actual, expected, name):
    assert np.array_equal(actual, expected, equal_nan=True), name


def csv_rows(root, name):
    with (root/name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(root, source, repo):
    scenario, stats, plot = [read(root/name) for name in
                            ("scenario.json", "statistics.json", "plot_metadata.json")]
    manifest = read(source/"manifest.json")
    dataset = manifest["datasets"]["may22"]
    with np.load(root/"estimates.npz", allow_pickle=False) as stored:
        a = {key: stored[key] for key in stored.files}
    input_sha = sha(root/"estimates.npz")
    assert input_sha == scenario["estimates_sha256"] == stats["estimates_sha256"] == plot["source_sha256"]
    assert a["isotopes"].tolist() == scenario["isotope_order"] == stats["isotope_order"] == plot["isotope_order"] == ORDER
    assert a["units"].item() == scenario["concentration_units"] == stats["concentration_units"] == plot["units"] == "mBq/m^3"
    assert a["covariance_units"].item() == stats["covariance_units"] == "(mBq/m^3)^2"
    assert a["source_stage"].item() == scenario["source_stage"] == "validation"
    assert a["physical_coordinate_label"].item() == scenario["physical_coordinates"] == "[a_V; gamma; zeta]"
    assert a["x"].shape == (100000, 21)
    equal(a["row_id"], np.arange(100000), "row identity")
    equal(a["batch_chunk"], np.repeat(np.arange(100), 1000), "chunk order")
    equal(a["row_in_batch"], np.tile(np.arange(1000), 100), "row within chunk")
    for name, digest in scenario["source_sha256"].items():
        assert sha(source/name) == digest, name
    for name, digest in scenario["production_sha256"].items():
        assert sha(repo/name) == digest, name
    for name, digest in read(source/"run_lock.json").items():
        assert sha(source/name) == digest, name

    # Exact byte provenance and equality to every saved free vector; no refits.
    batches = sorted(scenario["source_batches"], key=lambda record: record["chunk"])
    plan = sorted(read(source/"validation_plan.json"), key=lambda job: job["chunk"])
    assert len(batches) == len(plan) == 100
    for batch, job in zip(batches, plan):
        chunk = batch["chunk"]
        assert chunk == job["chunk"] and job["n"] == 1000
        assert job["target"] == 0 and job["factor"] == 0 and job["stage"] == "validation"
        for suffix in ("json", "npz"):
            assert sha(source/batch[suffix+"_file"]) == batch[suffix+"_sha256"]
        metadata = read(source/batch["json_file"])
        expected_seed = [2026102701, 0, 0, 0, 11, chunk]
        assert batch["seed"] == metadata["seed"] == expected_seed
        equal(a["seed_table"][chunk], expected_seed, "seed table")
        equal(metadata["truth"], dataset["truth0"], "batch truth")
        equal(metadata["mu"], dataset["mu0"], "batch mean")
        equal(metadata["mu"], a["mu0"], "saved mean")
        assert not [error for error in metadata["errors"] if error["phase"] == "free"]
        sl = slice(chunk*1000, (chunk+1)*1000)
        with np.load(source/batch["npz_file"], allow_pickle=False) as original:
            equal(a["x"][sl], original["x"], "shared original observations")
            for method, prefix in METHODS.items():
                equal(a[method][sl], original[prefix+"_free"][:, 1:5]*1000, "all signed xenon estimates")
                equal(a[method+"_full_native"][sl], original[prefix+"_free"], "full physical estimates")
                equal(a[method+"_success"][sl], original[prefix+"_free_success"], "fit status")
                equal(a[method+"_covariance_available"][sl], original[prefix+"_free_covariance_available"], "covariance diagnostic")

    # Verify the unchanged complete physical model by a transform identity.
    S, M, T = a["physical_matrix_native"], a["model_M_native"], a["model_T_V"]
    K = a["reference_covariance_full_native"]
    assert S.shape == M.shape == (21, 17) and K.shape == (17, 17)
    equal(M, dataset["model"]["M"], "M unchanged")
    equal(T, dataset["model"]["T_c"], "T_V unchanged")
    equal(a["truth_native"], dataset["truth0"], "truth unchanged")
    equal(a["truth_native"][:10], np.zeros(10), "zero sample and memory including radon")
    equal(a["truth_native"][10:], [522, 854, 494, 66, 64, 89, 97], "DETBK true means")
    equal(a["truth_concentration"], np.zeros(4), "true xenon concentrations")
    equal(a["model_b"], np.zeros(21), "no known additive detector background")
    Q = np.eye(17); Q[:5, :5] = T
    close(S@Q, M, "full physical-coordinate transformation", atol=2e-11)
    close(S@a["truth_native"]+a["model_b"], a["mu0"], "true mean spectrum")
    assert np.all(a["mu0"] > 0)
    metadata = dataset["metadata"]
    assert scenario["time_parameters_seconds"] == metadata["time_parameters"]
    assert scenario["effective_air_volume_m3"] == metadata["effective_air_volume_m3"]
    assert scenario["xenon_volume_cm3"] == metadata["xenon_volume_cm3"]
    assert scenario["efficiencies"] == manifest["efficiency_update"]
    assert scenario["calibration_matrices"] == dataset["calibration_matrices"]
    assert scenario["detector_id"] == metadata["input_detector_id"]
    assert scenario["systematic_errors_used"] is False
    assert scenario["reference_covariance"]["number_of_calls"] == 1
    for filename in ("prepare_data.py", "plot_estimates.py"):
        tree = ast.parse((root/filename).read_text())
        names = [node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))]
        assert not set(names) & {"fit", "nnls", "poisson", "compute_detection_limits"}, filename
        assert names.count("model_covariance") == (1 if filename == "prepare_data.py" else 0)
    # Residual of the saved covariance. No second inverse/solve/SVD is formed.
    information = (S.T/a["mu0"])@S
    scales = np.sqrt(np.diag(information))
    normalized_information = information/np.outer(scales, scales)
    normalized_K = K*np.outer(scales, scales)
    residual = float(np.max(np.abs(normalized_information@normalized_K-np.eye(17))))
    assert residual < 1e-10, residual
    close(K, K.T, "full covariance symmetry")
    reference = a["reference_covariance"]
    equal(reference, K[1:5, 1:5]*1e6, "xenon block after full inversion; covariance unit conversion")
    reference_sd = np.sqrt(np.diag(reference))
    reference_corr = reference/np.outer(reference_sd, reference_sd)
    close(a["reference_sd"], reference_sd, "reference SD")
    close(a["reference_correlation"], reference_corr, "reference correlation")
    close(stats["reference_covariance"], reference, "statistics reference covariance")
    close(stats["reference_sd"], reference_sd, "statistics reference SD")
    close(stats["reference_correlation"], reference_corr, "statistics reference correlation")

    independently = {}
    csv_checks = 0
    for method in METHODS:
        values, success = a[method], a[method+"_success"]
        assert values.shape == (100000, 4) and success.dtype == np.dtype(bool)
        valid = success & np.isfinite(values).all(axis=1)
        equal(valid, a[method+"_valid"], "validity mask")
        expected_status = np.full(len(values), "success", dtype="U32")
        expected_status[~success] = "fit_failure"
        expected_status[success & ~np.isfinite(values).all(axis=1)] = "nonfinite_estimate"
        assert np.array_equal(a[method+"_status"], expected_status)
        assert valid.sum() == 100000 and success.all()
        sample = values[valid]
        mean = np.mean(sample, axis=0)
        cov = np.cov(sample, rowvar=False, ddof=1)
        sd = np.std(sample, axis=0, ddof=1)
        corr = np.corrcoef(sample, rowvar=False)
        independent = dict(mean=mean, empirical_sd=sd, empirical_covariance=cov,
                           empirical_correlation=corr,
                           sd_ratio_to_reference=sd/reference_sd,
                           variance_ratio_to_reference=np.diag(cov)/np.diag(reference),
                           covariance_difference=cov-reference,
                           standardized_covariance_difference=(cov-reference)/np.outer(reference_sd, reference_sd),
                           correlation_difference=corr-reference_corr,
                           n_negative_by_isotope=np.count_nonzero(sample < 0, axis=0),
                           minimum=sample.min(axis=0), maximum=sample.max(axis=0))
        independently[method] = independent
        for key, value in independent.items():
            close(stats["methods"][method][key], value, "all full-sample statistics: "+key)
        for suffix in ("mean", "empirical_covariance", "empirical_correlation", "empirical_sd"):
            close(a[method+"_"+suffix], independent[suffix], "NPZ statistic")
        for key, value in (("mean", mean), ("sd", sd), ("correlation", corr)):
            close(plot["methods"][method][key], value, "full-sample plot statistic")
        assert plot["methods"][method]["n_valid"] == 100000
        unavailable = int((~a[method+"_covariance_available"]).sum())
        assert unavailable == (369 if method == "poisson_ml" else 0)
        assert stats["methods"][method]["n_retained_without_fit_covariance"] == unavailable
        assert stats["methods"][method]["n_fit_failures"] == stats["methods"][method]["n_excluded"] == 0

    def check_number(actual, expected, label):
        nonlocal csv_checks
        close(float(actual), float(expected), label)
        csv_checks += 1

    for method in [*METHODS, "reference"]:
        cov = reference if method == "reference" else independently[method]["empirical_covariance"]
        corr = reference_corr if method == "reference" else independently[method]["empirical_correlation"]
        for kind, matrix in (("covariance", cov), ("correlation", corr)):
            rows = csv_rows(root, f"{kind}_{method}.csv")
            assert len(rows) == 4 and [row["isotope"] for row in rows] == ORDER
            for i, row in enumerate(rows):
                for j, isotope in enumerate(ORDER):
                    check_number(row[isotope], matrix[i, j], "matrix CSV")
    rows = csv_rows(root, "summary_statistics.csv")
    assert len(rows) == 8
    for row in rows:
        method = row["method"]; j = ORDER.index(row["isotope"]); st = independently[method]
        assert row["concentration_units"] == "mBq/m^3"
        expected = dict(n_attempted=100000, n_valid=100000, n_fit_failures=0,
                        n_success_nonfinite=0, true_concentration=0,
                        empirical_mean=st["mean"][j], empirical_sd_ddof1=st["empirical_sd"][j],
                        reference_sd=reference_sd[j], empirical_sd_over_reference_sd=st["sd_ratio_to_reference"][j],
                        empirical_variance_over_reference_variance=st["variance_ratio_to_reference"][j],
                        n_negative=st["n_negative_by_isotope"][j], minimum=st["minimum"][j], maximum=st["maximum"][j])
        for key, value in expected.items(): check_number(row[key], value, "summary CSV: "+key)
    rows = csv_rows(root, "calculation_status.csv"); assert len(rows) == 2
    for row in rows:
        unavailable = 369 if row["method"] == "poisson_ml" else 0
        expected = dict(n_attempted=100000, n_fit_success=100000, n_valid=100000,
                        n_fit_failures=0, n_success_nonfinite=0,
                        n_fit_covariance_unavailable=unavailable, n_retained_without_fit_covariance=unavailable)
        for key, value in expected.items(): check_number(row[key], value, "status CSV")
    rows = csv_rows(root, "covariance_comparison.csv"); assert len(rows) == 32
    for row in rows:
        st = independently[row["method"]];i=ORDER.index(row["isotope_i"]);j=ORDER.index(row["isotope_j"])
        assert row["covariance_units"] == "(mBq/m^3)^2"
        expected = dict(empirical_covariance=st["empirical_covariance"][i,j], reference_covariance=reference[i,j],
                        covariance_difference=st["covariance_difference"][i,j],
                        difference_over_reference_sd_i_sd_j=st["standardized_covariance_difference"][i,j],
                        empirical_correlation=st["empirical_correlation"][i,j], reference_correlation=reference_corr[i,j],
                        correlation_difference=st["correlation_difference"][i,j])
        for key, value in expected.items(): check_number(row[key], value, "comparison CSV")

    indices = a["scatter_indices"]
    assert indices.shape == (5000,) and len(np.unique(indices)) == 5000
    assert np.all((indices >= 0) & (indices < 100000))
    assert a["scatter_seed"].item() == scenario["scatter_seed"] == 2026102401
    equal(indices, np.sort(np.random.default_rng(2026102401).choice(np.arange(100000), 5000, replace=False)), "display-only subset reproducibility")
    assert hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() == plot["scatter_indices_sha256"]
    assert plot["scatter_n"] == 5000 and plot["scatter_indices_shared"]
    limits=np.asarray(plot["concentration_axis_limits"]);edges=np.asarray(plot["histogram_bin_edges"])
    density_limits=np.asarray(plot["histogram_density_limits"])
    assert limits.shape == (4,2) and edges.shape == (4,81)
    pooled=np.vstack([a[method] for method in METHODS]);low=pooled.min(axis=0);high=pooled.max(axis=0)
    expected_limits=np.column_stack([low-.035*(high-low), high+.035*(high-low)])
    close(limits, expected_limits, "common complete-range axes")
    for j in range(4):
        close(edges[j], np.linspace(*limits[j],81), "shared bins")
        assert edges[j,0] <= low[j] < 0 < high[j] <= edges[j,-1]
        peak=0
        for method in METHODS:
            counts,_=np.histogram(a[method][:,j],bins=edges[j])
            assert counts.sum() == 100000, "all data present in each histogram"
            peak=max(peak, np.max(counts/(100000*np.diff(edges[j]))))
        close(density_limits[j], [0,1.16*peak], "common histogram density axes")

    # Inspect the actual plotting function's artists in memory; suppress file writes.
    spec=importlib.util.spec_from_file_location("audited_plot",root/"plot_estimates.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import matplotlib.figure
    import matplotlib.colors
    from matplotlib.collections import PathCollection
    inspections=[]
    for method in METHODS:
        def inspect_figure(figure, filename, **kwargs):
            assert kwargs["dpi"] == 300
            assert len(figure.axes) == 16
            for i in range(4):
                for j in range(4):
                    ax=figure.axes[4*i+j]
                    assert ax.get_visible() == (j <= i)
                    if j>i: continue
                    close(ax.get_xlim(),limits[j],"artist concentration x limits")
                    if i==j:
                        close(ax.get_ylim(),density_limits[j],"artist histogram y limits")
                        assert ax.get_title()==module.LABELS[j]
                        assert len(ax.patches)==80
                        density,_=np.histogram(a[method][:,j],bins=edges[j],density=True)
                        close([bar.get_height() for bar in ax.patches],density,"artist histogram uses full sample")
                        assert len(ax.lines)==2
                        close(ax.lines[0].get_xdata(),[0,0],"true zero line")
                        close(ax.lines[1].get_xdata(),[independently[method]["mean"][j]]*2,"empirical mean line")
                    else:
                        close(ax.get_ylim(),limits[i],"artist concentration y limits")
                        scatter=[artist for artist in ax.collections if isinstance(artist,PathCollection)]
                        assert len(scatter)==1 and scatter[0].get_rasterized()
                        close(scatter[0].get_offsets(),a[method][indices][:,[j,i]],"artist common scatter points",rtol=0,atol=0)
                        expected_r=f"$r = {independently[method]['empirical_correlation'][i,j]:.3f}$"
                        assert any(text.get_text()==expected_r for text in ax.texts)
            inspections.append((method,Path(filename).suffix))
        actual_stats={m:{"n_valid":100000,"mean":independently[m]["mean"],"correlation":independently[m]["empirical_correlation"]} for m in METHODS}
        with patch.object(matplotlib.figure.Figure,"savefig",inspect_figure):
            module.draw(method,a,{m:a[m] for m in METHODS},actual_stats,indices,limits,edges,density_limits,
                        root,12.4,13.0,3.4,.22,300)
    assert len(inspections)==4
    assert plot["colors"] == {"ls":"#1f77b4","poisson_ml":"#ff7f0e"}
    image_details={}
    for method in METHODS:
        path=root/f"corner_{method}.png"
        with Image.open(path) as im:
            assert im.size == (3720,3720)
            dpi=im.info["dpi"];close(dpi,[300,300],"PNG DPI",rtol=0,atol=.001)
            image_details[method]={"pixels":list(im.size),"dpi":list(dpi)}
        pdf=(root/f"corner_{method}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page\b",pdf)) == 1
        assert b"/Subtype /Image" in pdf, "PDF scatter rasterization"

    # Check the concise descriptions, without editing the manuscript or documents.
    readme=(root/"README.md").read_text()
    for j,isotope in enumerate(ORDER):
        fields=[part.strip() for line in readme.splitlines() if line.startswith("| "+isotope+" |") for part in line.split("|")[2:-1]]
        expected=[independently["ls"]["mean"][j],independently["ls"]["empirical_sd"][j],
                  independently["poisson_ml"]["mean"][j],independently["poisson_ml"]["empirical_sd"][j],reference_sd[j]]
        assert fields == [f"{value:.6f}" for value in expected]
    for method in METHODS:
        r=independently[method]["empirical_correlation"][1,2]
        assert f"{r:.5f}".replace("-","−").replace(".",",") in readme
        maximum=100*np.max(np.abs(independently[method]["sd_ratio_to_reference"]-1))
        assert f"{maximum:.2f}".replace(".",",")+"%" in readme
        assert np.argmax(np.abs(independently[method]["sd_ratio_to_reference"]-1)) == 1
    latex=(root/"figures.tex").read_text()
    assert latex.count(r"\begin{figure*}")==2 and latex.count(r"\caption{")==2
    assert "corner_ls.pdf" in latex and "corner_poisson_ml.pdf" in latex
    assert r"100\,000" in latex and r"5\,000" in latex
    assert "radon" in latex and "Negative estimates and outliers are retained" in latex
    checked_files=["estimates.npz","statistics.json","scenario.json","scenario.txt","plot_metadata.json",
                   "prepare_data.py","plot_estimates.py","README.md","figures.tex","verify_outputs.py"]
    checked_files += [path.name for path in sorted(root.glob("*.csv"))]
    checked_files += [f"corner_{method}.{ext}" for method in METHODS for ext in ("png","pdf")]
    return {
        "status":"passed","checked_at_utc":datetime.now(timezone.utc).isoformat(),
        "n_source_batches":100,"n_shared_observations":100000,"n_free_xenon_values_verified":800000,
        "n_valid":{"ls":100000,"poisson_ml":100000},"fit_failures":{"ls":0,"poisson_ml":0},
        "poisson_point_estimates_retained_without_fit_covariance":369,
        "csv_numeric_values_checked":csv_checks,"covariance_divisor":"N_valid-1",
        "source_arrays_exactly_equal":True,"negative_estimates_and_extrema_retained":True,
        "reference_covariance_recomputed":False,"new_fits_or_observations":False,
        "reference_full_information_scaled_residual_maxabs":residual,
        "reference_check":"Saved full17x17 covariance satisfies normalized information times covariance equals identity; Xe block and x1e6 conversion verified without second inverse/SVD/solve.",
        "scatter_count":5000,"scatter_shared":True,"histograms_include_all_valid_rows":True,
        "plot_statistics_from_full_samples":True,"common_axes_and_bins":True,
        "plot_artists_inspected_without_file_writes":True,"png_properties":image_details,
        "readme_numbers_and_latex_scope":"checked",
        "artifact_sha256":{name:sha(root/name) for name in checked_files},
        "source_manifest_sha256":sha(source/"manifest.json"),
        "source_validation_plan_sha256":sha(source/"validation_plan.json"),
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--source",type=Path,default=Path("/workspace/scratch/may22-clean-mdc-100k-20261023"))
    parser.add_argument("--repo",type=Path,default=Path(os.environ.get("XEMAT_REPO","/workspace/first-project")))
    args=parser.parse_args()
    result=verify(args.root.resolve(),args.source.resolve(),args.repo.resolve())
    (args.root/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key!="artifact_sha256"},ensure_ascii=False,indent=2))
