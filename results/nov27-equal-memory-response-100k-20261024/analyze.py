"""Compare processing models A/B on the same saved strong-memory observations.

No observations, fits, LC values, powers or calibrated thresholds are generated.
All rows and returned numerical values are retained, including failed fits.
The portable --from-merged mode rebuilds tables and the report from saved arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
FAMILIES = {"wls": "LS", "poisson": "Poisson ML"}
ISOTOPES = ["Xe135", "Xe133", "Xe131m", "Xe133m"]
VARIANTS = {"A": "Different base responses", "B": "Equal base responses"}
SHARED_ROWS = ("x", "replicate_id", "chunk", "row_in_batch")
SHARED_CONSTANTS = ("isotope_order", "concentration_units", "physical_coordinate_label",
    "physical_parameter_names", "physical_parameter_units", "observation_order",
    "observation_units", "seed_table")
SUMMARY_FIELDS = ["variant", "method", "isotope", "n_generated", "n_observations",
    "n_new_generated", "n_reused", "n_valid", "n_false", "frequency", "ci95_low",
    "ci95_high", "n_fail", "frequency_all_min", "frequency_all_max",
    "mean_estimate_mBq_m3", "sd_estimate_mBq_m3", "lc_median_mBq_m3",
    "lc_q025_mBq_m3", "lc_q975_mBq_m3", "n_available_estimate", "n_available_lc",
    "n_negative_estimate", "n_covariance_unavailable_decision"]
PAIRED_FIELDS = ["method", "isotope", "n_observations", "n_common_valid_decisions",
    "n_A_valid", "n_B_valid", "n_A_only_valid", "n_B_only_valid", "n_neither_valid",
    "n00", "n01", "n10", "n11", "n_A_false_common", "n_B_false_common",
    "frequency_A_common", "frequency_B_common", "delta_frequency_percentage_points",
    "n_paired_estimates", "mean_delta_estimate_B_minus_A_mBq_m3",
    "sd_delta_estimate_B_minus_A_mBq_m3", "n_paired_lc",
    "mean_delta_lc_B_minus_A_mBq_m3", "sd_delta_lc_B_minus_A_mBq_m3"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mean(values):
    return float(np.mean(values)) if len(values) else None


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def fields_for_family(archive):
    return [key for key in archive if any(key.startswith(family + "_") for family in FAMILIES)]


def load_baseline(root, scenario):
    source = Path(os.environ.get("XEMAT_BASELINE_A", scenario["source_baseline_A"]["directory"]))
    path = source / "replicates.npz"
    require(sha(path) == scenario["source_replicates_sha256"], "Baseline A estimates changed")
    require(sha(source / "scenario.json") == sha(root / "scenario_A.json"), "Baseline A scenario changed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    diagnostics = read(source / "diagnostics.json")
    summary = read(source / "summary.json")
    require(summary["replicates_sha256"] == sha(path), "Baseline A provenance differs")
    np.testing.assert_array_equal(arrays["replicate_id"], np.arange(scenario["n"]))
    require(arrays["isotope_order"].tolist() == ISOTOPES, "Baseline isotope order")
    require(arrays["concentration_units"].item() == "mBq/m3", "Baseline concentration units")
    provenance = dict(directory=str(source), published_commit=scenario["source_baseline_A"]["published_commit"],
        replicates_sha256=sha(path), summary_sha256=sha(source / "summary.json"),
        diagnostics_sha256=sha(source / "diagnostics.json"), source_batches=summary["source_batches"])
    return arrays, diagnostics, provenance


def collect_B(root, scenario, arrays_A):
    """Concatenate every B checkpoint and map its diagnostic codes globally."""
    n, size, batches = int(scenario["n"]), int(scenario["batch_size"]), int(scenario["n_batches"])
    require((n, size, batches) == (100000, 1000, 100), "Unexpected observation plan")
    lock = read(root / "runner_lock.json")
    for key, name in (("study_sha256", "study.py"), ("scenario_sha256", "scenario_B.json"),
                      ("scenario_arrays_sha256", "scenario_arrays_B.npz"),
                      ("observations_sha256", "observations.npz")):
        require(lock[key] == sha(root / name), "Runner input changed: " + name)
    require(lock["source_replicates_sha256"] == scenario["source_replicates_sha256"], "A input hash differs")
    with np.load(root / "observations.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["x"], arrays_A["x"])
        np.testing.assert_array_equal(archive["replicate_id"], arrays_A["replicate_id"])
    collected, catalogue, lookup, errors, provenance = {}, [], {}, [], []
    expected_keys, legends = None, None
    for chunk in range(batches):
        raw_path = root / "points" / f"batch_{chunk:03d}.npz"
        meta_path = raw_path.with_suffix(".json")
        meta = read(meta_path)
        seed = [int(scenario["master_seed"]), chunk]
        require(meta["variant"] == "B" and meta["chunk"] == chunk, "B batch identity")
        require(meta["n_observations"] == meta["n_reused"] == size and meta["n_generated"] == 0,
                "B batch must reuse existing observations")
        require(meta["seed"] == seed, "Original seed provenance mismatch")
        require(meta["runner_lock_sha256"] == sha(root / "runner_lock.json"), "B checkpoint lock differs")
        require(meta["npz_sha256"] == sha(raw_path), "B checkpoint NPZ changed")
        current_legends = {key: meta[key] for key in ("fit_status_legend", "lc_status_legend", "decision_legend")}
        if legends is None:
            legends = current_legends
        require(current_legends == legends, "B status definitions changed")
        mapping = []
        for diagnostic in meta["diagnostic_catalogue"]:
            encoded = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if encoded not in lookup:
                lookup[encoded] = len(catalogue)
                catalogue.append(diagnostic)
            mapping.append(lookup[encoded])
        mapping = np.asarray(mapping, dtype=np.int32)
        section = slice(chunk * size, (chunk + 1) * size)
        with np.load(raw_path, allow_pickle=False) as raw:
            if expected_keys is None:
                expected_keys = set(raw.files)
            require(set(raw.files) == expected_keys, "B checkpoint schema changed")
            for key in SHARED_ROWS:
                np.testing.assert_array_equal(raw[key], arrays_A[key][section])
            np.testing.assert_array_equal(raw["seed"], seed)
            for key in ("isotope_order", "concentration_units", "physical_coordinate_label"):
                np.testing.assert_array_equal(raw[key], arrays_A[key])
            for key in fields_for_family(raw.files):
                values = raw[key].copy()
                require(values.shape[0] == size, "Wrong B row count: " + key)
                if key.endswith("_diagnostic_code"):
                    require(np.all((values >= 0) & (values < len(mapping))), "B diagnostic code out of range")
                    values = mapping[values]
                collected.setdefault(key, []).append(values)
        errors.extend({"chunk": chunk, **item} for item in meta["errors"])
        provenance.append(dict(chunk=chunk, n_observations=size, n_generated=0, original_seed=seed,
            json_file=f"points/batch_{chunk:03d}.json", json_sha256=sha(meta_path),
            npz_file=f"points/batch_{chunk:03d}.npz", npz_sha256=meta["npz_sha256"],
            elapsed_seconds=meta["elapsed_seconds"]))
    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    return arrays, catalogue, legends, errors, provenance


def collect(root, scenario):
    A, diagnostics_A, provenance_A = load_baseline(root, scenario)
    B, catalogue_B, legends_B, errors_B, provenance_B = collect_B(root, scenario, A)
    catalogue_A = [json.loads(item) for item in A["diagnostic_catalogue_json"].tolist()]
    require(catalogue_A == diagnostics_A["diagnostic_catalogue"], "A diagnostic catalogue mismatch")
    arrays = {key: A[key].copy() for key in SHARED_ROWS + SHARED_CONSTANTS}
    for key in fields_for_family(A):
        arrays["A_" + key] = A[key]
    for key, values in B.items():
        arrays["B_" + key] = values
    arrays.update(variant_order=np.array(["A", "B"]),
        A_diagnostic_catalogue_json=A["diagnostic_catalogue_json"],
        B_diagnostic_catalogue_json=np.asarray([json.dumps(item, ensure_ascii=False, sort_keys=True) for item in catalogue_B]),
        A_status_legends_json=A["status_legends_json"],
        B_status_legends_json=np.asarray(json.dumps(legends_B, ensure_ascii=False, sort_keys=True)),
        A_source_replicates_sha256=np.asarray(scenario["source_replicates_sha256"]),
        A_scenario_sha256=np.asarray(sha(root / "scenario_A.json")),
        B_scenario_sha256=np.asarray(sha(root / "scenario_B.json")),
        A_runner_lock_sha256=A["runner_lock_sha256"],
        B_runner_lock_sha256=np.asarray(sha(root / "runner_lock.json")),
        observations_sha256=np.asarray(sha(root / "observations.npz")),
        newly_generated_observation_vectors=np.array(0),
        generation_model_variant=np.asarray("A"))
    diagnostics = {"A": dict(diagnostic_catalogue=catalogue_A,
        errors=diagnostics_A["errors"], status_legends=json.loads(A["status_legends_json"].item())),
        "B": dict(diagnostic_catalogue=catalogue_B, errors=errors_B, status_legends=legends_B)}
    return arrays, diagnostics, {"A": provenance_A, "B": provenance_B}


def view_variant(arrays, variant):
    prefix = variant + "_"
    return {key[len(prefix):]: value for key, value in arrays.items()
        if key.startswith(prefix) and any(key.startswith(prefix + family + "_") for family in FAMILIES)}


def diagnostic_counts(values, catalogue):
    output = {}
    for family in FAMILIES:
        output[family] = {}
        for phase in ("free", "auxiliary"):
            prefix = family + "_" + phase
            codes, amounts = np.unique(values[prefix + "_diagnostic_code"], return_counts=True)
            entries = [dict(code=int(code), n=int(amount), diagnostic=catalogue[int(code)])
                for code, amount in zip(codes, amounts)]
            messages, warnings, pywarnings = Counter(), Counter(), Counter()
            for entry in entries:
                item, amount = entry["diagnostic"], entry["n"]
                messages[item.get("message", item.get("exception", "No message"))] += amount
                for message in item.get("warnings", []):
                    warnings[message] += amount
                for message in item.get("python_warnings", []):
                    pywarnings[message] += amount
            output[family][phase] = dict(status_counts={str(k): int(v) for k, v in
                Counter(values[prefix + "_status"].tolist()).items()}, diagnostic_counts=entries,
                messages=dict(messages), warnings=dict(warnings), python_warnings=dict(pywarnings))
    return output


def summarize_variant(values, variant, n):
    rows, methods = [], {}
    for family, label in FAMILIES.items():
        estimates, lc = values[family + "_concentrations"], values[family + "_LC"]
        decisions = values[family + "_decision"]
        free_ok, aux_ok = values[family + "_free_status"] == 1, values[family + "_auxiliary_status"] == 1
        lc_ok = (values[family + "_LC_status"] == 1) & np.isfinite(lc) & (lc >= 0)
        valid = free_ok[:, None] & aux_ok[:, None] & lc_ok & np.isfinite(estimates)
        require(np.array_equal(decisions >= 0, valid), variant + " decision availability differs")
        require(np.all(np.isin(decisions, [-1, 0, 1])), "Unknown decision status")
        np.testing.assert_array_equal(decisions[valid], (estimates[valid] > lc[valid]).astype(np.int8))
        # Full free vectors are a supplementary file. Main arrays alone suffice
        # to rebuild the requested tables, with availability controlled by status.
        if family + "_free_physical" in values:
            np.testing.assert_equal(estimates, values[family + "_free_physical"][:, 1:5] * 1000)
        missing_covariance = ~values[family + "_free_covariance_available"]
        methods[family] = dict(label=label, n_observations=n,
            n_successful_free=int(free_ok.sum()), n_successful_auxiliary=int(aux_ok.sum()),
            n_successful_free_covariance_unavailable=int(np.sum(free_ok & missing_covariance)),
            n_covariance_unavailable_any_decision=int(np.sum(missing_covariance & valid.any(axis=1))),
            n_covariance_unavailable_all_decisions=int(np.sum(missing_covariance & valid.all(axis=1))),
            n_covariance_unavailable_decision_by_isotope=np.sum(missing_covariance[:, None] & valid, axis=0).tolist(),
            n_any_failed_decision=int(np.sum(~valid.all(axis=1))),
            n_all_four_available_decisions=int(np.sum(valid.all(axis=1))))
        for j, isotope in enumerate(ISOTOPES):
            count = int(np.sum(decisions[:, j] == 1))
            nvalid = int(valid[:, j].sum())
            ci = binomtest(count, nvalid).proportion_ci(.95, method="exact") if nvalid else None
            a = estimates[free_ok & np.isfinite(estimates[:, j]), j]
            thresholds = lc[lc_ok[:, j], j]
            q = np.quantile(thresholds, [.025, .5, .975], method="linear") if len(thresholds) else [None] * 3
            rows.append(dict(variant=variant, method=label, isotope=isotope,
                n_generated=n, n_observations=n, n_new_generated=0, n_reused=n,
                n_valid=nvalid, n_false=count, frequency=count/nvalid if nvalid else None,
                ci95_low=float(ci.low) if ci else None, ci95_high=float(ci.high) if ci else None,
                n_fail=n-nvalid, frequency_all_min=count/n, frequency_all_max=(count+n-nvalid)/n,
                mean_estimate_mBq_m3=mean(a), sd_estimate_mBq_m3=sd(a),
                lc_median_mBq_m3=float(q[1]) if len(thresholds) else None,
                lc_q025_mBq_m3=float(q[0]) if len(thresholds) else None,
                lc_q975_mBq_m3=float(q[2]) if len(thresholds) else None,
                n_available_estimate=len(a), n_available_lc=len(thresholds),
                n_negative_estimate=int(np.sum(a < 0)),
                n_covariance_unavailable_decision=int(np.sum(missing_covariance & valid[:, j]))))
    return rows, methods


def paired_summary(arrays):
    rows = []
    for family, label in FAMILIES.items():
        A, B = view_variant(arrays, "A"), view_variant(arrays, "B")
        da, db = A[family + "_decision"], B[family + "_decision"]
        ea, eb = A[family + "_concentrations"], B[family + "_concentrations"]
        la, lb = A[family + "_LC"], B[family + "_LC"]
        free_a, free_b = A[family + "_free_status"] == 1, B[family + "_free_status"] == 1
        for j, isotope in enumerate(ISOTOPES):
            va, vb = da[:, j] >= 0, db[:, j] >= 0
            common = va & vb
            av, bv = da[common, j], db[common, j]
            ncommon = int(common.sum())
            cells = {f"n{i}{k}": int(np.sum((av == i) & (bv == k))) for i in (0, 1) for k in (0, 1)}
            require(sum(cells.values()) == ncommon, "Paired contingency table does not sum")
            common_est = free_a & free_b & np.isfinite(ea[:, j]) & np.isfinite(eb[:, j])
            common_lc = ((A[family + "_LC_status"][:, j] == 1) & (B[family + "_LC_status"][:, j] == 1)
                & np.isfinite(la[:, j]) & np.isfinite(lb[:, j]))
            delta_est = eb[common_est, j] - ea[common_est, j]
            delta_lc = lb[common_lc, j] - la[common_lc, j]
            ka, kb = int(av.sum()), int(bv.sum())
            rows.append(dict(method=label, isotope=isotope, n_observations=len(da),
                n_common_valid_decisions=ncommon, n_A_valid=int(va.sum()), n_B_valid=int(vb.sum()),
                n_A_only_valid=int(np.sum(va & ~vb)), n_B_only_valid=int(np.sum(~va & vb)),
                n_neither_valid=int(np.sum(~va & ~vb)), **cells,
                n_A_false_common=ka, n_B_false_common=kb,
                frequency_A_common=ka/ncommon if ncommon else None,
                frequency_B_common=kb/ncommon if ncommon else None,
                delta_frequency_percentage_points=100*(kb-ka)/ncommon if ncommon else None,
                n_paired_estimates=int(common_est.sum()), mean_delta_estimate_B_minus_A_mBq_m3=mean(delta_est),
                sd_delta_estimate_B_minus_A_mBq_m3=sd(delta_est), n_paired_lc=int(common_lc.sum()),
                mean_delta_lc_B_minus_A_mBq_m3=mean(delta_lc), sd_delta_lc_B_minus_A_mBq_m3=sd(delta_lc)))
    return rows


def csv_write(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_table(root, rows):
    """Publication table from the already computed CSV rows; no new analysis."""
    labels = {"Xe135": r"$^{135}\mathrm{Xe}$", "Xe133": r"$^{133}\mathrm{Xe}$",
              "Xe131m": r"$^{131\mathrm{m}}\mathrm{Xe}$", "Xe133m": r"$^{133\mathrm{m}}\mathrm{Xe}$"}
    lines = [r"% Standalone insertion fragment; no changes to the main manuscript.",
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{False-detection frequencies under the original response model A and the equal-base-response processing model B. Rates and exact marginal 95\% Clopper--Pearson intervals are calculated separately among the available decisions of each variant.}",
        r"\label{tab:xemat-equal-memory-responses}", r"\begin{tabular}{lllrr}", r"\hline",
        r"Variant & Method & Isotope & False / available & Rate (\%) [95\% CI] \\", r"\hline"]
    for row in rows:
        rate = f"{100*row['frequency']:.3f} [{100*row['ci95_low']:.3f}, {100*row['ci95_high']:.3f}]"
        lines.append(f"{row['variant']} & {row['method']} & {labels[row['isotope']]} & "
            f"{row['n_false']} / {row['n_valid']} & {rate} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{0.98\textwidth}\footnotesize",
        r"The same 100\,000 observation vectors generated with A are used for both variants. Each variant uses its own auxiliary fit and $L_C$, without threshold rescaling. All four true fresh-xenon concentrations are zero. Failed decisions are retained as unavailable and are not counted as non-detections.",
        r"\end{minipage}", r"\end{table*}"])
    (root / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value, precision=6):
    return "—" if value is None else f"{value:.{precision}g}"


def pct(value):
    return "—" if value is None else f"{100*value:.3f}%"


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(root, scenario_A, scenario_B, arrays, rows, pairs, diagnostics, summary):
    q = scenario_A["truth_construction"]["original_fit_diagnostics"]
    quality = q["fit_quality"]
    lookup = {(r["variant"], r["method"], r["isotope"]): r for r in rows}
    am = lookup["A", "LS", "Xe133m"]
    bm = lookup["B", "LS", "Xe133m"]
    ap = lookup["A", "Poisson ML", "Xe133m"]
    bp = lookup["B", "Poisson ML", "Xe133m"]
    lines = ["# XeMat: последствия одинаковых базовых откликов пробы и памяти", "",
        "Парное сравнение обработки **одних и тех же 100 000 сохранённых наблюдений** "
        "сценария с сильной газовой памятью, детектор 221. Новые наблюдения не моделировались. "
        "Наблюдения были сгенерированы в предыдущем эксперименте моделью А с различными "
        "откликами пробы и памяти. Все четыре истинные свежие концентрации ксенона равны нулю.", "",
        "**Главный результат для этого сценария:** при упрощении откликов частота ложных "
        f"обнаружений Xe133m выросла с {pct(am['frequency'])} до {pct(bm['frequency'])} "
        f"для МНК и с {pct(ap['frequency'])} до {pct(bp['frequency'])} для ММП. "
        "Медиана собственного LC Xe133m при этом изменилась мало: "
        f"{am['lc_median_mBq_m3']:.6f} → {bm['lc_median_mBq_m3']:.6f} mBq/m³ для МНК "
        f"и {ap['lc_median_mBq_m3']:.6f} → {bp['lc_median_mBq_m3']:.6f} mBq/m³ для ММП. "
        "Изменения зависят от изотопа и метода: например, для Xe135 частоты заметно "
        "снизились, а для Xe133 направление изменения различается у двух методов. "
        "Снижение ложных обнаружений само по себе не устанавливает улучшение обнаружения: "
        "мощность в этом эксперименте не рассчитывалась.", "",
        "## Модели А и Б", "",
        "**А:** исходные отклики. Свободные и вспомогательные оценки, LC, решения и статусы "
        "взяты из сохранённого предыдущего эксперимента без повторной подгонки.", "",
        "**Б:** для четырёх ксеноновых столбцов базовой матрицы памяти выполнено "
        "`R_g0_B[:, 1:5] = R_c0[:, 1:5]`. Радоновый столбец сохранён. "
        "После замены снова собраны `R_g1_B = R_g0_B @ C_s_g` и "
        "`R_g2_B = R_g0_B @ C_g_g`. Оператор переноса памяти `F`, временные поправки, "
        "чувствительности, branching, матрица свежей пробы и преобразование к концентрациям сохранены.", "",
        "В частности, столбец Xe133m после поправки содержит базовый материнский "
        "столбец плюс коэффициент врастания, умноженный на базовый столбец Xe133. "
        "Поэтому замена базы влияет и на дочерний вклад. Коэффициенты врастания "
        "для длительностей SAMPLE/GASBK и перенос дочери через F остаются прежними. "
        "Повторная нормировка столбцов после поправок не выполнялась.", "",
        "В Б для каждого наблюдения заново выполнены свободная оценка и соответствующая "
        "вспомогательная неотрицательная подгонка. Одна вспомогательная оценка каждого "
        "метода использована для четырёх проверок. При расчёте собственного LC Б "
        "поочерёдно удалён только оцениваемый свежий вклад проверяемого изотопа; "
        "остальные оценённые координаты сохранены. Дополнительной подгонки под нулевой "
        "гипотезой нет. Обнаружение — строго `free concentration > LC`, без множителя c.", "",
        "Веса МНК/NNLS: `W = diag(1 / where(x == 0, 1, x))`, заново из каждого "
        "наблюдения. Исходные нулевые отсчёты не меняются. Пуассоновский критерий "
        "использует все исходные отсчёты. Решатели и настройки `max_iter=2000`, "
        "`tol=1e-10` неизменны. LC получен существующей процедурой полной модели "
        "из 21 наблюдения и 17 параметров, включая память и DETBK.", "",
        "## Сценарий и калибровка", "",
        "Фиксированная истина, исходная неотрицательная оценка и генерирующее μ₀ "
        "унаследованы из А без изменения: только свежие Xe равны нулю, "
        "свежий радон, память и средние DETBK сохранены. Значение модели Б на этом "
        "векторе не используется для генерации. Систематические ошибки и "
        "неопределённость выбранной фиксированной истины не разыгрываются.", "",
        "| Роль | Файл | Начало измерения | Реальное время, s | Живое время, s |",
        "|---|---|---|---:|---:|"]
    change_lines = ["## Точные изменения базовой матрицы памяти", "",
        "| ROI | Столбец | А | Б |", "|---:|---|---:|---:|"]
    for item in scenario_B["modification"]["changed_base_entries"]:
        change_lines.append(f"| {item['roi']} | {item['isotope']} | {fmt(item['A'])} | {fmt(item['B'])} |")
    change_lines.extend(["", "Остальные элементы базовой матрицы памяти совпадают. "
        "В этом детекторе базовый столбец Xe133m уже одинаков в Rs и Rg. "
        "Его эффективный столбец после поправок всё же меняется через "
        "изменённый дочерний столбец Xe133; сохранённые коэффициенты врастания "
        f"равны {fmt(scenario_B['modification']['Xe133m_daughter_checks']['sample_memory']['coefficient'],12)} "
        "для памяти в SAMPLE и "
        f"{fmt(scenario_B['modification']['Xe133m_daughter_checks']['gasbk']['coefficient'],12)} "
        "для GASBK. Полные столбцы до и после поправок сохранены в сценарии.", ""])
    position = lines.index("## Сценарий и калибровка")
    lines[position:position] = change_lines
    for role in ("SAMPLE", "GASBK", "DETBK"):
        item = scenario_A["measurement_metadata"][role]
        name = Path(scenario_A["input_files"][role]["path"]).name
        lines.append(f"| {role} | {name} | {item['start_time']} | {fmt(item['real_time_seconds'], 10)} | {fmt(item['live_time_seconds'], 10)} |")
    t = scenario_A["time_parameters_seconds"]
    lines.extend(["", f"Фиксированные ROI: `{scenario_A['roi_order']}`. Объём ксенона "
        f"{fmt(scenario_A['xenon_volume_cm3'], 10)} cm³; используемый объём воздуха "
        f"**{fmt(scenario_A['effective_air_volume_m3'], 12)} m³** (`V_Xe/0.087`). "
        f"Пробоотбор {fmt(t['t_a'],10)} s; ожидание после пробоотбора {fmt(t['t_p'],10)} s; "
        f"между началами GASBK и SAMPLE {fmt(t['tau'],10)} s. Полные границы ROI, "
        "интервалы и матрицы сохранены в файлах сценария.", "",
        "| Изотоп | ε SAMPLE | ε GASBK | Branching BR | ε × BR |",
        "|---|---:|---:|---:|---:|"])
    for j, isotope in enumerate(scenario_A["efficiencies"]["isotopes"]):
        eps = scenario_A["efficiencies"]["sample"][j]
        gas = scenario_A["efficiencies"]["gasbk"][j]
        br = scenario_A["branching"][isotope]
        lines.append(f"| {isotope} | {eps:.6f} | {gas:.6f} | {br:.4f} | {eps*br:.9f} |")
    lines.extend(["", "Эффективности указаны без branching. Для Rn222 сохраняется "
        "прежнее приближение `legacy` с ε=0.3 и периодом Rn222. Первая свежая "
        "координата радона вспомогательная, её не трактуем как атмосферную концентрацию.", "",
        "## Частоты ложных обнаружений", "",
        "Целевые 5% относятся к каждому изотопу отдельно. Точные 95%-е интервалы "
        "Clopper–Pearson маргинальные, не одновременные для всех строк. Частоты "
        "рассчитаны по доступным решениям, без замены отказов отсутствием обнаружения.", "",
        "![False-detection comparison](false_detection_comparison.png)", "",
        "| Вариант | Метод | Изотоп | Доступно / 100000 | Ложных | Частота | 95% CI | Сбоев | Границы среди всех N |",
        "|---|---|---|---:|---:|---:|---|---:|---|"])
    for r in rows:
        lines.append(f"| {r['variant']} | {r['method']} | {r['isotope']} | {r['n_valid']} | {r['n_false']} | "
            f"{pct(r['frequency'])} | {pct(r['ci95_low'])} … {pct(r['ci95_high'])} | {r['n_fail']} | "
            f"{pct(r['frequency_all_min'])} … {pct(r['frequency_all_max'])} |")
    lines.extend(["", "Границы среди всех N равны `k/N` и `(k+n_fail)/N`; это диапазон "
        "из-за неизвестных исходов отказавших расчётов, а не доверительный интервал.", "",
        "## Парное изменение решений", "",
        "В таблице ниже оставлены только наблюдения с доступными решениями "
        "обоих вариантов для данного метода и изотопа. Поэтому её частоты могут "
        "слегка отличаться от отдельных частот выше. Обозначения 01 и 10 означают "
        "соответственно отсутствие обнаружения в А/обнаружение в Б и обратный переход. "
        "Δ = 100 × (kБ − kА) / Nобщ выражено в процентных пунктах.", "",
        "| Метод | Изотоп | Nобщ | А, % | Б, % | Δ, п.п. | 00 | 01 | 10 | 11 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in pairs:
        lines.append(f"| {r['method']} | {r['isotope']} | {r['n_common_valid_decisions']} | "
            f"{pct(r['frequency_A_common'])} | {pct(r['frequency_B_common'])} | "
            f"{fmt(r['delta_frequency_percentage_points'])} | {r['n00']} | {r['n01']} | {r['n10']} | {r['n11']} |")
    lines.extend(["", "Все 100000 строк остаются в NPZ. Общая доступная подвыборка "
        "используется только для парного сравнения; количества решений, доступных только в А, "
        "только в Б и ни в одном варианте отдельно записаны в `paired_comparison.csv`.", "",
        "## Оценки концентраций и собственные LC", "",
        "Все значения — mBq/m³. Среднее и SD используют все успешные конечные "
        "свободные оценки, независимо от обнаружения и доступности LC. SD имеет "
        "делитель Nдоступных−1. Квантили LC используют все доступные LC и линейную "
        "интерполяцию NumPy. Никакие отрицательные оценки или выбросы не удалялись.", "",
        "| Вариант | Метод | Изотоп | Средняя оценка | SD | Медиана LC | LC: 2.5% … 97.5% | n оценок / n LC |",
        "|---|---|---|---:|---:|---:|---|---:|"])
    for r in rows:
        lines.append(f"| {r['variant']} | {r['method']} | {r['isotope']} | {fmt(r['mean_estimate_mBq_m3'])} | "
            f"{fmt(r['sd_estimate_mBq_m3'])} | {fmt(r['lc_median_mBq_m3'])} | "
            f"{fmt(r['lc_q025_mBq_m3'])} … {fmt(r['lc_q975_mBq_m3'])} | {r['n_available_estimate']} / {r['n_available_lc']} |")
    lines.extend(["", "`paired_comparison.csv` дополнительно содержит средние и SD "
        "разностей оценок Б−А и LCБ−LCА на общих доступных строках с отдельными знаменателями.", "",
        "## Диагностика", "",
        f"Исходная подгонка реальных спектров в предыдущем эксперименте: D={fmt(quality['statistic'],8)}, "
        f"номинальное ν={quality['degrees_of_freedom']}, D/ν={fmt(quality['reduced_statistic'],8)}. "
        "Значительное отклонение от единицы означает, что данный опыт не доказывает "
        "адекватность исходной модели реальным спектрам. Здесь изучается заданное "
        "упрощение обработки наблюдений, сгенерированных именно моделью А. "
        "Причины несогласия исходной подгонки с реальными спектрами не исследовались."])
    for variant in VARIANTS:
        values = view_variant(arrays, variant)
        for family, label in FAMILIES.items():
            d = summary["variants"][variant]["methods"][family]
            lines.extend(["", f"**{variant}, {label}:** успешных свободных оценок {d['n_successful_free']}, "
                f"вспомогательных {d['n_successful_auxiliary']}. Успешных свободных оценок с "
                f"недоступной ковариацией {d['n_successful_free_covariance_unavailable']}; "
                f"из них с доступным хотя бы одним решением {d['n_covariance_unavailable_any_decision']}, "
                f"со всеми четырьмя {d['n_covariance_unavailable_all_decisions']}. "
                f"По изотопам: `{d['n_covariance_unavailable_decision_by_isotope']}`. "
                "Отсутствие ковариации само по себе не исключает решение."])
    lines.extend(["", "Сообщения оптимизаторов учитываются отдельно от итогового "
        "статуса XeMat: исходный текст CONVERGENCE не отменяет отказ проверки "
        "допустимости/ККТ, а сообщение об исчерпании числа вычислений не отменяет "
        "успешную последующую проверку XeMat. Правила принятия результата не менялись.", "",
        "| Вариант | Метод | Подгонка | Сообщение | Число |",
        "|---|---|---|---|---:|"])
    for variant in VARIANTS:
        for family, label in FAMILIES.items():
            for phase in ("free", "auxiliary"):
                item = diagnostics[variant]["fit_diagnostics"][family][phase]
                for message, amount in sorted(item["messages"].items(), key=lambda item: -item[1]):
                    lines.append(f"| {variant} | {label} | {phase} | {md(message)} | {amount} |")
    lines.extend(["", "Предупреждения и отказы сохранены в `diagnostics.json`, "
        "а статусы/коды сообщений каждой строки — в NPZ. Стандартное предупреждение "
        "о неприменимости обычной ковариации ограниченной оценки не является отказом.", ""])
    for variant in VARIANTS:
        values = view_variant(arrays, variant)
        catalogue = diagnostics[variant]["diagnostic_catalogue"]
        lines.extend([f"### Вариант {variant}: записи ошибок и особые сообщения", "",
            f"Записей ошибок: **{len(diagnostics[variant]['errors'])}**. "
            "Отказавшие наблюдения не заменялись; численные векторы, если они были возвращены, сохранены.", ""])
        special = []
        for family, label in FAMILIES.items():
            for phase in ("free", "auxiliary"):
                prefix = family + "_" + phase
                for code, diag in enumerate(catalogue):
                    warnings = diag.get("warnings", [])
                    kkt = [message for message in warnings if "ККТ" in message]
                    maximum = "maximum number of function evaluations" in diag.get("message", "").lower()
                    if not kkt and not maximum:
                        continue
                    ids = np.flatnonzero(values[prefix + "_diagnostic_code"] == code)
                    for i in ids:
                        special.append((int(i), label, phase, int(values[prefix + "_status"][i]),
                            "; ".join(kkt) if kkt else diag["message"]))
        if special:
            lines.extend(["| ID | Метод | Подгонка | Статус | Причина/сообщение |", "|---:|---|---|---:|---|"])
            for i, label, phase, status, message in special:
                lines.append(f"| {i} | {label} | {phase} | {status} | {md(message)} |")
            lines.append("")
        for family, label in FAMILIES.items():
            for phase in ("free", "auxiliary"):
                item = diagnostics[variant]["fit_diagnostics"][family][phase]
                for category in ("warnings", "python_warnings"):
                    for message, amount in item[category].items():
                        lines.append(f"- {label}, {phase}, {category}: {md(message)} — {amount}.")
        lines.append("")
    lines.extend(["## Воспроизводимость и состав данных", "",
        f"Версия неизменённого ядра XeMat: `{scenario_A['production_commit']}`. "
        f"Python {scenario_A['software']['python']}, NumPy {scenario_A['software']['numpy']}, "
        f"SciPy {scenario_A['software']['scipy']}. Хеши ядра и экспериментальных скриптов сохранены.", "",
        f"Исходный seed наблюдений А: {scenario_A['master_seed']}; пачки "
        "`default_rng([master_seed, chunk])`, chunk=0…99, по 1000 строк. "
        "В данном сравнении генератор случайных наблюдений не запускался. "
        "Порядок строк сохранён: `replicate_id=1000*chunk+row_in_batch`.", "",
        "- `scenario_A.json`, `scenario_B.json`, `scenario_arrays_A.npz`, `scenario_arrays_B.npz` — "
        "метаданные, модели, база откликов, исходная истина А и разложение μ₀.",
        "- `observations.npz` — неизменённые наблюдения и идентификаторы для воспроизведения обработки Б.",
        "- `replicates.npz` — общие наблюдения, четыре свободные концентрации, собственные LC, "
        "решения и статусы обоих вариантов; полные вспомогательные 17-мерные векторы сохранены. "
        "Это все численные данные, необходимые для запрошенного сравнения.",
        "- `full_free_parameters.npz` — дополнительный отдельно скачиваемый файл с полными "
        "17-мерными свободными векторами А и Б в исходных смешанных единицах. Он вынесен "
        "за пределы основного ZIP из-за ограничения размера файла при публикации. "
        "Ни значения, ни точность хранения не сокращались; все строки доступны. "
        "Единицы обоих файлов описаны в `replicates_schema.md`.",
        "- `summary.csv` — 16 строк, варианты × методы × изотопы; `paired_comparison.csv` — 8 строк парных сравнений.",
        "- `diagnostics.json`, `summary.json` — диагностические сообщения, ошибки и происхождение всех данных.",
        "- `false_detection_comparison.pdf`, `false_detection_comparison.png` — итоговый рисунок с точными интервалами; PNG 300 dpi.",
        "- `plot_comparison.py`, `figures.tex`, `results_table.tex` — независимое оформление из CSV, английская подпись и таблица для статьи.",
        "- `prepare_scenario.py`, `study.py`, `analyze.py`, неизменённое ядро и контрольные суммы — воспроизводимость.",
        "", "`python analyze.py --from-merged` пересобирает таблицы и отчёт из "
        "основного `replicates.npz` без моделирования и подгонок; дополнительный файл "
        "полных свободных векторов для этого не требуется. Рисунок перестраивается "
        "скриптом `plot_comparison.py` из CSV. LD/MDC, кривые мощности, "
        "подстройка порогов и тесты Вальда не исследовались; для Б рассчитаны только "
        "собственные LC. Побочный фиксированный MDC, внутренне создаваемый API LC, "
        "не сохранялся и не анализировался. Основная статья не редактировалась."])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def schema(root, arrays):
    lines = ["# Saved A/B comparison arrays", "",
        "Load with `numpy.load('replicates.npz', allow_pickle=False)`. "
        "All 100000 source rows are retained in original replicate_id order. "
        "No observations were generated in this comparison.", "",
        "`x` and IDs are shared once. Prefixes A_ and B_ select processing model. "
        "`wls` is LS and `poisson` is free Poisson ML. `*_concentrations` and "
        "`*_LC` contain four columns [Xe135,Xe133,Xe131m,Xe133m] in mBq/m3. "
        "The full 17-column `*_auxiliary_physical` arrays retain native mixed units [a_V;gamma;zeta], "
        "specified by physical_parameter_units. Their four fresh Xe columns1:5 "
        "are in Bq/m3. Negative free estimates are retained.", "",
        "The separately downloadable `full_free_parameters.npz` contains "
        "A_wls_free_physical, A_poisson_free_physical, B_wls_free_physical, "
        "B_poisson_free_physical, each 100000 × 17 in the same native mixed units, "
        "plus replicate_id, parameter names/units and scenario hashes. "
        "Fresh Xe columns 1:5 multiplied by 1000 exactly equal the main archive's "
        "four concentration columns. This supplemental file is outside the main "
        "ZIP to satisfy publication file-size limits; no numerical data or precision "
        "have been discarded. Rebuilding requested tables needs only the main file.", "",
        "Each variant has its own Unicode JSON diagnostic catalogue and status "
        "legends. Diagnostic codes index that variant's catalogue. A codes are "
        "unchanged; B batch-local codes were remapped to one global B catalogue. "
        "Decision−1 means unavailable,0 means no detection,1 means detection. "
        "Unavailable decisions never count as non-detections. Returned numerical "
        "values from unsuccessful fits remain present with non-success statuses.", "",
        "## Arrays", "", "| Key | Shape | dtype |", "|---|---|---|"]
    for key, value in arrays.items():
        lines.append(f"| `{key}` | `{value.shape}` | `{value.dtype}` |")
    lines.extend(["", "## Tables", "",
        "summary.csv has 16 rows. Frequency/CI/bounds fields are proportions. "
        "n_generated denotes the size of the ORIGINAL generated ensemble, "
        "n_new_generated=0 for both variants in the current comparison. "
        "n_reused=n_observations=100000. Exact equal-tailed 95% Clopper–Pearson "
        "intervals use available decisions only. Missing-outcome bounds use allN. "
        "Mean/SD use all successful finite free estimates independently of LC; "
        "SD uses ddof1. LC quantiles use all finite available LC and NumPy linear interpolation.", "",
        "paired_comparison.csv uses common available decisions for its frequency "
        "difference and 2×2 table. n01 means A=0,B=1; n10 means A=1,B=0. "
        "delta_frequency_percentage_points=100*(n01−n10)/n_common_valid_decisions. "
        "Estimate and LC differences each use their own common-availability mask, "
        "with separately recorded denominators and empirical SD(ddof1). "
        "These paired quantities are descriptive; no hypothesis tests or extra studies are run.",
        "", "Full per-realization covariance matrices are intentionally absent. "
        "A missing free-fit covariance alone does not invalidate its decision."])
    (root / "replicates_schema.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--from-merged", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    scenario_A, scenario_B = read(root / "scenario_A.json"), read(root / "scenario_B.json")
    output = root / "replicates.npz"
    supplement = root / "full_free_parameters.npz"
    if args.from_merged:
        prior, diagnostics = read(root / "summary.json"), read(root / "diagnostics.json")
        require(sha(output) == prior["replicates_sha256"], "Merged arrays changed")
        require(sha(root / "scenario_A.json") == prior["scenario_A_sha256"], "A scenario changed")
        require(sha(root / "scenario_B.json") == prior["scenario_B_sha256"], "B scenario changed")
        with np.load(output, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        provenance = prior["source_provenance"]
        supplement_hash = prior["full_free_parameters_sha256"]
        supplement_bytes = prior["full_free_parameters_bytes"]
        if supplement.exists():
            require(sha(supplement) == supplement_hash, "Supplementary free vectors changed")
    else:
        arrays, diagnostics, provenance = collect(root, scenario_B)
    require(np.array_equal(arrays["replicate_id"], np.arange(100000)), "Merged ID mismatch")
    rows, variants = [], {}
    for variant in VARIANTS:
        values = view_variant(arrays, variant)
        new_rows, methods = summarize_variant(values, variant, scenario_B["n"])
        rows.extend(new_rows)
        variants[variant] = dict(label=VARIANTS[variant], methods=methods,
            n_error_records=len(diagnostics[variant]["errors"]))
        catalogue = [json.loads(item) for item in arrays[variant + "_diagnostic_catalogue_json"].tolist()]
        require(catalogue == diagnostics[variant]["diagnostic_catalogue"], variant + " catalogue mismatch")
        diagnostics[variant]["fit_diagnostics"] = diagnostic_counts(values, catalogue)
    pairs = paired_summary(arrays)
    if not args.from_merged:
        full_free = {f"{variant}_{family}_free_physical": arrays.pop(f"{variant}_{family}_free_physical")
            for variant in VARIANTS for family in FAMILIES}
        for key in ("replicate_id", "physical_parameter_names", "physical_parameter_units",
                    "physical_coordinate_label", "A_scenario_sha256", "B_scenario_sha256"):
            full_free[key] = arrays[key]
        temporary_supplement = root / "full_free_parameters.tmp.npz"
        np.savez_compressed(temporary_supplement, **full_free)
        os.replace(temporary_supplement, supplement)
        supplement_hash, supplement_bytes = sha(supplement), supplement.stat().st_size
        arrays["full_free_parameters_sha256"] = np.asarray(supplement_hash)
        temporary = root / "replicates.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    csv_write(root / "summary.csv", SUMMARY_FIELDS, rows)
    csv_write(root / "paired_comparison.csv", PAIRED_FIELDS, pairs)
    latex_table(root, rows)
    summary = dict(schema_version=1, n_observations=100000, n_new_generated=0,
        generation_model_variant="A", variants=variants, isotope_order=ISOTOPES,
        concentration_units="mBq/m3", frequency_units="proportion", alpha=.05,
        ci_method="Exact equal-tailed95% Clopper-Pearson conditional on available decisions",
        rows=rows, paired_rows=pairs, source_provenance=provenance,
        scenario_A_sha256=sha(root / "scenario_A.json"), scenario_B_sha256=sha(root / "scenario_B.json"),
        runner_lock_sha256=sha(root / "runner_lock.json"), observations_sha256=sha(root / "observations.npz"),
        replicates_sha256=sha(output), summary_csv_sha256=sha(root / "summary.csv"),
        full_free_parameters_sha256=supplement_hash, full_free_parameters_bytes=supplement_bytes,
        paired_csv_sha256=sha(root / "paired_comparison.csv"), analyze_script_sha256=sha(__file__),
        production_commit=scenario_B["production_commit"])
    write(root / "summary.json", summary)
    write(root / "diagnostics.json", diagnostics)
    report(root, scenario_A, scenario_B, arrays, rows, pairs, diagnostics, summary)
    schema(root, arrays)
    print(json.dumps(dict(n_observations=100000, n_new_generated=0, variants=variants,
        summary_rows=rows, paired_rows=pairs, merged_bytes=output.stat().st_size,
        supplemental_bytes=supplement_bytes, supplemental_sha256=supplement_hash,
        merged_sha256=sha(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
