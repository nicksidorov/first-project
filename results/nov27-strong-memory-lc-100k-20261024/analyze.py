"""Merge every saved November MC row and prepare the false-detection report.

This script does not generate observations, fit spectra, or calculate LC.
All statistical summaries come from the saved arrays. The denominator for
false-detection frequency is the number of available decisions, not N when
some decisions are missing. Failed rows remain in replicates.npz.
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
SUMMARY_FIELDS = [
    "method", "isotope", "n_generated", "n_valid", "n_false", "frequency",
    "ci95_low", "ci95_high", "n_fail", "mean_estimate_mBq_m3",
    "sd_estimate_mBq_m3", "lc_median_mBq_m3", "lc_q025_mBq_m3",
    "lc_q975_mBq_m3", "frequency_all_min", "frequency_all_max",
    "n_available_estimate", "n_available_lc", "n_covariance_unavailable_decision",
]
CONSTANT_ARRAYS = {
    "isotope_order", "concentration_units", "physical_coordinate_label", "seed",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(root, scenario):
    """Concatenate all planned batches and remap diagnostic codes globally."""
    n, size = int(scenario["n"]), int(scenario["batch_size"])
    batches = int(scenario["n_batches"])
    require(n == 100_000 and size == 1000 and batches == 100, "Unexpected planned N")
    scenario_hash = sha256(root / "scenario.json")
    lock = read_json(root / "runner_lock.json")
    require(lock["scenario_sha256"] == scenario_hash, "Runner scenario changed")
    require(lock["scenario_arrays_sha256"] == sha256(root / "scenario_arrays.npz"),
            "Scenario arrays changed")
    require(lock["study_sha256"] == sha256(root / "study.py"), "Runner changed")
    lock_hash = sha256(root / "runner_lock.json")
    collected, catalogue, catalogue_lookup = {}, [], {}
    errors, provenance, seeds = [], [], []
    legends = None
    expected_keys = None
    for chunk in range(batches):
        stem = f"batch_{chunk:03d}"
        raw_path, meta_path = root / "points" / (stem + ".npz"), root / "points" / (stem + ".json")
        meta = read_json(meta_path)
        seed = [int(scenario["master_seed"]), chunk]
        require(meta["chunk"] == chunk and meta["n_generated"] == size, "Wrong batch metadata")
        require(meta["seed"] == seed, "Batch seed mismatch")
        require(meta["runner_lock_sha256"] == lock_hash, "Batch lock mismatch")
        require(meta["npz_sha256"] == sha256(raw_path), "Batch array hash mismatch")
        this_legends = {key: meta[key] for key in ("fit_status_legend", "lc_status_legend", "decision_legend")}
        if legends is None:
            legends = this_legends
        require(this_legends == legends, "Status definitions differ between batches")
        mapping = []
        for diagnostic in meta["diagnostic_catalogue"]:
            encoded = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if encoded not in catalogue_lookup:
                catalogue_lookup[encoded] = len(catalogue)
                catalogue.append(diagnostic)
            mapping.append(catalogue_lookup[encoded])
        mapping = np.asarray(mapping, dtype=np.int32)
        with np.load(raw_path, allow_pickle=False) as raw:
            keys = set(raw.files)
            if expected_keys is None:
                expected_keys = keys
            require(keys == expected_keys, "Batch schema changed")
            require(np.array_equal(raw["replicate_id"], np.arange(chunk*size, (chunk+1)*size)),
                    "Replicate IDs missing/reordered")
            require(np.array_equal(raw["seed"], seed), "NPZ seed mismatch")
            require(raw["isotope_order"].tolist() == ISOTOPES, "Wrong isotope order")
            require(str(raw["concentration_units"].item()) == "mBq/m3", "Wrong units")
            require(raw["x"].shape == (size, 21) and np.all(raw["x"] >= 0), "Wrong counts")
            require(np.issubdtype(raw["x"].dtype, np.integer), "Counts must be integers")
            for key in sorted(keys - CONSTANT_ARRAYS):
                value = raw[key].copy()
                require(value.shape[0] == size, f"Wrong row count: {key}")
                if key.endswith("_diagnostic_code"):
                    require(np.all((value >= 0) & (value < len(mapping))), "Unknown diagnostic code")
                    value = mapping[value]
                collected.setdefault(key, []).append(value)
        seeds.append(seed)
        errors.extend({"chunk": chunk, **error} for error in meta["errors"])
        provenance.append(dict(chunk=chunk, n=size, seed=seed,
            json_file=f"points/{stem}.json", json_sha256=sha256(meta_path),
            npz_file=f"points/{stem}.npz", npz_sha256=meta["npz_sha256"],
            elapsed_seconds=meta["elapsed_seconds"]))
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    require(np.array_equal(arrays["replicate_id"], np.arange(n)), "Merged IDs not exactly 0..N-1")
    arrays.update(
        isotope_order=np.array(ISOTOPES), concentration_units=np.array("mBq/m3"),
        physical_coordinate_label=np.array(scenario["physical_coordinates"]),
        physical_parameter_names=np.asarray(scenario["physical_parameter_names"]),
        physical_parameter_units=np.asarray(scenario["physical_parameter_units"]),
        observation_order=np.asarray(scenario["observation_order"]), observation_units=np.array("counts"),
        seed_table=np.asarray(seeds, dtype=np.int64),
        diagnostic_catalogue_json=np.asarray([json.dumps(d, ensure_ascii=False, sort_keys=True) for d in catalogue]),
        status_legends_json=np.asarray(json.dumps(legends, ensure_ascii=False, sort_keys=True)),
        scenario_sha256=np.asarray(scenario_hash), runner_lock_sha256=np.asarray(lock_hash),
    )
    return arrays, catalogue, errors, provenance, legends


def diagnostic_counts(arrays, catalogue):
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for phase in ("free", "auxiliary"):
            prefix = family + "_" + phase
            codes, amounts = np.unique(arrays[prefix + "_diagnostic_code"], return_counts=True)
            status_values, status_amounts = np.unique(arrays[prefix + "_status"], return_counts=True)
            entries = [dict(code=int(code), n=int(amount), diagnostic=catalogue[int(code)])
                       for code, amount in zip(codes, amounts)]
            entries.sort(key=lambda item: (-item["n"], item["code"]))
            warnings, python_warnings, messages = Counter(), Counter(), Counter()
            for entry in entries:
                diagnostic, count = entry["diagnostic"], entry["n"]
                messages[str(diagnostic.get("message", diagnostic.get("exception", "No message")))] += count
                for value in diagnostic.get("warnings", []):
                    warnings[str(value)] += count
                for value in diagnostic.get("python_warnings", []):
                    python_warnings[str(value)] += count
            result[family][phase] = dict(
                n_attempted=int(np.count_nonzero(arrays[prefix + "_status"] != -2)),
                status_counts={str(int(s)): int(a) for s, a in zip(status_values, status_amounts)},
                diagnostic_counts=entries, messages=dict(messages),
                warnings=dict(warnings), python_warnings=dict(python_warnings),
            )
    return result


def summarize(arrays, scenario):
    n = int(scenario["n"])
    rows, method_summaries = [], {}
    for family, method in FAMILIES.items():
        estimates, thresholds = arrays[family + "_concentrations"], arrays[family + "_LC"]
        decisions = arrays[family + "_decision"]
        free_ok = arrays[family + "_free_status"] == 1
        auxiliary_ok = arrays[family + "_auxiliary_status"] == 1
        lc_ok = (arrays[family + "_LC_status"] == 1) & np.isfinite(thresholds) & (thresholds >= 0)
        no_cov = ~arrays[family + "_free_covariance_available"]
        valid = free_ok[:, None] & auxiliary_ok[:, None] & lc_ok & np.isfinite(estimates)
        require(np.array_equal(decisions >= 0, valid), f"Decision availability differs: {family}")
        require(np.all(np.isin(decisions, [-1, 0, 1])), "Unknown decision code")
        require(np.array_equal(decisions[valid], (estimates[valid] > thresholds[valid]).astype(np.int8)),
                f"Saved decisions differ from free > LC: {family}")
        np.testing.assert_equal(estimates, arrays[family + "_free_physical"][:, 1:5] * 1000.0)
        method_summaries[family] = dict(
            label=method, n_generated=n,
            n_successful_free=int(free_ok.sum()), n_successful_auxiliary=int(auxiliary_ok.sum()),
            n_successful_free_covariance_unavailable=int((free_ok & no_cov).sum()),
            n_covariance_unavailable_any_decision=int((no_cov & valid.any(axis=1)).sum()),
            n_covariance_unavailable_all_four_decisions=int((no_cov & valid.all(axis=1)).sum()),
            n_covariance_unavailable_decision_by_isotope=(no_cov[:, None] & valid).sum(axis=0).tolist(),
            n_any_failed_decision=int((~valid).any(axis=1).sum()),
            n_all_four_available_decisions=int(valid.all(axis=1).sum()),
        )
        for j, isotope in enumerate(ISOTOPES):
            available = valid[:, j]
            n_valid = int(available.sum())
            k = int((decisions[:, j] == 1).sum())
            n_fail = n - n_valid
            interval = binomtest(k, n_valid).proportion_ci(.95, method="exact") if n_valid else None
            good_estimate = free_ok & np.isfinite(estimates[:, j])
            a = estimates[good_estimate, j]
            lc = thresholds[lc_ok[:, j], j]
            quantiles = np.quantile(lc, [.025, .5, .975], method="linear") if len(lc) else [None]*3
            rows.append(dict(method=method, isotope=isotope, n_generated=n, n_valid=n_valid,
                n_false=k, frequency=k/n_valid if n_valid else None,
                ci95_low=float(interval.low) if interval else None,
                ci95_high=float(interval.high) if interval else None, n_fail=n_fail,
                mean_estimate_mBq_m3=float(np.mean(a)) if len(a) else None,
                sd_estimate_mBq_m3=float(np.std(a, ddof=1)) if len(a) > 1 else None,
                lc_median_mBq_m3=float(quantiles[1]) if len(lc) else None,
                lc_q025_mBq_m3=float(quantiles[0]) if len(lc) else None,
                lc_q975_mBq_m3=float(quantiles[2]) if len(lc) else None,
                frequency_all_min=k/n, frequency_all_max=(k+n_fail)/n,
                n_available_estimate=int(len(a)), n_available_lc=int(len(lc)),
                n_covariance_unavailable_decision=int(np.sum(no_cov & available))))
    return rows, method_summaries


def fmt(value, precision=6):
    return "—" if value is None else f"{value:.{precision}g}"


def pct(value):
    return "—" if value is None else f"{100*value:.3f}%"


def escape_md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(root, scenario, arrays, rows, diagnostics, summary):
    q = scenario["truth_construction"]["original_fit_diagnostics"]
    quality = q.get("fit_quality", {})
    lines = [
        "# XeMat: ложные обнаружения при сильной газовой памяти",
        "",
        "Сценарий построен по SPIKE20251127072232736.PHD (роль SAMPLE), "
        "GASBK20251126153029165.PHD и прежнему DETBK детектора **221**. "
        "Для обоих методов использованы одни и те же **100 000** комплектов "
        "из 21 пуассоновского отсчёта: 7 ROI SAMPLE, 7 GASBK и 7 DETBK.",
        "",
        "## Фиксированная истина и процедура",
        "",
        "Один раз выполнена существующая совместная неотрицательная пуассоновская "
        "подгонка исходных трёх измерений. В её физическом векторе `[a_V; gamma; zeta]` "
        "обнулены только четыре свежие концентрации ксенона. Сохранены оценённые "
        "свежий радон, вся газовая память и семь средних DETBK. Никакой дополнительной "
        "подгонки при нулевой гипотезе не было. Фиксированный вектор и средний спектр "
        "`mu0 = S_V @ truth + b` находятся в `scenario_arrays.npz`.",
        "",
        "В каждой реализации все 21 отсчёт разыграны заново независимо как "
        "`Poisson(mu0)`. Для МНК и свободного пуассоновского ММП выполнены свободная "
        "и соответствующая вспомогательная неотрицательная подгонки. Одна "
        "вспомогательная оценка каждого метода используется для всех четырёх "
        "проверок: поочерёдно убирается только свежий вклад проверяемого изотопа, "
        "прочие компоненты сохраняются, LC рассчитывается прежней формулой полной "
        "модели. Обнаружение — строго `free concentration > LC`. Оба числа переведены "
        "в **mBq/m³**, множитель порога равен 1. Отрицательные оценки и все строки сохранены.",
        "",
        "Веса МНК и NNLS в каждой реализации: "
        "`W = diag(1 / where(x == 0, 1, x))`. Заменяется только дисперсия нулевого "
        "отсчёта; сам нулевой отсчёт остаётся нулём. Пуассоновский критерий использует "
        "исходные отсчёты. Настройки `fit`: `max_iter=2000`, `tol=1e-10`, без изменений. "
        "Систематические неопределённости и неопределённость выбора фиксированной "
        "истины здесь не разыгрываются. Кривые мощности, подстройка порога, LD/MDC "
        "и тест Вальда не исследовались. API LC в режиме `fixed` внутренне создаёт "
        "неиспользуемый формульный побочный результат MDC; он не сохранялся и не анализировался.",
        "",
        "## Измерения и калибровки",
        "",
        "| Роль | Файл | Начало измерения из файла | Реальное время, s | Живое время, s |",
        "|---|---|---|---:|---:|",
    ]
    for role in ("SAMPLE", "GASBK", "DETBK"):
        data = scenario["measurement_metadata"][role]
        filename = Path(scenario["input_files"][role]["path"]).name
        lines.append(f"| {role} | {filename} | {data['start_time']} | {fmt(data['real_time_seconds'], 10)} | {fmt(data['live_time_seconds'], 10)} |")
    times = scenario["time_parameters_seconds"]
    intervals = scenario["measurement_intervals_seconds"]
    lines.extend([
        "", f"ROI: **{scenario['roi_order']}**, фиксированные энергетические границы; "
        "точные границы в энергии и каналах для каждого измерения сохранены в `scenario.json`.",
        "", f"Пробоотбор: {fmt(times['t_a'], 10)} s; от конца пробоотбора до SAMPLE: "
        f"{fmt(times['t_p'], 10)} s; между началами GASBK и SAMPLE: "
        f"{fmt(intervals['sample_start_minus_gasbk_start'], 10)} s; от конца GASBK до SAMPLE: "
        f"{fmt(intervals['sample_start_minus_gasbk_end'], 10)} s.",
        "", f"Объём извлечённого ксенона: **{fmt(scenario['xenon_volume_cm3'], 10)} cm³**. "
        f"Используемый эквивалентный объём воздуха: **{fmt(scenario['effective_air_volume_m3'], 12)} m³** "
        "по формуле `V0 = V_Xe / 0.087`. Это объём модели; записанный в исходном "
        "файле расходный объём воздуха сохранён отдельно в метаданных.",
        "", "Эффективности фиксированные, без включённого branching. "
        "Чувствительность `I = epsilon × BR`.", "",
        "| Изотоп | ε SAMPLE | ε GASBK | Branching (BR) | ε × BR |",
        "|---|---:|---:|---:|---:|",
    ])
    for i, name in enumerate(scenario["efficiencies"]["isotopes"]):
        a, g = scenario["efficiencies"]["sample"][i], scenario["efficiencies"]["gasbk"][i]
        br = scenario["branching"][name]
        lines.append(f"| {name} | {a:.6f} | {g:.6f} | {br:.4f} | {a*br:.9f} |")
    lines.extend([
        "", "Для Rn222 сохранена прежняя модель `legacy`: ε=0.3 и поправка "
        "по периоду Rn222; это приближение радоновой цепочки. Первая координата "
        "`a_V:Rn222` служебная и не трактуется как атмосферная концентрация радона. "
        "Матрицы `R_c0`, `R_g0`, преобразования и времена неизменны и сохранены полностью.",
        "", "Фиксированная память в координатах `gamma` (ожидаемые отсчёты опорных "
        "ROI GASBK до поправки на дочь), порядок Rn222, Xe135, Xe133, Xe131m, Xe133m: "
        "`" + ", ".join(fmt(v, 10) for v in scenario["truth_physical"][5:10]) + "`.",
        "", "Фиксированные средние DETBK в порядке ROI: `" + ", ".join(fmt(v, 10) for v in scenario["truth_physical"][10:]) + "`. "
        "Это совместно подогнанные средние по прежнему файлу DETBK, а не его неизменённые наблюдаемые отсчёты.",
        "", "## Частоты ложных обнаружений", "",
        "Все четыре истинные свежие концентрации равны нулю. Целевой уровень 5% "
        "относится **отдельно к каждому изотопу**. Интервалы точные Clopper–Pearson, "
        "95%, маргинальные; одновременное накрытие всех восьми строк ими не заявляется.",
        "", "![False-detection frequencies](false_detections.png)", "",
        "| Метод | Изотоп | Доступно / N | Ложных | Частота | 95% CI | Сбоев решений | Границы среди всех N |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ])
    for row in rows:
        lines.append(f"| {row['method']} | {row['isotope']} | {row['n_valid']} / {row['n_generated']} | "
            f"{row['n_false']} | {pct(row['frequency'])} | {pct(row['ci95_low'])} … {pct(row['ci95_high'])} | "
            f"{row['n_fail']} | {pct(row['frequency_all_min'])} … {pct(row['frequency_all_max'])} |")
    lines.extend([
        "", "Частота и CI рассчитаны по доступным решениям: `k / n_valid`. "
        "Последний столбец — границы возможной частоты среди всех N: "
        "`k/N` и `(k+n_fail)/N`; это не доверительный интервал. Сбой не приравнивается "
        "к отсутствию обнаружения, и неудачные реализации не заменялись.",
        "", "## Оценки концентраций и LC", "",
        "Все величины ниже — **mBq/m³**. Среднее и SD используют все успешные "
        "конечные свободные оценки данного изотопа, независимо от обнаружения "
        "и доступности LC. SD имеет делитель `n_available_estimate−1`. "
        "Квантили LC используют все доступные LC; линейная интерполяция NumPy. "
        "В `summary.csv` отдельно указаны оба размера выборки.", "",
        "| Метод | Изотоп | Средняя оценка | SD оценки | Медиана LC | LC: 2.5% … 97.5% | n оценок / n LC |",
        "|---|---|---:|---:|---:|---|---:|",
    ])
    for row in rows:
        lines.append(f"| {row['method']} | {row['isotope']} | {fmt(row['mean_estimate_mBq_m3'])} | "
            f"{fmt(row['sd_estimate_mBq_m3'])} | {fmt(row['lc_median_mBq_m3'])} | "
            f"{fmt(row['lc_q025_mBq_m3'])} … {fmt(row['lc_q975_mBq_m3'])} | "
            f"{row['n_available_estimate']} / {row['n_available_lc']} |")
    lines.extend(["", "## Диагностика и границы интерпретации", "",
        f"Исходная неотрицательная пуассоновская подгонка: success={q['success']}, "
        f"итераций {q['iterations']}, ранг {q['rank']}, сообщение «{escape_md(q['message'])}».",
    ])
    if quality:
        lines.extend(["", f"Для исходных измерений девианс **D={fmt(quality['statistic'], 8)}**, "
            f"номинальное ν={quality['degrees_of_freedom']}, **D/ν={fmt(quality['reduced_statistic'], 8)}**. "
            "Это заметно выше 1: данная совместная модель описывает реальные спектры "
            "не полностью. Для ограниченной подгонки ν номинальное. Представленное "
            "Монте-Карло проверяет статистическое правило **внутри фиксированной "
            "подогнанной модели**, а не правильность её калибровок для реальных данных. "
            "Причины расхождения в этой задаче не исследовались."])
    for family, method in FAMILIES.items():
        item = summary["methods"][family]
        lines.extend(["", f"**{method}:** успешных свободных оценок {item['n_successful_free']}; "
            f"успешных вспомогательных {item['n_successful_auxiliary']}. "
            f"У {item['n_successful_free_covariance_unavailable']} успешных свободных оценок "
            "обычная ковариация недоступна. Из них имеют хотя бы одно доступное "
            f"решение **{item['n_covariance_unavailable_any_decision']}**, все четыре решения — "
            f"**{item['n_covariance_unavailable_all_four_decisions']}**. "
            "По отдельным изотопам (в порядке Xe135, Xe133, Xe131m, Xe133m): "
            f"`{item['n_covariance_unavailable_decision_by_isotope']}`. Эти решения остаются в статистике."])
    kkt_rejections = []
    for family, method in FAMILIES.items():
        for phase, phase_ru in (("free", "свободная"), ("auxiliary", "вспомогательная")):
            prefix = family + "_" + phase
            for row_index in np.flatnonzero(arrays[prefix + "_status"] == 0):
                code = int(arrays[prefix + "_diagnostic_code"][row_index])
                diagnostic = diagnostics["diagnostic_catalogue"][code]
                kkt_warnings = [str(value) for value in diagnostic.get("warnings", [])
                                if "ККТ" in str(value) or "KKT" in str(value)]
                if kkt_warnings:
                    kkt_rejections.append(dict(method=method, phase=phase_ru,
                        replicate_id=int(arrays["replicate_id"][row_index]),
                        warning="; ".join(kkt_warnings),
                        optimizer_message=str(diagnostic.get("message", "")),
                        missing_lc=int(np.count_nonzero(arrays[family + "_LC_status"][row_index] != 1)),
                        missing_decisions=int(np.count_nonzero(arrays[family + "_decision"][row_index] < 0))))
    if kkt_rejections:
        lines.extend(["", f"**Отказы внутренней проверки оптимальности XeMat: {len(kkt_rejections)} подгонок.** "
            "Они завершились с `success=False`, потому что существующая проверка "
            "условий ККТ не подтвердила оптимальность. При этом исходный оптимизатор "
            "мог вернуть текст `CONVERGENCE: RELATIVE REDUCTION ...`: этот текст "
            "означает выполнение его критерия остановки и не отменяет последующий "
            "отказ проверки XeMat. Поэтому такая вспомогательная оценка не "
            "используется для LC; отсутствие LC и решения сохраняется как сбой. "
            "Дополнительные повторные подгонки или замены реализаций не выполнялись, "
            "настройки и алгоритмы не менялись.", "",
            "| ID реализации | Метод | Подгонка | Причина XeMat | Недоступно LC / решений |",
            "|---:|---|---|---|---:|",
        ])
        for entry in kkt_rejections:
            lines.append(f"| {entry['replicate_id']} | {entry['method']} | {entry['phase']} | "
                f"{escape_md(entry['warning'])} | {entry['missing_lc']} / {entry['missing_decisions']} |")
    lines.extend(["", "Обычная ковариация вспомогательной ограниченной оценки не используется "
        "для решения об обнаружении. Соответствующее стандартное предупреждение "
        "о её неприменимости само по себе не является сбоем.", "",
        "Сообщения оптимизаторов (число реализаций):", "",
        "| Метод | Подгонка | Сообщение | Число |",
        "|---|---|---|---:|",
    ])
    for family, method in FAMILIES.items():
        for phase, phase_ru in (("free", "свободная"), ("auxiliary", "вспомогательная")):
            entry = diagnostics["fit_diagnostics"][family][phase]
            for message, count in sorted(entry["messages"].items(), key=lambda item: -item[1]):
                lines.append(f"| {method} | {phase_ru} | {escape_md(message)} | {count} |")
    for family, method in FAMILIES.items():
        for phase, phase_ru in (("free", "свободная"), ("auxiliary", "вспомогательная")):
            entries = diagnostics["fit_diagnostics"][family][phase]["diagnostic_counts"]
            accepted_codes = [entry["code"] for entry in entries
                if entry["diagnostic"].get("success") is True
                and "maximum number of function evaluations" in entry["diagnostic"].get("message", "").lower()]
            if accepted_codes:
                prefix = family + "_" + phase
                accepted = np.isin(arrays[prefix + "_diagnostic_code"], accepted_codes)
                ids = arrays["replicate_id"][accepted].tolist()
                lines.extend(["", f"{method}, {phase_ru}: в {len(ids)} случаях оптимизатор сообщил "
                    "о достижении лимита вычислений функции, однако последующая существующая "
                    "проверка допустимости и ККТ в XeMat прошла (`success=True`). "
                    f"Эти решения сохранены как доступные; ID: `{ids}`. "
                    "Критерии принятия результата не менялись."])
    lines.extend(["", "Предупреждения возвращаемого результата и Python:", ""])
    any_warning = False
    for family, method in FAMILIES.items():
        for phase, phase_ru in (("free", "свободная"), ("auxiliary", "вспомогательная")):
            entry = diagnostics["fit_diagnostics"][family][phase]
            for category in ("warnings", "python_warnings"):
                for message, count in sorted(entry[category].items(), key=lambda item: -item[1]):
                    any_warning = True
                    lines.append(f"- {method}, {phase_ru}, {category}: {escape_md(message)} — {count}.")
    if not any_warning:
        lines.append("Предупреждений нет.")
    lines.extend(["", f"Записей об ошибках/восстановлении отдельных LC: **{len(diagnostics['errors'])}**. "
        "Их точные причины и идентификаторы сохранены в `diagnostics.json`; "
        "статусы и коды диагностик каждого расчёта находятся в `replicates.npz`. "
        "Статус success и доступность решения учитываются явно; текст сообщения "
        "оптимизатора отдельно от этих статусов не трактуется как отказ."])
    if diagnostics["errors"]:
        error_counts = Counter((e["family"], e["phase"], e["message"]) for e in diagnostics["errors"])
        lines.extend(["", "| Метод | Этап | Причина | Записей |", "|---|---|---|---:|"])
        for (family, phase, message), amount in error_counts.most_common():
            lines.append(f"| {FAMILIES[family]} | {phase} | {escape_md(message)} | {amount} |")
    above = [r for r in rows if r["ci95_low"] is not None and r["ci95_low"] > .05]
    below = [r for r in rows if r["ci95_high"] is not None and r["ci95_high"] < .05]
    lines.extend(["", "Наблюдаемые особенности этой серии:", ""])
    if above:
        lines.append("- Маргинальный 95% интервал целиком выше 5%: " + "; ".join(f"{r['method']} {r['isotope']} ({pct(r['frequency'])})" for r in above) + ".")
    if below:
        lines.append("- Маргинальный 95% интервал целиком ниже 5%: " + "; ".join(f"{r['method']} {r['isotope']} ({pct(r['frequency'])})" for r in below) + ".")
    if not above and not below:
        lines.append("- Все восемь маргинальных 95% интервалов включают целевой уровень 5%.")
    lines.append("- Это описание данного фиксированного сценария; обобщение на иные составы памяти и проверка причин различий не выполнялись.")
    lines.extend(["", "## Воспроизведение и состав архива", "",
        f"Версия ядра XeMat: `{scenario['production_commit']}`. Python "
        f"{scenario['software']['python']}, NumPy {scenario['software']['numpy']}, "
        f"SciPy {scenario['software']['scipy']}. Исходники неизменённого ядра и их "
        "SHA-256 сохранены в `kernel_snapshot/` и `scenario.json`.",
        "", f"Master seed: **{scenario['master_seed']}**. Для пачки `chunk=0,…,99` "
        "используется `numpy.random.default_rng([master_seed, chunk])`, затем "
        "`rng.poisson(mu0, size=(1000,21))`. Каждая строка имеет постоянный "
        "`replicate_id=1000*chunk+row_in_batch`. Порядок выполнения пачек не влияет "
        "на их генерацию. Все 100 пачек вошли в результат; технических отбрасываемых серий не было.",
        "", f"SHA-256 сценария: `{summary['scenario_sha256']}`.",
        "", "Основные файлы:", "",
        "- `scenario.json`, `scenario_arrays.npz` — идентификаторы, времена, ROI, объём, "
        "калибровки, правила, исходные спектры/матрицы, исходная неотрицательная оценка, "
        "нулевой свежий Xe и разложение `mu0`.",
        "- `replicates.npz` — все 100 000 строк, наблюдения, свободные и полные "
        "вспомогательные оценки, LC, решения, статусы, диагностика; "
        "`replicates_schema.md` описывает ключи и единицы.",
        "- `summary.csv` — восемь строк численных результатов; `summary.json` "
        "содержит те же итоги, размеры выборок и происхождение пачек.",
        "- `report.md`, `false_detections.pdf`, `false_detections.png` — отчёт "
        "и единственный итоговый рисунок; PNG 300 dpi.",
        "- `plot_false_detections.py` — отдельное построение рисунка из CSV без моделирования.",
        "- `diagnostics.json` — сообщения и предупреждения оптимизаторов, "
        "ошибки с идентификаторами и сводные статусы.",
        "- `batch_metadata/` — все 100 исходных JSON пачек с seed, сообщениями, "
        "ошибками и контрольными суммами. Дублирующие отдельные NPZ пачек "
        "не включены: все их строки сохранены в общем `replicates.npz`.",
        "- `inputs/`, `kernel_snapshot/`, `prepare_scenario.py`, `study.py`, "
        "`analyze.py` и файлы фиксации SHA-256 — исходные измерения, код и воспроизводимость.",
        "- `verify_study.py`, `verification.json` и контрольные суммы архива — "
        "независимая проверка согласованности сохранённых решений и сводной таблицы.",
        "", "Для изменения оформления используйте только `plot_false_detections.py` "
        "с существующим `summary.csv` (аргументы перечислены при `--help`); "
        "повторное Монте-Карло не требуется. Команда `python analyze.py --from-merged` "
        "повторно строит таблицу, диагностику и текст отчёта из сохранённого "
        "`replicates.npz` без отдельных файлов пачек и без подгонок. Изменённые "
        "выходные файлы после такого запуска потребуют обновления контрольных сумм. "
        "`python verify_study.py` выполняет проверку архивных данных и воспроизводимости "
        "наблюдений по seed; повторной подгонки при этом нет. "
        "Полные ковариационные матрицы отдельных "
        "реализаций не сохранялись. Основная статья и производственное приложение не менялись.",
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def schema(root, arrays, legends):
    lines = ["# replicates.npz: saved arrays and units", "",
        "The archive loads with `numpy.load(path, allow_pickle=False)`. All 100000 "
        "planned rows are retained in ascending `replicate_id` order. No negative "
        "estimates or outliers are clipped or removed.", "",
        "`wls` denotes LS and `poisson` denotes free Poisson ML. Each family has "
        "one corresponding auxiliary constrained fit (`nnls`, `poisson_nonnegative`).", "",
        "| Key pattern | Meaning / units |", "|---|---|",
        "| `x` | N×21 original integer ROI observations; SAMPLE, GASBK, DETBK blocks |",
        "| `replicate_id`, `chunk`, `row_in_batch` | Stable global and within-batch identities |",
        "| `seed_table` | 100×2 seeds `[master_seed, chunk]` |",
        "| `{family}_concentrations` | N×4 free Xe estimates, mBq/m³; isotope_order defines columns |",
        "| `{family}_LC` | N×4 LC, mBq/m³; no threshold multiplier |",
        "| `{family}_decision` | N×4 int8: -1 unavailable, 0 available nondetection, 1 false detection |",
        "| `{family}_{free,auxiliary}_physical` | N×17 native physical estimates `[a_V; gamma; zeta]` |",
        "| `{family}_{free,auxiliary}_status` | Fit status codes; see table below |",
        "| `{family}_{free,auxiliary}_diagnostic_code` | Index into shared diagnostic_catalogue_json |",
        "| `{family}_{free,auxiliary}_{iterations,objective,rank,condition_number}` | Saved solver diagnostics |",
        "| `{family}_free_covariance_available` | Bool; covariance availability, never itself a decision filter |",
        "| `{family}_LC_status` | N×4 LC availability; see table below |",
        "| `{family}_LC_diagnostic_code` | N×4 diagnostic catalogue indices |",
        "| `diagnostic_catalogue_json` | Unicode JSON strings; decode selected entry with json.loads |",
        "| `physical_parameter_names`, `physical_parameter_units` | Full 17-parameter column meaning and native units |",
        "| `observation_order`, `observation_units` | Full 21-observation row meaning and units |",
        "| `isotope_order`, `concentration_units`, `physical_coordinate_label` | Coordinate conventions |",
        "| `status_legends_json` | Machine-readable versions of status tables |",
        "| `scenario_sha256`, `runner_lock_sha256` | Provenance checksums |", "",
        "The native physical vector is **not** entirely in concentration units: "
        "fresh Xe columns 1:5 are Bq/m³ (multiply by1000 for concentration output); "
        "fresh Rn column0 is an auxiliary radon coordinate; gamma columns5:10 are "
        "reference-ROI expected counts in GASBK; zeta columns10:17 are expected "
        "DETBK counts. A returned numerical estimate from an unsuccessful solver "
        "is retained but has a non-success status and is not an available estimate "
        "in summary.csv. Missing numerical values are NaN, unavailable decisions -1.", "",
    ]
    for name, legend in legends.items():
        lines.extend(["## " + name, "", "| Code | Meaning |", "|---:|---|"])
        for key in sorted(legend, key=int):
            lines.append(f"| {key} | {legend[key]} |")
        lines.append("")
    lines.extend(["## Concrete array schema", "", "| Key | Shape | dtype |", "|---|---|---|"])
    for key, value in arrays.items():
        lines.append(f"| `{key}` | `{value.shape}` | `{value.dtype}` |")
    lines.extend(["", "## summary.csv", "",
        "Frequency and CI columns contain **proportions**, not percentages. "
        "The figure/report multiply these by100. CI is exact equal-tailed "
        "95% Clopper–Pearson for k out of n_valid available decisions. "
        "`frequency_all_min=k/N`, `frequency_all_max=(k+n_fail)/N` are "
        "missing-outcome bounds, not confidence intervals. The interval's "
        "conditional interpretation among available decisions does not make "
        "assumptions about failed decisions.", "",
        "Mean and empirical SD use all successful finite free estimates, "
        "independent of LC availability or detection. SD uses denominator "
        "n_available_estimate−1. LC quantiles use all entries with LC_status=1 "
        "and finite nonnegative LC (NumPy linear interpolation). "
        "`n_covariance_unavailable_decision` counts rows having an available "
        "decision but unavailable ordinary free-fit covariance.", "",
        "Batch-local diagnostic codes are remapped to a single global catalogue "
        "during merging; the corresponding messages are unchanged. Every batch "
        "hash is recorded in summary.json. Original batch JSON metadata are "
        "included in `batch_metadata/`; separate NPZ batch files are omitted "
        "because all their rows are retained in replicates.npz. References "
        "to `points/batch_NNN.*` in source provenance describe original run "
        "locations. `python analyze.py --from-merged` needs no points directory. "
        "Full per-replicate covariance matrices "
        "are intentionally absent.",
    ])
    (root / "replicates_schema.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--from-merged", action="store_true",
        help="Rebuild CSV, diagnostics and report from archived replicates.npz; no raw batches or fits needed")
    args = parser.parse_args()
    root = args.root.resolve()
    scenario = read_json(root / "scenario.json")
    output = root / "replicates.npz"
    if args.from_merged:
        previous_summary = read_json(root / "summary.json")
        previous_diagnostics = read_json(root / "diagnostics.json")
        require(previous_summary["replicates_sha256"] == sha256(output), "Saved merged arrays changed")
        require(previous_summary["scenario_sha256"] == sha256(root / "scenario.json"), "Saved scenario changed")
        require(previous_summary["runner_lock_sha256"] == sha256(root / "runner_lock.json"), "Saved runner lock changed")
        with np.load(output, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        catalogue = [json.loads(value) for value in arrays["diagnostic_catalogue_json"].tolist()]
        legends = json.loads(arrays["status_legends_json"].item())
        errors = previous_diagnostics["errors"]
        provenance = previous_summary["source_batches"]
        require(catalogue == previous_diagnostics["diagnostic_catalogue"], "Saved diagnostic catalogue differs")
        require(np.array_equal(arrays["replicate_id"], np.arange(scenario["n"])), "Wrong saved IDs")
        require(arrays["isotope_order"].tolist() == ISOTOPES, "Wrong saved isotope order")
        require(arrays["concentration_units"].item() == "mBq/m3", "Wrong saved concentration units")
        require(arrays["scenario_sha256"].item() == previous_summary["scenario_sha256"], "Wrong embedded scenario hash")
    else:
        arrays, catalogue, errors, provenance, legends = collect(root, scenario)
    rows, methods = summarize(arrays, scenario)
    diagnostics = dict(schema_version=1, fit_diagnostics=diagnostic_counts(arrays, catalogue),
        errors=errors, diagnostic_catalogue=catalogue, status_legends=legends)
    if not args.from_merged:
        temporary = root / "replicates.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema_version=1, scenario_id=scenario["scenario_id"],
        n_generated=int(scenario["n"]), n_batches=len(provenance), isotope_order=ISOTOPES,
        concentration_units="mBq/m3", frequency_units="proportion", alpha=scenario["alpha"],
        ci_method="Exact equal-tailed 95% Clopper-Pearson conditional on available decisions",
        covariance_unavailable_excludes_decision=False, empirical_sd_ddof=1,
        estimate_statistics="All successful finite free estimates, independently of decision/LC availability",
        lc_statistics="All available finite nonnegative LC; NumPy linear quantiles",
        n_error_records=len(errors), methods=methods, rows=rows,
        scenario_sha256=sha256(root / "scenario.json"), runner_lock_sha256=sha256(root / "runner_lock.json"),
        replicates_sha256=sha256(output), summary_csv_sha256=sha256(root / "summary.csv"),
        production_commit=scenario["production_commit"], analyze_script_sha256=sha256(__file__),
        source_batches=provenance)
    write_json(root / "diagnostics.json", diagnostics)
    write_json(root / "summary.json", summary)
    schema(root, arrays, legends)
    report(root, scenario, arrays, rows, diagnostics, summary)
    print(json.dumps({"n_generated": summary["n_generated"], "methods": methods,
        "rows": rows, "n_error_records": len(errors), "replicates_bytes": output.stat().st_size,
        "replicates_sha256": summary["replicates_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
