"""Aggregate all 100,000 saved XeMat rows, report LC, and draw one figure.

No fitting, observation simulation, covariance recalculation, or MDC/LD work.
Default: validate and merge all batch checkpoints, then save all results.
--from-replicates: rebuild summaries/figure from the merged NPZ without batches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest, norm

FAMILIES = ("wls", "poisson")
LABELS = {"wls": "LS", "poisson": "Poisson ML"}
META_KEYS = {"isotope_order", "target_isotope", "target_index", "concentration_units",
             "physical_coordinate_label", "physical_units", "seed"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                          encoding="utf-8")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def collect(root, scenario):
    """Concatenate every row, preserving failures and remapping message IDs."""
    batches = []
    metadata = []
    catalogue = []
    catalogue_index = {}
    row_keys = None
    constant_arrays = None
    runner_hash = sha256(root / "runner_lock.json")
    for chunk in range(scenario["n_batches"]):
        path = root / "points" / f"batch_{chunk:03d}.npz"
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"All 100 batches are required; missing {path.name}")
        meta = read_json(meta_path)
        expected_seed = [scenario["master_seed"], chunk]
        assert meta["chunk"] == chunk
        assert meta["n_generated"] == scenario["batch_size"]
        assert meta["seed"] == expected_seed
        assert meta["runner_lock_sha256"] == runner_hash
        assert meta["npz_sha256"] == sha256(path), f"Changed checkpoint {path.name}"
        with np.load(path, allow_pickle=False) as stored:
            batch = {key: stored[key].copy() for key in stored.files}
        assert np.array_equal(batch.pop("seed"), expected_seed)
        constants = {key: batch.pop(key) for key in META_KEYS-{"seed"}}
        if constant_arrays is None:
            constant_arrays = constants
            row_keys = set(batch)
        else:
            assert set(batch) == row_keys
            for key, value in constants.items():
                assert np.array_equal(value, constant_arrays[key]), key
        n = scenario["batch_size"]
        assert all(value.shape[0] == n for value in batch.values())
        assert np.array_equal(batch["replicate_id"], np.arange(chunk*n, (chunk+1)*n))
        assert np.all(batch["chunk"] == chunk)
        assert np.array_equal(batch["row_in_batch"], np.arange(n))
        remap = []
        for entry in meta["diagnostic_catalogue"]:
            text = canonical(entry)
            if text not in catalogue_index:
                catalogue_index[text] = len(catalogue)
                catalogue.append(entry)
            remap.append(catalogue_index[text])
        for key in batch:
            if key.endswith("_diagnostic_code"):
                codes = batch[key]
                assert np.all((codes >= 0) & (codes < len(remap)))
                batch[key] = np.asarray(remap, dtype=np.int32)[codes]
        batches.append(batch)
        metadata.append(meta)
    arrays = {key: np.concatenate([batch[key] for batch in batches]) for key in sorted(row_keys)}
    arrays.update(constant_arrays)
    arrays.update({
        "physical_parameter_names": np.asarray(scenario["parameter_names"]),
        "observation_names": np.asarray(scenario["observation_names"]),
        "seed_table": np.asarray([meta["seed"] for meta in metadata]),
        "scenario_sha256": np.asarray(sha256(root / "scenario.json")),
        "runner_lock_sha256": np.asarray(runner_hash),
        "diagnostic_catalogue_json": np.asarray([canonical(entry) for entry in catalogue]),
        "batch_metadata_json": np.asarray([canonical(meta) for meta in metadata]),
    })
    for key in ("fit_status_legend", "lc_status_legend", "decision_legend"):
        arrays[key+"_json"] = np.asarray(canonical(metadata[0][key]))
    return arrays


def validate_arrays(arrays, scenario, expected_n=None):
    """Check saved decisions algebraically, never recompute an estimate."""
    n = scenario["n_generated"] if expected_n is None else expected_n
    assert np.array_equal(arrays["replicate_id"], np.arange(n))
    assert arrays["x"].shape == (n, 21)
    assert np.issubdtype(arrays["x"].dtype, np.integer)
    assert np.all(arrays["x"] >= 0)
    assert np.array_equal(arrays["isotope_order"], scenario["concentration_order"])
    assert int(arrays["target_index"]) == 4
    assert str(arrays["target_isotope"]) == "Xe133m"
    assert str(arrays["concentration_units"]) == "mBq/m3"
    for family in FAMILIES:
        def get(suffix):
            return arrays[f"{family}_{suffix}"]
        for phase in ("free", "auxiliary"):
            values, status = get(phase+"_physical"), get(phase+"_status")
            assert values.shape == (n, 17)
            assert np.isin(status, [-2, -1, 0, 1, 2]).all()
            assert np.isfinite(values[status == 1]).all()
        concentrations = get("concentrations")
        assert concentrations.shape == (n, 4)
        assert np.array_equal(concentrations, 1000*get("free_physical")[:, 1:5], equal_nan=True)
        good_lc = get("LC_status") == 1
        assert np.isin(get("LC_status"), [-2, -1, 0, 1]).all()
        assert np.all(get("auxiliary_status")[good_lc] == 1)
        assert np.isfinite(get("LC")[good_lc]).all()
        assert np.isfinite(get("sigma0")[good_lc]).all()
        assert np.all(get("LC")[good_lc] >= 0)
        assert np.allclose(get("LC")[good_lc], norm.ppf(.95)*get("sigma0")[good_lc],
                           rtol=3e-15, atol=1e-15)
        valid = (get("free_status") == 1) & good_lc
        expected = np.full(n, -1, dtype=np.int8)
        expected[valid] = concentrations[valid, 3] > get("LC")[valid]
        assert np.array_equal(expected, get("decision")), family


def finite_stat(values, which):
    if len(values) == 0 or (which == "sd" and len(values) < 2):
        return None
    if which == "mean":
        return float(np.mean(values))
    if which == "sd":
        return float(np.std(values, ddof=1))
    return float(np.quantile(values, which))


def summarize(arrays):
    rows = []
    n = len(arrays["replicate_id"])
    for family in FAMILIES:
        def get(suffix):
            return arrays[f"{family}_{suffix}"]
        decision = get("decision")
        valid = decision >= 0
        n_valid = int(valid.sum())
        k = int(np.sum(decision == 1))
        n_fail = n-n_valid
        interval = binomtest(k, n_valid).proportion_ci(.95, method="exact") if n_valid else None
        free_ok = get("free_status") == 1
        lc_ok = get("LC_status") == 1
        estimates = get("concentrations")[:, 3]
        thresholds = get("LC")
        cov_missing = ~get("free_covariance_available")
        correlation = None
        if n_valid > 1 and np.std(estimates[valid]) > 0 and np.std(thresholds[valid]) > 0:
            correlation = float(np.corrcoef(estimates[valid], thresholds[valid])[0, 1])
        rows.append({
            "method": LABELS[family], "method_key": family, "isotope": "Xe133m",
            "n_generated": n, "n_valid": n_valid, "n_false": k,
            "frequency": k/n_valid if n_valid else None,
            "ci95_low": float(interval.low) if interval else None,
            "ci95_high": float(interval.high) if interval else None,
            "n_fail": n_fail, "frequency_all_min": k/n,
            "frequency_all_max": (k+n_fail)/n,
            "n_available_estimate": int(free_ok.sum()),
            "mean_estimate_mBq_m3": finite_stat(estimates[free_ok], "mean"),
            "sd_estimate_mBq_m3": finite_stat(estimates[free_ok], "sd"),
            "n_available_lc": int(lc_ok.sum()),
            "lc_mean_mBq_m3": finite_stat(thresholds[lc_ok], "mean"),
            "lc_sd_mBq_m3": finite_stat(thresholds[lc_ok], "sd"),
            "lc_median_mBq_m3": finite_stat(thresholds[lc_ok], .5),
            "lc_q025_mBq_m3": finite_stat(thresholds[lc_ok], .025),
            "lc_q975_mBq_m3": finite_stat(thresholds[lc_ok], .975),
            "n_paired": n_valid, "correlation_estimate_LC": correlation,
            "n_successful_free_covariance_unavailable": int(np.sum(cov_missing & free_ok)),
            "n_covariance_unavailable_decision": int(np.sum(cov_missing & valid)),
        })
    return rows


def diagnostics(arrays):
    catalogue = [json.loads(str(value)) for value in arrays["diagnostic_catalogue_json"]]
    batches = [json.loads(str(value)) for value in arrays["batch_metadata_json"]]
    result = {"diagnostic_catalogue": catalogue, "families": {},
              "errors": [error for batch in batches for error in batch["errors"]],
              "batch_provenance": [{key: batch[key] for key in
                  ("chunk", "seed", "n_generated", "npz_sha256", "runner_lock_sha256", "elapsed_seconds")}
                  for batch in batches]}
    for family in FAMILIES:
        data = {}
        for phase in ("free", "auxiliary", "LC"):
            status = arrays[f"{family}_{phase}_status"]
            data[phase+"_status_counts"] = {str(int(k)): int(v) for k, v in
                                             zip(*np.unique(status, return_counts=True))}
            counts = Counter(int(code) for code in arrays[f"{family}_{phase}_diagnostic_code"])
            data[phase+"_messages"] = [
                {"code": code, "count": count, "diagnostic": catalogue[code]}
                for code, count in sorted(counts.items())]
        result["families"][family] = data
    result["n_error_records"] = len(result["errors"])
    result["n_unique_replicates_with_errors"] = len({e["replicate_id"] for e in result["errors"]})
    return result


def schema_text(arrays):
    lines = ["# Схема replicates.npz", "",
             "Все 100 000 строк сохранены в порядке replicate_id=0,…,99999. Ничего не удалено и не заменено.",
             "NPZ читается `numpy.load('replicates.npz', allow_pickle=False)`.", "",
             "## Основные величины", "",
             "`x`: исходные целочисленные наблюдения SAMPLE/GASBK/DETBK, по 7 ROI в каждом блоке.",
             "`*_physical`: полный вектор `[a_V; gamma; zeta]` из 17 координат в **исходных** единицах; первые пять — Bq/m³ (Rn вспомогательный), следующие пять — ожидаемые опорные отсчёты GASBK, последние семь — ожидаемые отсчёты DETBK.",
             "`*_concentrations`: четыре свободные оценки Xe135, Xe133, Xe131m, Xe133m в **mBq/m³**, включая отрицательные.",
             "`*_LC`, `*_sigma0`: только проверяемый свежий Xe133m, **mBq/m³**. Четыре порога в этом эксперименте не вычислялись.",
             "`*_free_covariance_available`: диагностический флаг; отсутствие ковариации свободного решения само по себе не запрещает решение об обнаружении.",
             "`*_decision`: −1 — недоступное решение, 0 — свободная оценка ≤ LC, 1 — свободная оценка > LC.",
             "`*_status`: коды определены в JSON-строках `fit_status_legend_json` и `lc_status_legend_json`.",
             "`*_diagnostic_code`: глобальный индекс в `diagnostic_catalogue_json`. Каталог и вся исходная метаинформация пакетов доступны без pickle.",
             "Даже если solver вернул success=False, его конечные численные значения сохраняются. Для статистики оценок требуется free_status=1, для статистики LC — LC_status=1, для частот и корреляции — decision≥0. Не трактуйте конечные значения неудачной подгонки как успешные.",
             "`batch_metadata_json` содержит исходные сообщения/ошибки и хеши всех пакетов; локальные коды в этой метаинформации относятся к локальному каталогу пакета.",
             "`seed_table`: master seed и номер пакета; внутри пакета используется один вызов `default_rng([master_seed,chunk]).poisson(mu0,size=(1000,21))`.", "",
             "## Полный список полей", "", "| Поле | Размер | dtype |", "|---|---:|---|"]
    for key in sorted(arrays):
        lines.append(f"| `{key}` | `{arrays[key].shape}` | `{arrays[key].dtype}` |")
    lines += ["", "## summary.csv", "",
              "Две строки: LS и Poisson ML для Xe133m. Все frequency/ci/bounds записаны долями, не процентами. Интервал Clopper–Pearson построен по n_valid, а границы frequency_all_min/max — по всем n_generated с крайними трактовками недоступных решений. SD использует делитель n−1. Квантили используют numpy.quantile с default method='linear'. Корреляция Пирсона берётся по всем доступным парам, без отбора по обнаружению."]
    return "\n".join(lines)+"\n"


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}g}"


def report_text(scenario, rows, diag, root):
    meta = scenario["metadata"]
    temporal = [("SAMPLE", "sample_temporal"), ("GASBK", "background_temporal"), ("DETBK", "detector_temporal")]
    lines = ["# XeMat: повторная проверка LC для майской пробы при новых эффективностях", "",
        "Исследован комплект 22 мая 2026 года, детектор 221: **100 000 новых реализаций**, исходное правило LC для свежего **Xe133m**, методы МНК и свободного пуассоновского ММП. Целевой односторонний уровень — 5%. Ни LD/MDC, ни мощность, ни множители коррекции порога не рассчитывались.", "",
        "## Что именно считалось истинным", "",
        "Это **не пустая проба**. Исходные реальные SAMPLE, GASBK и DETBK один раз совместно подогнаны неотрицательным пуассоновским ММП с новыми эффективностями. В полном физическом векторе из 17 координат обнулена **только свежая Xe133m**. Остальные 16 координат, включая другие свежие изотопы, радон, память и детекторный фон, сохранены. Дополнительной подгонки под нулевой гипотезой нет.", "",
        "После этого вектор истинных параметров фиксирован. В каждой реализации независимо разыграны все 21 отсчёт SAMPLE/GASBK/DETBK: `x* ~ Poisson(mu0)`. Один и тот же вектор наблюдений использован обоими методами. Все три спектра, включая DETBK, флуктуируют. Неопределённость опорной истины, эффективности, branching, объёма и матриц R не разыгрывалась. Выводы условны относительно этой фиксированной модели и смеси.", "",
        "| Свежая координата | Опорная неотрицательная оценка, mBq/m³ | Генерирующая истина, mBq/m³ |", "|---|---:|---:|"]
    for j, name in enumerate(["Rn222 (вспом.)"]+scenario["concentration_order"]):
        lines.append(f"| {name} | {scenario['z_reference'][j]*1000:.9g} | {scenario['z_truth'][j]*1000:.9g} |")
    lines += ["", "Первая радоновая координата вспомогательная, не атмосферная концентрация Rn222. Полная память, фон и разложение mu0 сохранены в scenario_arrays.npz; в таблице выше показаны только первые пять координат.", "",
        "## Исходные данные и калибровка", "", "| Блок | Исходный файл | Начало измерения из файла | Реальное время, с | Живое время, с |", "|---|---|---|---:|---:|"]
    for block, key in temporal:
        data = meta[key]
        lines.append(f"| {block} | `{scenario['input_files'][block]['file']}` | {data['start_time']} | {data['real_time']:.9g} | {data['live_time']:.9g} |")
    lines += ["", f"Фиксированные ROI: {', '.join(map(str, scenario['roi_order']))}. Матрицы Rs221/Rg221 исходные, без правок. Эффективный объём V0 = **{meta['effective_air_volume_m3']:.9g} м³**; объём выделенного Xe = {meta['xenon_volume_cm3']:g} см³.",
        f"Времена: сбор t_a={meta['time_parameters']['t_a']:g} с, ожидание t_p={meta['time_parameters']['t_p']:g} с, интервал tau={meta['time_parameters']['tau']:g} с. Все кинетические поправки сохранены.", "",
        "Эффективности ниже не включают branching; чувствительность использует их произведение. Для SAMPLE и GASBK установлены одинаковые ε:", "",
        "| Компонента | ε SAMPLE | ε GASBK | Branching |", "|---|---:|---:|---:|"]
    lines.append(f"| Rn222 (legacy) | {scenario['radon_efficiency']:g} | {scenario['radon_efficiency']:g} | {scenario['branching']['Rn222']:g} |")
    for j, name in enumerate(scenario["concentration_order"]):
        lines.append(f"| {name} | {scenario['sample_efficiencies'][j]:.6f} | {scenario['gasbk_efficiencies'][j]:.6f} | {scenario['branching'][name]:.4f} |")
    lines += ["", "## Неизменённая процедура оценки и LC", "",
        "1. Для каждого набора x выполняется свободная оценка: МНК или пуассоновский ММП. Отрицательные оценки сохраняются.",
        "2. Выполняется соответствующая вспомогательная подгонка: NNLS для МНК или неотрицательный пуассоновский ММП для ММП. Неотрицательность относится к физическим координатам.",
        "3. В полученном вспомогательном векторе обнуляется только свежая Xe133m, сохраняя остальные оценки. По этому среднему mu_null и **всем 17 столбцам** модели вычисляется существующая статистическая ковариация K0=(Sᵀ diag(1/mu_null) S)⁻¹ посредством текущей реализации SVD. LC=1.64485362695 sqrt(K0[4,4]). Новая нулевая подгонка не выполняется.",
        "4. Обнаружение: свободная оценка Xe133m > LC; обе величины переведены в mBq/m³. Поскольку истинная свежая Xe133m равна нулю, каждое такое событие ложно положительно.", "",
        "Веса МНК/NNLS: W=diag(1/where(x==0,1,x)), заново по исходным отсчётам каждой реализации. Замена нуля единицей относится только к дисперсии для веса; нулевые наблюдения сохраняются. Пуассоновские методы используют исходные целочисленные отсчёты. Оптимизаторы, веса и допуски ядра не менялись (max_iter=2000, tol=10⁻¹⁰). LC вычисляется небольшим адаптером lc_only.py, который точно воспроизводит только часть LC существующей процедуры, не запуская ненужный расчёт MDC как побочный результат.", "",
        "## Частота ложных обнаружений", "",
        "Интервалы — **точные двусторонние 95% интервалы Clopper–Pearson**. Знаменатель частоты — число доступных решений. Сбой не считается отсутствием обнаружения.", "",
        "| Метод | Ложные / доступные | Частота | 95% интервал | Недоступные решения | Возможная частота среди всех N |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['n_false']} / {row['n_valid']} | {100*row['frequency']:.3f}% | {100*row['ci95_low']:.3f}–{100*row['ci95_high']:.3f}% | {row['n_fail']} | {100*row['frequency_all_min']:.3f}–{100*row['frequency_all_max']:.3f}% |")
    lines += ["", "![False-positive frequencies](false_detections.png)", "",
        "На рисунке **две точки**: одна для МНК, другая для ММП, каждая описывает весь прогон из 100 000 реализаций. Отрезки показывают погрешность оценки частоты Монте-Карло. Пунктир — номинальные 5%, а не наблюдённый результат. Это проверка уровня только для свежей Xe133m; другие изотопы в этой задаче не тестировались.", "",
        "| Метод | Средняя свободная оценка | SD свободной оценки | Средний LC | Медиана LC | 2.5–97.5% LC | corr(оценка, LC) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {fmt(row['mean_estimate_mBq_m3'])} | {fmt(row['sd_estimate_mBq_m3'])} | {fmt(row['lc_mean_mBq_m3'])} | {fmt(row['lc_median_mBq_m3'])} | {fmt(row['lc_q025_mBq_m3'])}–{fmt(row['lc_q975_mBq_m3'])} | {fmt(row['correlation_estimate_LC'],5)} |")
    lines += ["", "Все концентрации и LC в этой таблице — mBq/m³. SD рассчитано с делителем n−1. Статистика свободной оценки использует все успешные свободные решения; статистика LC — все доступные LC, даже если свободная подгонка не удалась. Корреляция Пирсона рассчитана по всем доступным парам оценки и LC, без отбора по обнаружению; их количества приведены в summary.csv.", ""]
    for row in rows:
        relation = "выше" if row["ci95_low"] > .05 else "ниже" if row["ci95_high"] < .05 else "совместим с"
        lines.append(f"Для {row['method']} полученный интервал {relation} номинального уровня 5%." if relation != "совместим с" else f"Для {row['method']} полученный интервал включает номинальные 5%.")
    old_path = root / "historical_reference" / "poisson_10000_summary.json"
    if old_path.exists():
        lines += ["", "## Сопоставление с прежними примерно 7%", "",
            "В сохранённом прежнем майском опыте ММП дал **697/10 000 = 6.97%** при старых ε=(0.2057, 0.314, 0.258, 0.258). Исторические файлы включены в historical_reference/ только для справки; прежние калибровки здесь повторно не моделировались.",
            f"При новых ε текущий результат ММП — **{100*rows[1]['frequency']:.3f}%**. Это сравнение двух конечных независимых экспериментов: старый N=10 000, новый N=100 000, другое случайное зерно, и опорная смесь заново оценена с новой калибровкой. Разность процентов поэтому нельзя целиком приписать изменению ε или считать парной оценкой её эффекта. Старый отдельный МНК-прогон использовал свою вспомогательную истину; его число не используется для парного сравнения с нынешним общим для двух методов генератором."]
    lines += ["", "## Сходимость и недоступные величины", "",
        f"Сохранено {diag['n_error_records']} записей о сбоях, затрагивающих {diag['n_unique_replicates_with_errors']} уникальных реализаций. Все строки оставлены в replicates.npz; неудачные реализации не заменялись и не отбирались. Полные причины и коды — в diagnostics.json и встроенных каталогах replicates.npz.", "",
        "| Метод | Успешная свободная оценка без ковариации | Из них доступное решение об обнаружении |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['n_successful_free_covariance_unavailable']} | {row['n_covariance_unavailable_decision']} |")
    lines += ["", "Недоступная ковариация свободного ММП не исключает реализацию, если сама оценка и LC доступны. LC использует ковариацию при среднем после удаления изотопа, а не ковариацию свободной подгонки. Предупреждение о том, что обратная информационная матрица не является ковариацией ограниченной оценки, ожидаемо и само по себе не является отказом.", "",
        "Сообщения оптимизаторов (число реализаций; окончательный флаг success ядра сохранён отдельно):", ""]
    for family in FAMILIES:
        for phase in ("free", "auxiliary"):
            for entry in diag["families"][family][phase+"_messages"]:
                d = entry["diagnostic"]
                lines.append(f"- {LABELS[family]}, {phase}: {entry['count']}; success={d.get('success', 'exception')}; «{d.get('message', d.get('exception', ''))}».")
    if diag["errors"]:
        lines += ["", "Причины настоящих сбоев (агрегированные; полные идентификаторы сохранены):", ""]
        error_counts = Counter((e.get("family"), e.get("phase"), e.get("message", "")) for e in diag["errors"])
        for (family, phase, message), count in error_counts.items():
            lines.append(f"- {family}, {phase}: {count}; «{message}».")
    lines += ["", "## Воспроизведение и файлы", "",
        f"Master seed: **{scenario['master_seed']}**. Сто пакетов по 1000 строк. Для пакета j=0,…,99: `numpy.random.default_rng([{scenario['master_seed']}, j]).poisson(mu0, size=(1000,21))`. Алгоритм генератора PCG64. Все исходные наблюдения, четыре свободные концентрации и полные свободные/вспомогательные оценки обоих методов сохранены.",
        f"Версия ядра: commit `{scenario['production_commit']}`. Python {scenario['software']['python']}, NumPy {scenario['software']['numpy']}, SciPy {scenario['software']['scipy']}. Хеши исходных данных, модели и кода — scenario.json, scenario.lock.json и runner_lock.json. Снимок ядра включён в kernel_snapshot/.", "",
        "- `scenario.json`, `scenario_arrays.npz`, `source_fit.json`: исходные условия, матрицы, опорная оценка, изменённая истина и mu0.",
        "- `replicates.npz`, `replicates_schema.md`: все наблюдения и результаты, статусы, причины, порядок и единицы параметров.",
        "- `summary.csv`, `summary.json`, `diagnostics.json`: частоты, точные интервалы, описательные характеристики и полная диагностика.",
        "- `false_detections.pdf`, `false_detections.png`: векторный PDF и PNG 300 dpi.",
        "- `study.py`, `lc_only.py`, `prepare_scenario.py`, `kernel_snapshot/`, `inputs/`: воспроизводимый расчёт.",
        "- `analyze.py`, `plot_false_detections.py`: пересборка отчёта и изменение рисунка без нового Монте-Карло.",
        "- `README.md`: инструкция запуска и чтения архива; `figure.tex`: вставка рисунка в статью с английской подписью.",
        "- `verify_study.py`: независимая проверка сохранённых решений, частот, интервалов и порогов; результат полного аудита включается при упаковке архива.", "",
        "Из каталога архива: `python analyze.py --from-replicates` пересоздаёт таблицы, отчёт и рисунок из сохранённых результатов. `python plot_false_detections.py` читает только summary.csv. Полное намеренное повторение MC: `python study.py --workers 4`, затем `python analyze.py`. В рабочем каталоге для возобновления сохранены пакеты points/; архив не дублирует их массивы. Переносимый replicates.npz содержит все строки и исходную метаинформацию, а batch_metadata/ — отдельные JSON каждого пакета. Основная статья и алгоритмы приложения этим экспериментом не изменялись.", "",
        "**English figure caption.** False-positive frequencies for fresh ¹³³ᵐXe in the May 22 detector-221 scenario using updated fixed efficiencies. LS and unrestricted Poisson ML use the same 100,000 Poisson realizations of SAMPLE, GASBK and DETBK, with their respective nonnegative auxiliary fits and the original LC rule. Points and error bars denote observed frequencies and exact 95% Clopper–Pearson intervals among available decisions; the dashed line marks the nominal 5% level. Only the true fresh ¹³³ᵐXe concentration is zero; the fitted nuisance mixture is retained.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--from-replicates", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    scenario = read_json(root / "scenario.json")
    assert sha256(root / "scenario_arrays.npz") == scenario["scenario_arrays_sha256"]
    if args.from_replicates:
        with np.load(root / "replicates.npz", allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        assert str(arrays["scenario_sha256"]) == sha256(root / "scenario.json")
        assert str(arrays["runner_lock_sha256"]) == sha256(root / "runner_lock.json")
    else:
        arrays = collect(root, scenario)
    validate_arrays(arrays, scenario)
    rows = summarize(arrays)
    diag = diagnostics(arrays)
    if not args.from_replicates:
        np.savez_compressed(root / "replicates.npz", **arrays)
    # Portable checkpoint metadata: the archive need not duplicate every raw
    # batch NPZ, because all row arrays are already in replicates.npz.
    metadata_dir = root / "batch_metadata"
    metadata_dir.mkdir(exist_ok=True)
    for encoded in arrays["batch_metadata_json"]:
        batch_meta = json.loads(str(encoded))
        write_json(metadata_dir / f"batch_{batch_meta['chunk']:03d}.json", batch_meta)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "diagnostics.json", diag)
    write_json(root / "summary.json", {
        "scenario_id": scenario["scenario_id"], "alpha": .05,
        "n_generated": len(arrays["replicate_id"]), "no_rows_removed": True,
        "decision_rule": "free fresh Xe133m concentration > corresponding LC",
        "interval": "exact two-sided 95% Clopper-Pearson, denominator n_valid",
        "sd_denominator": "n_available - 1", "quantile_method": "linear",
        "correlation_sample": "all available paired estimates and LC, without selection on detection",
        "rows": rows, "replicates_sha256": sha256(root / "replicates.npz"),
        "scenario_sha256": sha256(root / "scenario.json"),
        "runner_lock_sha256": sha256(root / "runner_lock.json"),
        "analysis_script_sha256": sha256(Path(__file__)),
        "analysis_software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "n_error_records": diag["n_error_records"],
        "n_unique_replicates_with_errors": diag["n_unique_replicates_with_errors"],
    })
    (root / "replicates_schema.md").write_text(schema_text(arrays), encoding="utf-8")
    (root / "report.md").write_text(report_text(scenario, rows, diag, root), encoding="utf-8")
    from plot_false_detections import plot
    plot(root / "summary.csv", root)
    for row in rows:
        print(f"{row['method']}: {row['n_false']}/{row['n_valid']} = "
              f"{100*row['frequency']:.4f}% [{100*row['ci95_low']:.4f}, {100*row['ci95_high']:.4f}]%; "
              f"{row['n_fail']} unavailable; corr={row['correlation_estimate_LC']:.6f}")


if __name__ == "__main__":
    main()
