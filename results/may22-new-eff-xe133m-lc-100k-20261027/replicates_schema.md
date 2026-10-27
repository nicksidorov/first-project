# Схема replicates.npz

Все 100 000 строк сохранены в порядке replicate_id=0,…,99999. Ничего не удалено и не заменено.
NPZ читается `numpy.load('replicates.npz', allow_pickle=False)`.

## Основные величины

`x`: исходные целочисленные наблюдения SAMPLE/GASBK/DETBK, по 7 ROI в каждом блоке.
`*_physical`: полный вектор `[a_V; gamma; zeta]` из 17 координат в **исходных** единицах; первые пять — Bq/m³ (Rn вспомогательный), следующие пять — ожидаемые опорные отсчёты GASBK, последние семь — ожидаемые отсчёты DETBK.
`*_concentrations`: четыре свободные оценки Xe135, Xe133, Xe131m, Xe133m в **mBq/m³**, включая отрицательные.
`*_LC`, `*_sigma0`: только проверяемый свежий Xe133m, **mBq/m³**. Четыре порога в этом эксперименте не вычислялись.
`*_free_covariance_available`: диагностический флаг; отсутствие ковариации свободного решения само по себе не запрещает решение об обнаружении.
`*_decision`: −1 — недоступное решение, 0 — свободная оценка ≤ LC, 1 — свободная оценка > LC.
`*_status`: коды определены в JSON-строках `fit_status_legend_json` и `lc_status_legend_json`.
`*_diagnostic_code`: глобальный индекс в `diagnostic_catalogue_json`. Каталог и вся исходная метаинформация пакетов доступны без pickle.
Даже если solver вернул success=False, его конечные численные значения сохраняются. Для статистики оценок требуется free_status=1, для статистики LC — LC_status=1, для частот и корреляции — decision≥0. Не трактуйте конечные значения неудачной подгонки как успешные.
`batch_metadata_json` содержит исходные сообщения/ошибки и хеши всех пакетов; локальные коды в этой метаинформации относятся к локальному каталогу пакета.
`seed_table`: master seed и номер пакета; внутри пакета используется один вызов `default_rng([master_seed,chunk]).poisson(mu0,size=(1000,21))`.

## Полный список полей

| Поле | Размер | dtype |
|---|---:|---|
| `batch_metadata_json` | `(100,)` | `<U2511` |
| `chunk` | `(100000,)` | `int16` |
| `concentration_units` | `()` | `<U6` |
| `decision_legend_json` | `()` | `<U137` |
| `diagnostic_catalogue_json` | `(6,)` | `<U245` |
| `fit_status_legend_json` | `()` | `<U200` |
| `isotope_order` | `(4,)` | `<U6` |
| `lc_status_legend_json` | `()` | `<U150` |
| `observation_names` | `(21,)` | `<U11` |
| `physical_coordinate_label` | `()` | `<U18` |
| `physical_parameter_names` | `(17,)` | `<U13` |
| `physical_units` | `(17,)` | `<U31` |
| `poisson_LC` | `(100000,)` | `float64` |
| `poisson_LC_diagnostic_code` | `(100000,)` | `int32` |
| `poisson_LC_status` | `(100000,)` | `int8` |
| `poisson_auxiliary_condition_number` | `(100000,)` | `float64` |
| `poisson_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `poisson_auxiliary_iterations` | `(100000,)` | `int32` |
| `poisson_auxiliary_objective` | `(100000,)` | `float64` |
| `poisson_auxiliary_physical` | `(100000, 17)` | `float64` |
| `poisson_auxiliary_rank` | `(100000,)` | `int16` |
| `poisson_auxiliary_status` | `(100000,)` | `int8` |
| `poisson_concentrations` | `(100000, 4)` | `float64` |
| `poisson_decision` | `(100000,)` | `int8` |
| `poisson_free_condition_number` | `(100000,)` | `float64` |
| `poisson_free_covariance_available` | `(100000,)` | `bool` |
| `poisson_free_diagnostic_code` | `(100000,)` | `int32` |
| `poisson_free_iterations` | `(100000,)` | `int32` |
| `poisson_free_objective` | `(100000,)` | `float64` |
| `poisson_free_physical` | `(100000, 17)` | `float64` |
| `poisson_free_rank` | `(100000,)` | `int16` |
| `poisson_free_status` | `(100000,)` | `int8` |
| `poisson_sigma0` | `(100000,)` | `float64` |
| `replicate_id` | `(100000,)` | `int64` |
| `row_in_batch` | `(100000,)` | `int16` |
| `runner_lock_sha256` | `()` | `<U64` |
| `scenario_sha256` | `()` | `<U64` |
| `seed_table` | `(100, 2)` | `int64` |
| `target_index` | `()` | `int64` |
| `target_isotope` | `()` | `<U6` |
| `wls_LC` | `(100000,)` | `float64` |
| `wls_LC_diagnostic_code` | `(100000,)` | `int32` |
| `wls_LC_status` | `(100000,)` | `int8` |
| `wls_auxiliary_condition_number` | `(100000,)` | `float64` |
| `wls_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `wls_auxiliary_iterations` | `(100000,)` | `int32` |
| `wls_auxiliary_objective` | `(100000,)` | `float64` |
| `wls_auxiliary_physical` | `(100000, 17)` | `float64` |
| `wls_auxiliary_rank` | `(100000,)` | `int16` |
| `wls_auxiliary_status` | `(100000,)` | `int8` |
| `wls_concentrations` | `(100000, 4)` | `float64` |
| `wls_decision` | `(100000,)` | `int8` |
| `wls_free_condition_number` | `(100000,)` | `float64` |
| `wls_free_covariance_available` | `(100000,)` | `bool` |
| `wls_free_diagnostic_code` | `(100000,)` | `int32` |
| `wls_free_iterations` | `(100000,)` | `int32` |
| `wls_free_objective` | `(100000,)` | `float64` |
| `wls_free_physical` | `(100000, 17)` | `float64` |
| `wls_free_rank` | `(100000,)` | `int16` |
| `wls_free_status` | `(100000,)` | `int8` |
| `wls_sigma0` | `(100000,)` | `float64` |
| `x` | `(100000, 21)` | `int64` |

## summary.csv

Две строки: LS и Poisson ML для Xe133m. Все frequency/ci/bounds записаны долями, не процентами. Интервал Clopper–Pearson построен по n_valid, а границы frequency_all_min/max — по всем n_generated с крайними трактовками недоступных решений. SD использует делитель n−1. Квантили используют numpy.quantile с default method='linear'. Корреляция Пирсона берётся по всем доступным парам, без отбора по обнаружению.
