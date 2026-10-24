# replicates.npz: saved arrays and units

The archive loads with `numpy.load(path, allow_pickle=False)`. All 100000 planned rows are retained in ascending `replicate_id` order. No negative estimates or outliers are clipped or removed.

`wls` denotes LS and `poisson` denotes free Poisson ML. Each family has one corresponding auxiliary constrained fit (`nnls`, `poisson_nonnegative`).

| Key pattern | Meaning / units |
|---|---|
| `x` | N×21 original integer ROI observations; SAMPLE, GASBK, DETBK blocks |
| `replicate_id`, `chunk`, `row_in_batch` | Stable global and within-batch identities |
| `seed_table` | 100×2 seeds `[master_seed, chunk]` |
| `{family}_concentrations` | N×4 free Xe estimates, mBq/m³; isotope_order defines columns |
| `{family}_LC` | N×4 LC, mBq/m³; no threshold multiplier |
| `{family}_decision` | N×4 int8: -1 unavailable, 0 available nondetection, 1 false detection |
| `{family}_{free,auxiliary}_physical` | N×17 native physical estimates `[a_V; gamma; zeta]` |
| `{family}_{free,auxiliary}_status` | Fit status codes; see table below |
| `{family}_{free,auxiliary}_diagnostic_code` | Index into shared diagnostic_catalogue_json |
| `{family}_{free,auxiliary}_{iterations,objective,rank,condition_number}` | Saved solver diagnostics |
| `{family}_free_covariance_available` | Bool; covariance availability, never itself a decision filter |
| `{family}_LC_status` | N×4 LC availability; see table below |
| `{family}_LC_diagnostic_code` | N×4 diagnostic catalogue indices |
| `diagnostic_catalogue_json` | Unicode JSON strings; decode selected entry with json.loads |
| `physical_parameter_names`, `physical_parameter_units` | Full 17-parameter column meaning and native units |
| `observation_order`, `observation_units` | Full 21-observation row meaning and units |
| `isotope_order`, `concentration_units`, `physical_coordinate_label` | Coordinate conventions |
| `status_legends_json` | Machine-readable versions of status tables |
| `scenario_sha256`, `runner_lock_sha256` | Provenance checksums |

The native physical vector is **not** entirely in concentration units: fresh Xe columns 1:5 are Bq/m³ (multiply by1000 for concentration output); fresh Rn column0 is an auxiliary radon coordinate; gamma columns5:10 are reference-ROI expected counts in GASBK; zeta columns10:17 are expected DETBK counts. A returned numerical estimate from an unsuccessful solver is retained but has a non-success status and is not an available estimate in summary.csv. Missing numerical values are NaN, unavailable decisions -1.

## decision_legend

| Code | Meaning |
|---:|---|
| -1 | unavailable decision; never count as nondetection |
| 0 | available: free estimate <= LC |
| 1 | available: free estimate > LC |

## fit_status_legend

| Code | Meaning |
|---:|---|
| -2 | not attempted |
| -1 | exception |
| 0 | solver returned success=False |
| 1 | successful finite physical estimate |
| 2 | missing/nonfinite/wrong-size physical estimate |

## lc_status_legend

| Code | Meaning |
|---:|---|
| -2 | not attempted: required fit estimate unavailable |
| -1 | LC exception |
| 0 | nonfinite/negative LC |
| 1 | available LC |

## Concrete array schema

| Key | Shape | dtype |
|---|---|---|
| `chunk` | `(100000,)` | `int16` |
| `poisson_LC` | `(100000, 4)` | `float64` |
| `poisson_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `poisson_LC_status` | `(100000, 4)` | `int8` |
| `poisson_auxiliary_condition_number` | `(100000,)` | `float64` |
| `poisson_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `poisson_auxiliary_iterations` | `(100000,)` | `int32` |
| `poisson_auxiliary_objective` | `(100000,)` | `float64` |
| `poisson_auxiliary_physical` | `(100000, 17)` | `float64` |
| `poisson_auxiliary_rank` | `(100000,)` | `int16` |
| `poisson_auxiliary_status` | `(100000,)` | `int8` |
| `poisson_concentrations` | `(100000, 4)` | `float64` |
| `poisson_decision` | `(100000, 4)` | `int8` |
| `poisson_free_condition_number` | `(100000,)` | `float64` |
| `poisson_free_covariance_available` | `(100000,)` | `bool` |
| `poisson_free_diagnostic_code` | `(100000,)` | `int32` |
| `poisson_free_iterations` | `(100000,)` | `int32` |
| `poisson_free_objective` | `(100000,)` | `float64` |
| `poisson_free_physical` | `(100000, 17)` | `float64` |
| `poisson_free_rank` | `(100000,)` | `int16` |
| `poisson_free_status` | `(100000,)` | `int8` |
| `replicate_id` | `(100000,)` | `int64` |
| `row_in_batch` | `(100000,)` | `int16` |
| `wls_LC` | `(100000, 4)` | `float64` |
| `wls_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `wls_LC_status` | `(100000, 4)` | `int8` |
| `wls_auxiliary_condition_number` | `(100000,)` | `float64` |
| `wls_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `wls_auxiliary_iterations` | `(100000,)` | `int32` |
| `wls_auxiliary_objective` | `(100000,)` | `float64` |
| `wls_auxiliary_physical` | `(100000, 17)` | `float64` |
| `wls_auxiliary_rank` | `(100000,)` | `int16` |
| `wls_auxiliary_status` | `(100000,)` | `int8` |
| `wls_concentrations` | `(100000, 4)` | `float64` |
| `wls_decision` | `(100000, 4)` | `int8` |
| `wls_free_condition_number` | `(100000,)` | `float64` |
| `wls_free_covariance_available` | `(100000,)` | `bool` |
| `wls_free_diagnostic_code` | `(100000,)` | `int32` |
| `wls_free_iterations` | `(100000,)` | `int32` |
| `wls_free_objective` | `(100000,)` | `float64` |
| `wls_free_physical` | `(100000, 17)` | `float64` |
| `wls_free_rank` | `(100000,)` | `int16` |
| `wls_free_status` | `(100000,)` | `int8` |
| `x` | `(100000, 21)` | `int64` |
| `isotope_order` | `(4,)` | `<U6` |
| `concentration_units` | `()` | `<U6` |
| `physical_coordinate_label` | `()` | `<U18` |
| `physical_parameter_names` | `(17,)` | `<U12` |
| `physical_parameter_units` | `(17,)` | `<U68` |
| `observation_order` | `(21,)` | `<U11` |
| `observation_units` | `()` | `<U6` |
| `seed_table` | `(100, 2)` | `int64` |
| `diagnostic_catalogue_json` | `(11,)` | `<U309` |
| `status_legends_json` | `()` | `<U512` |
| `scenario_sha256` | `()` | `<U64` |
| `runner_lock_sha256` | `()` | `<U64` |

## summary.csv

Frequency and CI columns contain **proportions**, not percentages. The figure/report multiply these by100. CI is exact equal-tailed 95% Clopper–Pearson for k out of n_valid available decisions. `frequency_all_min=k/N`, `frequency_all_max=(k+n_fail)/N` are missing-outcome bounds, not confidence intervals. The interval's conditional interpretation among available decisions does not make assumptions about failed decisions.

Mean and empirical SD use all successful finite free estimates, independent of LC availability or detection. SD uses denominator n_available_estimate−1. LC quantiles use all entries with LC_status=1 and finite nonnegative LC (NumPy linear interpolation). `n_covariance_unavailable_decision` counts rows having an available decision but unavailable ordinary free-fit covariance.

Batch-local diagnostic codes are remapped to a single global catalogue during merging; the corresponding messages are unchanged. Every batch hash is recorded in summary.json. Original batch JSON metadata are included in `batch_metadata/`; separate NPZ batch files are omitted because all their rows are retained in replicates.npz. References to `points/batch_NNN.*` in source provenance describe original run locations. `python analyze.py --from-merged` needs no points directory. Full per-replicate covariance matrices are intentionally absent.
