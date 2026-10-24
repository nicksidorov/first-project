# Saved A/B comparison arrays

Load with `numpy.load('replicates.npz', allow_pickle=False)`. All 100000 source rows are retained in original replicate_id order. No observations were generated in this comparison.

`x` and IDs are shared once. Prefixes A_ and B_ select processing model. `wls` is LS and `poisson` is free Poisson ML. `*_concentrations` and `*_LC` contain four columns [Xe135,Xe133,Xe131m,Xe133m] in mBq/m3. The full 17-column `*_auxiliary_physical` arrays retain native mixed units [a_V;gamma;zeta], specified by physical_parameter_units. Their four fresh Xe columns1:5 are in Bq/m3. Negative free estimates are retained.

The separately downloadable `full_free_parameters.npz` contains A_wls_free_physical, A_poisson_free_physical, B_wls_free_physical, B_poisson_free_physical, each 100000 × 17 in the same native mixed units, plus replicate_id, parameter names/units and scenario hashes. Fresh Xe columns 1:5 multiplied by 1000 exactly equal the main archive's four concentration columns. This supplemental file is outside the main ZIP to satisfy publication file-size limits; no numerical data or precision have been discarded. Rebuilding requested tables needs only the main file.

Each variant has its own Unicode JSON diagnostic catalogue and status legends. Diagnostic codes index that variant's catalogue. A codes are unchanged; B batch-local codes were remapped to one global B catalogue. Decision−1 means unavailable,0 means no detection,1 means detection. Unavailable decisions never count as non-detections. Returned numerical values from unsuccessful fits remain present with non-success statuses.

## Arrays

| Key | Shape | dtype |
|---|---|---|
| `x` | `(100000, 21)` | `int64` |
| `replicate_id` | `(100000,)` | `int64` |
| `chunk` | `(100000,)` | `int16` |
| `row_in_batch` | `(100000,)` | `int16` |
| `isotope_order` | `(4,)` | `<U6` |
| `concentration_units` | `()` | `<U6` |
| `physical_coordinate_label` | `()` | `<U18` |
| `physical_parameter_names` | `(17,)` | `<U12` |
| `physical_parameter_units` | `(17,)` | `<U68` |
| `observation_order` | `(21,)` | `<U11` |
| `observation_units` | `()` | `<U6` |
| `seed_table` | `(100, 2)` | `int64` |
| `A_poisson_LC` | `(100000, 4)` | `float64` |
| `A_poisson_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `A_poisson_LC_status` | `(100000, 4)` | `int8` |
| `A_poisson_auxiliary_condition_number` | `(100000,)` | `float64` |
| `A_poisson_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `A_poisson_auxiliary_iterations` | `(100000,)` | `int32` |
| `A_poisson_auxiliary_objective` | `(100000,)` | `float64` |
| `A_poisson_auxiliary_physical` | `(100000, 17)` | `float64` |
| `A_poisson_auxiliary_rank` | `(100000,)` | `int16` |
| `A_poisson_auxiliary_status` | `(100000,)` | `int8` |
| `A_poisson_concentrations` | `(100000, 4)` | `float64` |
| `A_poisson_decision` | `(100000, 4)` | `int8` |
| `A_poisson_free_condition_number` | `(100000,)` | `float64` |
| `A_poisson_free_covariance_available` | `(100000,)` | `bool` |
| `A_poisson_free_diagnostic_code` | `(100000,)` | `int32` |
| `A_poisson_free_iterations` | `(100000,)` | `int32` |
| `A_poisson_free_objective` | `(100000,)` | `float64` |
| `A_poisson_free_rank` | `(100000,)` | `int16` |
| `A_poisson_free_status` | `(100000,)` | `int8` |
| `A_wls_LC` | `(100000, 4)` | `float64` |
| `A_wls_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `A_wls_LC_status` | `(100000, 4)` | `int8` |
| `A_wls_auxiliary_condition_number` | `(100000,)` | `float64` |
| `A_wls_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `A_wls_auxiliary_iterations` | `(100000,)` | `int32` |
| `A_wls_auxiliary_objective` | `(100000,)` | `float64` |
| `A_wls_auxiliary_physical` | `(100000, 17)` | `float64` |
| `A_wls_auxiliary_rank` | `(100000,)` | `int16` |
| `A_wls_auxiliary_status` | `(100000,)` | `int8` |
| `A_wls_concentrations` | `(100000, 4)` | `float64` |
| `A_wls_decision` | `(100000, 4)` | `int8` |
| `A_wls_free_condition_number` | `(100000,)` | `float64` |
| `A_wls_free_covariance_available` | `(100000,)` | `bool` |
| `A_wls_free_diagnostic_code` | `(100000,)` | `int32` |
| `A_wls_free_iterations` | `(100000,)` | `int32` |
| `A_wls_free_objective` | `(100000,)` | `float64` |
| `A_wls_free_rank` | `(100000,)` | `int16` |
| `A_wls_free_status` | `(100000,)` | `int8` |
| `B_wls_free_status` | `(100000,)` | `int8` |
| `B_wls_free_diagnostic_code` | `(100000,)` | `int32` |
| `B_wls_free_iterations` | `(100000,)` | `int32` |
| `B_wls_free_objective` | `(100000,)` | `float64` |
| `B_wls_free_rank` | `(100000,)` | `int16` |
| `B_wls_free_condition_number` | `(100000,)` | `float64` |
| `B_wls_auxiliary_physical` | `(100000, 17)` | `float64` |
| `B_wls_auxiliary_status` | `(100000,)` | `int8` |
| `B_wls_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `B_wls_auxiliary_iterations` | `(100000,)` | `int32` |
| `B_wls_auxiliary_objective` | `(100000,)` | `float64` |
| `B_wls_auxiliary_rank` | `(100000,)` | `int16` |
| `B_wls_auxiliary_condition_number` | `(100000,)` | `float64` |
| `B_wls_free_covariance_available` | `(100000,)` | `bool` |
| `B_wls_concentrations` | `(100000, 4)` | `float64` |
| `B_wls_LC` | `(100000, 4)` | `float64` |
| `B_wls_LC_status` | `(100000, 4)` | `int8` |
| `B_wls_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `B_wls_decision` | `(100000, 4)` | `int8` |
| `B_poisson_free_status` | `(100000,)` | `int8` |
| `B_poisson_free_diagnostic_code` | `(100000,)` | `int32` |
| `B_poisson_free_iterations` | `(100000,)` | `int32` |
| `B_poisson_free_objective` | `(100000,)` | `float64` |
| `B_poisson_free_rank` | `(100000,)` | `int16` |
| `B_poisson_free_condition_number` | `(100000,)` | `float64` |
| `B_poisson_auxiliary_physical` | `(100000, 17)` | `float64` |
| `B_poisson_auxiliary_status` | `(100000,)` | `int8` |
| `B_poisson_auxiliary_diagnostic_code` | `(100000,)` | `int32` |
| `B_poisson_auxiliary_iterations` | `(100000,)` | `int32` |
| `B_poisson_auxiliary_objective` | `(100000,)` | `float64` |
| `B_poisson_auxiliary_rank` | `(100000,)` | `int16` |
| `B_poisson_auxiliary_condition_number` | `(100000,)` | `float64` |
| `B_poisson_free_covariance_available` | `(100000,)` | `bool` |
| `B_poisson_concentrations` | `(100000, 4)` | `float64` |
| `B_poisson_LC` | `(100000, 4)` | `float64` |
| `B_poisson_LC_status` | `(100000, 4)` | `int8` |
| `B_poisson_LC_diagnostic_code` | `(100000, 4)` | `int32` |
| `B_poisson_decision` | `(100000, 4)` | `int8` |
| `variant_order` | `(2,)` | `<U1` |
| `A_diagnostic_catalogue_json` | `(11,)` | `<U309` |
| `B_diagnostic_catalogue_json` | `(35,)` | `<U275` |
| `A_status_legends_json` | `()` | `<U512` |
| `B_status_legends_json` | `()` | `<U512` |
| `A_source_replicates_sha256` | `()` | `<U64` |
| `A_scenario_sha256` | `()` | `<U64` |
| `B_scenario_sha256` | `()` | `<U64` |
| `A_runner_lock_sha256` | `()` | `<U64` |
| `B_runner_lock_sha256` | `()` | `<U64` |
| `observations_sha256` | `()` | `<U64` |
| `newly_generated_observation_vectors` | `()` | `int64` |
| `generation_model_variant` | `()` | `<U1` |
| `full_free_parameters_sha256` | `()` | `<U64` |

## Tables

summary.csv has 16 rows. Frequency/CI/bounds fields are proportions. n_generated denotes the size of the ORIGINAL generated ensemble, n_new_generated=0 for both variants in the current comparison. n_reused=n_observations=100000. Exact equal-tailed 95% Clopper–Pearson intervals use available decisions only. Missing-outcome bounds use allN. Mean/SD use all successful finite free estimates independently of LC; SD uses ddof1. LC quantiles use all finite available LC and NumPy linear interpolation.

paired_comparison.csv uses common available decisions for its frequency difference and 2×2 table. n01 means A=0,B=1; n10 means A=1,B=0. delta_frequency_percentage_points=100*(n01−n10)/n_common_valid_decisions. Estimate and LC differences each use their own common-availability mask, with separately recorded denominators and empirical SD(ddof1). These paired quantities are descriptive; no hypothesis tests or extra studies are run.

Full per-realization covariance matrices are intentionally absent. A missing free-fit covariance alone does not invalidate its decision.
