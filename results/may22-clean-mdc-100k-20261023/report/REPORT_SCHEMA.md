# Input contract for the single-configuration report

The renderer never fits spectra, generates observations, calibrates thresholds,
or copies results from the earlier May study. It requires `report_data.json`
and `histogram_values.npz` in this directory. The parent analysis creates these
only after the new experiment has been completed and independently checked.

`report_data.json` uses these fields:

- `verification` and `histogram_verification`: both `{status: "passed", ...}`;
  retain the full audit provenance. The adapter verifies each audit's recorded
  source hashes before constructing report inputs.
- `dataset`: `"may22"`.
- `efficiencies`: `{detector:221, isotope_order:[...], sample:[...], gasbk:[...]}`.
- `conditions`: `{times:{t_sl,t_gl,t_bl,t_sr,t_gr,t_a,t_p,tau}, effective_air_volume_m3:...}`.
- `n_by_stage`: total numbers of vector observations, including `calibration`,
  `validation`, `signal`, optional `signal_extension`, and `hist_signal`.
  The first two must each be 100000. These are independent stages. Validation
  counts are reused for the null MDC histograms and must not be counted twice.
- `n_signal_per_point`: 20000.
- `bootstrap_repetitions`: number used in the independent analysis.
- `errors`, `histogram_errors`: separate explicit arrays, empty if none. Do not
  add their lengths as if they were independent errors: validation is reused.
- `covariance_diagnostics`: optional array with `stage`, `n`,
  `wls_unavailable`, `poisson_unavailable` for unavailable free-fit covariances.
- `technical_vectors_excluded`: optional count.
- `curves`: four records, one per isotope, preserving the old summary layout:
  `dataset`, `target`, `isotope`, `nominal_LC_mBq_m3`, `nominal_MDC_mBq_m3`,
  `methods.wls` and `methods.poisson`.
- Each method record includes:
  `calibration.{c,n,k_order,ci95_c}`;
  `validation.{standard,calibrated}`;
  `formal_points[]` with `factor`, `activity_mBq_m3`, `standard`, `calibrated`;
  `standard_ld_mBq_m3`, `calibrated_ld_mBq_m3`;
  `standard_ci95_mBq_m3`, `conditional_ci95_mBq_m3`, `total_ci95_mBq_m3`;
  optional `bootstrap.{standard,conditional,total}` preserving all censoring.
- Every probability record has `n` (attempts), `n_valid`, `missing`, `k`, `rate`,
  `ci95`; if missing results exist, also `conservative_ci` or `bounds`.
- `histograms.curves`: eight records, one per method/target, in the previous
  histogram summary format: `dataset`, `family`, `target`, `isotope`,
  `nominal_MDC_mBq_m3`, `signal_A_mBq_m3`, `cutoff_c`, `null`, `signal`.
  Each condition has `n_attempted`, `n_valid`, `n_missing`, `mean`, `median`,
  `sd`, `q025`, `q975`. The signal condition also has
  `detection.calibrated` (`n_attempted` or `n`, `n_valid`, `n_missing` or
  `missing`, `k`, `rate`, `ci95`) for the independent power check at the fixed
  previously estimated Ld95. These observations must not be used to retune c/Ld.
- `histograms.values_sha256`: SHA-256 of `histogram_values.npz`.
- `source_hashes`: optional provenance dictionary, retained in the report bundle.

The compact NPZ has 16 one-dimensional arrays, named
`may22_{wls|poisson}_i{1|2|3|4}_{null|signal}`. Values are **mBq/m³**, including
NaN for unavailable attempts. Null arrays contain 100000 attempts; signal arrays
contain 20000. All valid values are included regardless of detection.

The renderer verifies the counts, values SHA-256, target mapping and histogram
summary statistics. Missing/censored results are displayed explicitly; no
threshold, probability or MDC is silently replaced. Point estimates and
intervals may use `null` for a missing boundary, which is shown as unavailable.
