# Independent statistical design review

Reviewed the previous clean-background `study.py`, `manifest.json`, protocol, saved power-grid design and report code. The proposed calibration is appropriate for the stated conditional clean-background experiment.

## Model and decision rule

- Keep the complete 21-observation, 17-parameter model. Generating sample activities and memory are zero; generating zeta is the original DETBK ROI vector. All three spectra are generated anew and all parameters remain estimated.
- Keep the same May times, effective-volume convention, fixed ROI/efficiencies, matrices and physical coordinates. The calibration pertains to this particular background and measurement protocol.
- For each trial and target, use the free estimate and the freshly recomputed standard threshold: `R = A_hat_free / Lc_standard`. Lc comes from the existing nonnegative pilot with removal of the target fresh contribution. Do not replace the random Lc by its mean.
- Calibrate separately for dataset, isotope and method. Freeze the resulting coefficient c; the new rule is `R > c`, equivalently `A_hat_free > c * Lc_standard`. The Ld value is then found anew from power; it is not generally c times the old Ld.
- The 5% target is per isotope, not the probability of any false xenon declaration among four tests.

## Calibration and held-out validation

Use 20000 fresh calibration vectors and 10000 independent validation vectors per dataset. Both estimators and all four targets use the same vectors, retaining their dependence for comparisons. Preserve separate seeds/stages. Save c before reading validation outcomes, and do not tune c using that validation series.

For n calibration ratios in increasing order, choose the 1-based rank

`k = ceil((n + 1) * 0.95)`; `c = R_sorted[k - 1]`.

For n=20000 this is k=19001. With an independent future observation and exchangeability, strict rejection `R > c` has marginal probability at most 0.05; without ties it is 1000/20001. This is a statement averaged over calibration samples, not a guarantee that every realized coefficient has conditional false-positive probability exactly 0.05. The training exceedance frequency is largely fixed by construction and is not validation evidence.

Report tie counts at c. Strict `>` is conservative when there are atoms; do not add jitter. If ties are appreciable, bootstrap-quantile regularity and exact 5% attainment require separate discussion. All intended vectors must retain their outcome/error records; silently excluding failed fits changes the target population.

Report the independent validation frequency and a binomial interval, together with the uncalibrated frequency on these same held-out vectors. If exactly 500/10000 validate positive, the Clopper–Pearson 95% interval is approximately 4.581–5.445%. Sampling variation alone prevents promising numerically 5.00%.

## Calibration uncertainty

A nonparametric interval for the population .95 quantile uses order statistics. For desired confidence 1-delta, define

`l = binom.ppf(delta/2, n, .95)` and `u = binom.ppf(1-delta/2, n, .95) + 1`.

The bounds are `R_sorted[l-1]` and `R_sorted[u-1]`, with suitable infinite conventions only if an endpoint is outside 1..n. At n=20000:

- 95% interval: ranks [18939, 19061], actual binomial coverage approximately .95221;
- 97.5% interval: ranks [18930, 19069], coverage approximately .97584.

For a continuous null CDF, across repeated calibration datasets the realized conditional false-positive rate has distribution Beta(1000,19001). Its central 95% range is approximately 4.702–5.306%. This quantifies calibration variation; it is not an additional measurement of the actual frozen coefficient's level.

## Reusing signal trials and finding Ld95

The existing 75000 confirmation/extension signal vectors are independent of the fresh null calibration and can be reclassified at the frozen c. Exclude the original 500-per-point screening series from formal interpolation, as planned. Keep every original formal point. Disclose that the earlier grid was selected/adapted for the uncalibrated rule.

Use independent additional points if the calibrated curve or relevant uncertainty interval is not bracketed. Select and record each additional activity before generating its vectors. Do not rerun solver fits merely to change c: saved free estimates and standard Lc contain all the necessary information.

Distinguish:

1. Conditional uncertainty of Ld(c_hat), with the fitted coefficient held fixed.
2. Uncertainty in the size-calibrated target Ld(c0), including finite calibration of the population cutoff c0.

For an efficient paired bootstrap, resample calibration rows jointly across methods/isotopes to obtain c_b. Use the same c_b across all signal activities within replicate b. At each signal activity, classify the existing paired WLS/ML ratios at those thresholds, count the four joint binary states 00/01/10/11, and draw one multinomial vector of size N from those state proportions. This is equivalent to row resampling for the paired decisions, while avoiding an explicit nested bootstrap of all signal vectors. For the conditional interval use fixed c_hat instead. Preserve out-of-grid bootstrap roots as censored observations; do not discard them before calculating quantiles.

A cheap conservative interval including calibration uncertainty is also possible. Take a 97.5% order-statistic interval [cL,cU] for c0. Evaluate signal power at both endpoints with a 97.5% simultaneous binomial band, using a fixed maximum grid/endpoints budget. Under monotone true power, the largest activity whose upper power bound at cL is below .95 is a lower Ld bound; the smallest activity whose lower power bound at cU exceeds .95 is an upper Ld bound. The combined per-curve coverage is at least 95% by the union bound. It may be wide or unbounded. Keep interpolation/bootstrap intervals labelled approximate and conditional on the realized grid; they do not capture interpolation-shape error.

## Interpretation

The exercise supports comparisons at approximately matched 5% marginal size under the specified clean-background model. It does not establish uniform 5% control for arbitrary unknown mixtures or background changes. Report achieved held-out levels next to the calibrated Ld values. For comparisons between WLS and ML, use paired uncertainty, including calibration uncertainty, before ranking small differences in Ld.
