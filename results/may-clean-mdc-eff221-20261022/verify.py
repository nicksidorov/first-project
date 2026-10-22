"""Independent audit of Lc calibration, held-out level and nominal Ld95.

This module does not import the study's calibration or analysis helpers.
It rebuilds scores and decisions from stored fits, reproduces the Poisson
draws, and checks a sample of the full 17-parameter fits against the kernel.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('XEMAT_REPO', '/workspace/first-project'))
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.stats import beta, binom, norm
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from xemat.model import JointModel
from xemat.detection import compute_detection_limits
from xemat.solvers import fit

FAMILIES = ('wls', 'poisson')
PILOTS = {'wls': 'nnls', 'poisson': 'poisson_nonnegative'}
STAGE_CODES = {'calibration': 10, 'validation': 11, 'signal': 12, 'extend': 13, 'tech': 14}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key_of(meta):
    return tuple(meta[k] for k in ('dataset', 'stage', 'target', 'factor', 'chunk', 'n'))


def cp_interval(k, n):
    """Clopper--Pearson endpoints from beta quantiles, independent of binomtest."""
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k))]


def verify_rate(reported, scores, cutoff):
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    n = int(valid.sum())
    k = int(np.count_nonzero(scores[valid] > cutoff))
    assert reported['n'] == len(scores)
    assert reported.get('n_valid', n) == n
    assert reported.get('missing', len(scores)-n) == len(scores)-n
    assert reported['k'] == k
    np.testing.assert_allclose(reported['rate'], k/n, rtol=0, atol=1e-15)
    np.testing.assert_allclose(reported['ci95'], cp_interval(k, n), rtol=1e-10, atol=1e-12)


def linear_root(points, rule):
    """First upward crossing, written as a weighted activity average."""
    if points[0][rule]['rate'] > .95:
        return None
    if points[0][rule]['rate'] == .95:
        return points[0]['activity_mBq_m3']
    for left, right in zip(points[:-1], points[1:]):
        p, q = left[rule]['rate'], right[rule]['rate']
        if p <= .95 <= q and q > p:
            return (left['activity_mBq_m3']*(q-.95)
                    + right['activity_mBq_m3']*(.95-p))/(q-p)
    return None


def assert_endpoints(actual, expected, rtol=1e-11):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            assert a is not None
            np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-14)


def independent_grid_bounds(scores_by_factor, null_scores, cutoff_low, cutoff_high, nominal, total):
    confidence = 1-.025/18 if total else 1-.05/9
    tail = (1-confidence)/2
    lo, hi = 0., None
    for factor, scores in [(0., null_scores), *scores_by_factor]:
        finite = np.isfinite(scores)
        N, n = len(scores), int(finite.sum())
        k_liberal = int(np.count_nonzero(scores[finite] > cutoff_low))
        k_strict = int(np.count_nonzero(scores[finite] > cutoff_high))
        # Allow every completion of missing decisions in the conservative bound.
        high_k = k_liberal+N-n
        liberal_upper = 1. if high_k == N else beta.ppf(1-tail, high_k+1, N-high_k)
        strict_lower = 0. if k_strict == 0 else beta.ppf(tail, k_strict, N-k_strict+1)
        if liberal_upper < .95:
            lo = max(lo, factor)
        if strict_lower > .95:
            hi = factor if hi is None else min(hi, factor)
    return [lo*nominal, None if hi is None else hi*nominal]


def independent_bootstrap_curve(curve, signal_scores, calibration_bootstrap, master_seed):
    """Audit two complete paired curves using categorical enumeration.

    The production analysis obtains category sizes via marginal/both sums.
    Here direct bincount enumeration provides an independent calculation.
    """
    dataset, target = curve['dataset'], curve['target']
    dataset_index = ('may22', 'may26').index(dataset)
    points = curve['methods']['wls']['formal_points']
    factors = [point['factor'] for point in points]
    nominal = curve['nominal_MDC_mBq_m3']
    fitted = np.array([curve['methods'][f]['calibration']['c'] for f in FAMILIES])
    varying = calibration_bootstrap[dataset][:, :, target-1]
    repetitions = len(varying)
    result = {}
    for mode, code in [('standard', 3), ('conditional', 1), ('total', 2)]:
        fixed = np.ones(2) if mode == 'standard' else fitted
        cuts = np.broadcast_to(fixed, (repetitions, 2)) if mode != 'total' else varying
        rng = np.random.default_rng([master_seed, dataset_index, target, code])
        simulated = []
        for factor in factors:
            scores = np.stack([signal_scores[(dataset, target, factor, f)] for f in FAMILIES], axis=1)
            assert np.all(np.isfinite(scores)), 'Paired-bootstrap audit assumes no missing scores'
            if mode != 'total':
                decisions = scores > fixed
                category = 2*decisions[:, 0].astype(int)+decisions[:, 1].astype(int)
                counts = np.broadcast_to(np.bincount(category, minlength=4), (repetitions, 4))
            else:
                counts = []
                for cut in cuts:
                    decisions = scores > cut
                    category = 2*decisions[:, 0].astype(int)+decisions[:, 1].astype(int)
                    counts.append(np.bincount(category, minlength=4))
                counts = np.asarray(counts)
            draws = rng.multinomial(len(scores), counts/len(scores))
            simulated.append(np.column_stack([draws[:, 2]+draws[:, 3], draws[:, 1]+draws[:, 3]])/len(scores))
        powers = np.stack(simulated, axis=1)
        roots_by_method = []
        for j, family in enumerate(FAMILIES):
            roots = []
            for row in powers[:, :, j]:
                value = None
                if row[0] > .95:
                    value = -np.inf
                elif row[0] == .95:
                    value = factors[0]
                else:
                    for index in range(len(factors)-1):
                        p, q = row[index:index+2]
                        if p <= .95 <= q and q > p:
                            value = (factors[index]*(q-.95)+factors[index+1]*(.95-p))/(q-p)
                            break
                    if value is None:
                        value = np.inf if np.all(row < .95) else np.nan
                roots.append(value)
            roots = np.asarray(roots)
            roots_by_method.append(roots)
            metadata = curve['methods'][family]['bootstrap'][mode]
            assert metadata['n_repetitions'] == repetitions
            assert metadata['n_crossings'] == np.isfinite(roots).sum()
            assert metadata['n_below_grid'] == np.isneginf(roots).sum()
            assert metadata['n_above_grid'] == np.isposinf(roots).sum()
            assert metadata['n_ambiguous'] == np.isnan(roots).sum()
            if np.any(np.isnan(roots)):
                expected_ci = [None, None]
            else:
                percentile = np.quantile(roots, [.025, .975], method='inverted_cdf')
                expected_ci = [float(v)*nominal if np.isfinite(v) else None for v in percentile]
            assert_endpoints(curve['methods'][family][mode+'_ci95_mBq_m3'], expected_ci)
        wls, poisson = roots_by_method
        finite = np.isfinite(wls) & np.isfinite(poisson)
        difference = np.zeros(repetitions)
        difference[finite] = (wls[finite]-poisson[finite])*nominal
        lower = np.where(finite, difference, -np.inf)
        upper = np.where(finite, difference, np.inf)
        bounds = [np.quantile(lower, .025, method='inverted_cdf'),
                  np.quantile(upper, .975, method='inverted_cdf')]
        bounds = [float(v) if np.isfinite(v) else None for v in bounds]
        if mode != 'standard':
            pair = curve['paired_ld_difference']
            assert pair[mode+'_n_finite'] == finite.sum()
            assert_endpoints(pair[mode+'_ci95'], bounds)
        result[mode] = dict(repetitions=repetitions, paired_finite=int(finite.sum()))
    return dict(dataset=dataset, target=target, modes=result)


def qr_covariance(S, mu):
    """Full Fisher inverse via scaled QR, independently of production SVD."""
    assert np.all(mu > 0)
    weighted = S / np.sqrt(mu)[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    assert np.all(scale > 0)
    _, R = np.linalg.qr(weighted/scale, mode='reduced')
    inverse = solve_triangular(R, np.eye(S.shape[1]))
    return (inverse @ inverse.T) / scale[:, None] / scale[None, :]


def verify_inputs():
    """Verify new calibration and independently solve all eight formula limits."""
    manifest = read_json(ROOT/'manifest.json')
    for name, digest in manifest['production_sha256'].items():
        assert sha256(REPO/name) == digest, name
    for name, digest in manifest['prior_document_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    old = read_json(ROOT/'prior_calibrated_manifest.json')
    old_formula = read_json(ROOT/'prior_formula_manifest.json')
    assert manifest['systematics'] is False
    assert list(manifest['datasets']) == ['may22', 'may26']
    epsilon = np.array([0.241169, 0.367784, 0.233937, 0.233937])
    branching = np.array([.902, .370, .540, .5577])
    assert manifest['efficiency_update']['detector'] == 221
    assert manifest['master_seed'] not in (old['master_seed'], old_formula['master_seed'])
    models, audits = {}, []
    original_inputs_verified = 0
    for dataset, data in manifest['datasets'].items():
        prior = old['datasets'][dataset]
        metadata = data['metadata']
        for field in ('sample_temporal', 'background_temporal', 'detector_temporal',
                      'time_parameters', 'xenon_volume_cm3', 'effective_air_volume_m3',
                      'sample_roi_numbers', 'gasbk_roi_numbers', 'detector_roi_numbers',
                      'detector_roi_counts', 'kappa_s', 'kappa_g', 'radon_mode', 'radon_h'):
            assert metadata[field] == prior['metadata'][field], (dataset, field)
        assert data['input_sha256'] == prior['input_sha256']
        for key, path in data['input_paths'].items():
            if Path(path).is_file():
                assert sha256(path) == data['input_sha256'][key]
                original_inputs_verified += 1
        for field in ('sample_efficiencies', 'gasbk_efficiencies'):
            np.testing.assert_array_equal(metadata[field], epsilon)
        cal = {k: np.asarray(v) for k, v in data['calibration_matrices'].items()}
        oldcal = {k: np.asarray(v) for k, v in prior['calibration_matrices'].items()}
        for key in ('R_c0', 'R_g0', 'B', 'Gamma', 'G_p', 'G_s', 'V0', 'kappa_s', 'kappa_g'):
            np.testing.assert_array_equal(cal[key], oldcal[key])
        for key in ('I_c', 'I_g'):
            np.testing.assert_allclose(cal[key][1:], epsilon*branching, rtol=0, atol=1e-16)
            np.testing.assert_array_equal(cal[key][0], oldcal[key][0])
        # Separate algebra checks on the changed efficiencies and chain ratio.
        for key, sensitivity in [('H', 'I_c'), ('H_g1', 'I_g'), ('H_g2', 'I_g')]:
            expected = oldcal[key] * (oldcal[sensitivity]/cal[sensitivity])[None, :]
            np.testing.assert_allclose(cal[key], expected, rtol=1e-13, atol=1e-15)
        for key, sensitivity in [('C_s_c', 'I_c'), ('C_s_g', 'I_g'), ('C_g_g', 'I_g')]:
            expected = oldcal[key].copy()
            expected[2, 4] *= ((cal[sensitivity][2]/cal[sensitivity][4])
                                /(oldcal[sensitivity][2]/oldcal[sensitivity][4]))
            np.testing.assert_allclose(cal[key], expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(cal['F'], np.linalg.solve(cal['H_g1'], cal['B']@cal['H_g2']), rtol=1e-13)
        model = JointModel(**data['model'])
        S = model.physical_matrix
        assert S.shape == (21, 17)
        assert np.linalg.matrix_rank(S/np.linalg.norm(S, axis=0)) == 17
        np.testing.assert_array_equal(model.x, prior['model']['x'])
        np.testing.assert_array_equal(model.b, 0)
        expected_M = np.block([
            [cal['R_c0']@cal['C_s_c'], cal['R_g0']@cal['C_s_g']@cal['F'], metadata['kappa_s']*np.eye(7)],
            [np.zeros((7, 5)), cal['R_g0']@cal['C_g_g'], metadata['kappa_g']*np.eye(7)],
            [np.zeros((7, 5)), np.zeros((7, 5)), np.eye(7)],
        ])
        np.testing.assert_allclose(model.M, expected_M, rtol=1e-13, atol=1e-15)
        expected_T = cal['G_s']@cal['G_p']@cal['Gamma']@cal['H']/cal['V0']
        np.testing.assert_allclose(model.T_c, expected_T, rtol=1e-13, atol=1e-15)
        truth = np.asarray(data['truth0'])
        np.testing.assert_array_equal(truth[:10], 0)
        np.testing.assert_array_equal(truth[10:], [522, 854, 494, 66, 64, 89, 97])
        np.testing.assert_array_equal(truth[10:], model.x[14:])
        np.testing.assert_array_equal(S[14:, 10:], np.eye(7))
        np.testing.assert_array_equal(S[7:, :5], 0)
        mu0 = S@truth + model.b
        np.testing.assert_allclose(mu0, data['mu0'], rtol=1e-13)
        np.testing.assert_allclose(mu0, prior['mu0'], rtol=0, atol=1e-13)
        K0 = qr_covariance(S, mu0)
        z_alpha = norm.isf(manifest['alpha'])
        z_beta = norm.isf(manifest['beta_ii'])
        for target in (1, 2, 3, 4):
            sigma0 = np.sqrt(K0[target, target])
            lc = z_alpha*sigma0
            def residual(activity):
                return activity-lc-z_beta*np.sqrt(qr_covariance(S, mu0+activity*S[:, target])[target, target])
            upper = 2*lc
            while residual(upper) <= 0:
                upper *= 2
            ld = brentq(residual, 0., upper, xtol=1e-15, rtol=2e-12)
            index = target-1
            np.testing.assert_allclose(data['nominal_sigma0'][index], sigma0, rtol=2e-11)
            np.testing.assert_allclose(data['nominal_LC'][index], lc, rtol=2e-11)
            np.testing.assert_allclose(data['nominal_MDC'][index], ld, rtol=3e-8)
            np.testing.assert_allclose(data['nominal_sigma_at_MDC'][index],
                np.sqrt(qr_covariance(S, mu0+ld*S[:, target])[target, target]), rtol=3e-8)
            audits.append(dict(dataset=dataset, target=target, LC_Bq_m3=lc, MDC_Bq_m3=ld,
                               relative_equation_residual=abs(residual(ld))/ld))
        models[dataset] = model
    return manifest, models, dict(formula_checks=audits, original_input_files_verified=original_inputs_verified,
        all_models_rank=17, updated_efficiencies_verified=True, original_times_volumes_R_DETBK_preserved=True)




def verify(*, output=None, portable=False):
    # All observations are new and stored in this directory. The compatibility
    # --portable flag simply confirms no prior raw study is needed.
    manifest, models, input_audit = verify_inputs()
    old_manifest = read_json(ROOT/'prior_calibrated_manifest.json')
    old_summary = read_json(ROOT/'prior_calibrated_summary.json')
    calibration = read_json(ROOT/'calibration.json')
    summary = read_json(ROOT/'summary.json')
    cutoff_marker = read_json(ROOT/'cutoff.json')
    assert cutoff_marker['frozen'] is True and cutoff_marker['validation_not_used'] is True
    assert calibration['validation_not_used'] is True
    assert cutoff_marker['calibration_sha256'] == sha256(ROOT/'calibration.json')
    assert cutoff_marker['bootstrap_sha256'] == sha256(ROOT/'calibration_bootstrap.npz')
    assert calibration['bootstrap']['sha256'] == sha256(ROOT/'calibration_bootstrap.npz')
    for name, digest in calibration['source_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    plans = [ROOT/'calibration_plan.json', ROOT/'validation_plan.json', ROOT/'signal_plan.json']
    plans.extend(sorted(ROOT.glob('extension_plan*.json')))
    planned = [job for path in plans for job in read_json(path)]
    expected_keys = {key_of(job) for job in planned}
    assert len(expected_keys) == len(planned)
    new_paths = sorted((ROOT/'points').glob('*.json'))
    assert len(new_paths) == len(planned), 'Not all planned batches are complete'
    assert {p.stem for p in new_paths} == {p.stem for p in (ROOT/'points').glob('*.npz')}
    null_scores = defaultdict(list)
    signal_scores = defaultdict(list)
    new_counts = defaultdict(int)
    new_seeds = set()
    errors = []
    missing_decisions = 0
    covariance_missing = defaultdict(int)
    batches_by_dataset_stage = defaultdict(int)
    n_refits = n_fixed_iterative = 0
    seen_new_keys = set()
    # A deterministic row of every new batch is refitted in all four methods.
    def records():
        for path in new_paths:
            with np.load(path.with_suffix('.npz')) as stored:
                arrays = {key: stored[key] for key in stored.files}
            yield path, True, read_json(path), arrays, True
    for path, fresh, meta, arrays, full_vectors in records():
        model = models[meta['dataset']]
        indices = np.asarray(meta['targets'], dtype=int)
        expected_targets = [1, 2, 3, 4] if meta['target'] == 0 else [meta['target']]
        np.testing.assert_array_equal(indices, expected_targets)
        assert arrays['x'].shape == (meta['n'], 21)
        np.testing.assert_array_equal(arrays['targets'], indices)
        np.testing.assert_array_equal(arrays['indices'], np.arange(meta['n']))
        truth = np.asarray(manifest['datasets'][meta['dataset']]['truth0']).copy()
        if meta['target']:
            truth[meta['target']] = (meta['factor']
                * manifest['datasets'][meta['dataset']]['nominal_MDC'][meta['target']-1])
        np.testing.assert_array_equal(meta['truth'], truth)
        mu = model.physical_matrix@truth + model.b
        np.testing.assert_allclose(mu, meta['mu'], rtol=1e-13)
        expected_x = np.random.default_rng(meta['seed']).poisson(mu, size=arrays['x'].shape)
        np.testing.assert_array_equal(expected_x, arrays['x'])

        if fresh:
            item_key = key_of(meta)
            assert item_key in expected_keys and item_key not in seen_new_keys
            seen_new_keys.add(item_key)
            seed = tuple(meta['seed'])
            assert seed not in new_seeds
            assert seed[0] != old_manifest['master_seed']
            new_seeds.add(seed)
            expected_seed = [manifest['master_seed'], list(manifest['datasets']).index(meta['dataset']),
                             meta['target'], round(10000*meta['factor']),
                             STAGE_CODES[meta['stage']], meta['chunk']]
            assert list(seed) == expected_seed
            new_counts[meta['stage']] += meta['n']
            batches_by_dataset_stage[(meta['dataset'], meta['stage'])] += meta['n']
            assert meta['stage'] in ('calibration', 'validation', 'signal', 'extend')
            if meta['stage'] in ('calibration', 'validation'):
                assert meta['target'] == 0 and meta['factor'] == 0

        for family in FAMILIES:
            assert arrays[family+'_free_covariance_available'].shape == (meta['n'],)
            assert arrays[family+'_free_covariance_available'].dtype == np.dtype(bool)
            covariance_missing[(meta['dataset'], meta['stage'], family)] += int(
                np.count_nonzero(~arrays[family+'_free_covariance_available']))
            lc = arrays[family+'_LC']
            free = arrays[family+'_free'][:, indices] if full_vectors else arrays[family+'_free']
            saved_decisions = arrays[family+'_decision']
            valid = saved_decisions >= 0
            expected_fit_shape = (meta['n'], 17 if full_vectors else len(indices))
            assert arrays[family+'_free'].shape == arrays[family+'_positive'].shape == expected_fit_shape
            valid_rows = np.any(valid, axis=1)
            assert np.all(np.isfinite(arrays[family+'_free'][valid_rows]))
            assert np.all(np.isfinite(arrays[family+'_positive'][valid_rows]))
            assert np.all(arrays[family+'_positive'][valid_rows] >= 0)
            if family == 'poisson':
                # Converting the stored physical parameters back to expected
                # counts may introduce tiny cancellation roundoff at mu=0.
                for kind in ('free', 'positive'):
                    parameters = arrays[family+'_'+kind][valid_rows]
                    fitted_mu = parameters@model.physical_matrix.T + model.b
                    roundoff = 256*np.finfo(float).eps * (
                        np.abs(parameters)@np.abs(model.physical_matrix).T + model.b + 1)
                    assert np.all(fitted_mu >= -roundoff)
                    assert np.all(fitted_mu[arrays['x'][valid_rows] > 0] > 0)
            assert np.all(np.isfinite(lc[valid])) and np.all(lc[valid] > 0)
            assert np.all(np.isfinite(free[valid]))
            np.testing.assert_array_equal((free > lc)[valid], saved_decisions[valid])
            missing_decisions += int(np.count_nonzero(~valid))
            scores = np.divide(free, lc, out=np.full_like(free, np.nan), where=valid)
            np.testing.assert_array_equal((scores > 1)[valid], saved_decisions[valid])
            for col, target in enumerate(indices):
                score = scores[:, col]
                if fresh and meta['stage'] in ('calibration', 'validation'):
                    null_scores[(meta['dataset'], meta['stage'], family, int(target))].append((meta['chunk'], score))
                else:
                    signal_scores[(meta['dataset'], int(target), meta['factor'], family)].append(score)
        errors.extend(dict(batch=path.name, fresh=fresh, **error) for error in meta['errors'])

        if fresh:
            row = meta['n']//3
            trial = JointModel(model.M, arrays['x'][row], model.b, model.n_sample_parameters,
                               model.T_c, model.coordinates, model.n_detector_parameters)
            for family, pilot in PILOTS.items():
                fits = {}
                for label, method in [('free', family), ('positive', pilot)]:
                    result = fit(trial, method)
                    assert result.success
                    if label == 'free':
                        assert bool(arrays[family+'_free_covariance_available'][row]) == (result.physical_covariance is not None)
                    comparable = result.physical_parameters if full_vectors else result.physical_parameters[indices]
                    np.testing.assert_allclose(comparable, arrays[family+'_'+label][row],
                                               rtol=1e-8, atol=1e-10)
                    fits[label] = result.physical_parameters
                    n_refits += 1
                limits = compute_detection_limits(model.physical_matrix, model.b,
                    fits['free'], fits['positive'], component_indices=indices, mode='iterative')
                np.testing.assert_allclose(limits.L_C, arrays[family+'_LC'][row], rtol=1e-13)
                np.testing.assert_array_equal(limits.detected, arrays[family+'_decision'][row])
                n_fixed_iterative += len(indices)

    assert seen_new_keys == expected_keys
    technical_plan = read_json(ROOT/'technical_plan.json')
    assert summary['technical_vectors_excluded'] == sum(job['n'] for job in technical_plan)
    null_scores = {k: np.concatenate([score for _, score in sorted(v, key=lambda item: item[0])])
                   for k, v in null_scores.items()}
    signal_scores = {k: np.concatenate(v) for k, v in signal_scores.items()}
    calibrated = {}
    calibration_audit = []
    for dataset in manifest['datasets']:
        for family in FAMILIES:
            for target in (1, 2, 3, 4):
                item = calibration['datasets'][dataset]['methods'][family][target-1]
                assert item['target'] == target
                scores = null_scores[(dataset, 'calibration', family, target)]
                assert len(scores) == manifest['calibration_n_per_dataset'] == 20000
                assert np.all(np.isfinite(scores)), 'Calibration contains unavailable scores'
                ordered = np.sort(scores)
                n = len(scores)
                rank = int(np.ceil((n+1)*.95))
                cutoff = ordered[rank-1]
                assert item['n'] == n and item['k_order'] == rank == 19001
                np.testing.assert_allclose(item['c'], cutoff, rtol=0, atol=0)
                np.testing.assert_allclose(cutoff_marker['coefficients'][dataset][family][target-1],
                                           cutoff, rtol=0, atol=0)
                assert cutoff > 0
                ties = int(np.count_nonzero(scores == cutoff))
                assert item['ties'] == ties
                training_rate = float(np.mean(scores > cutoff))
                np.testing.assert_allclose(item['achieved_training_rate'], training_rate, rtol=0, atol=1e-15)
                lower_rank = int(binom.ppf(.025, n, .95))
                upper_rank = int(binom.ppf(.975, n, .95)) + 1
                expected_ci = [ordered[lower_rank-1], ordered[upper_rank-1]]
                np.testing.assert_allclose(item['ci95_c'], expected_ci, rtol=0, atol=0)
                lower_rank975 = int(binom.ppf(.0125, n, .95))
                upper_rank975 = int(binom.ppf(.9875, n, .95)) + 1
                np.testing.assert_allclose(item['ci975_c'],
                    [ordered[lower_rank975-1], ordered[upper_rank975-1]], rtol=0, atol=0)
                alpha_content_ci = beta.ppf([.025, .975], n+1-rank, rank)
                np.testing.assert_allclose(item['alpha_interval_from_calibration'], alpha_content_ci,
                                           rtol=1e-12, atol=1e-15)
                calibrated[(dataset, family, target)] = float(cutoff)
                calibration_audit.append(dict(dataset=dataset, family=family, target=target,
                    n=n, rank=rank, c=float(cutoff), ties=ties, training_exceedances=int(np.sum(scores > cutoff))))

    # Recompute the first three common-row calibration bootstrap resamples using
    # full sorting, independently of calibration.py's chunked partition code.
    n_calibration_bootstrap_checks = 0
    with np.load(ROOT/'calibration_bootstrap.npz') as saved_bootstrap:
        for dataset_index, dataset in enumerate(manifest['datasets']):
            ratios = np.stack([np.stack([null_scores[(dataset, 'calibration', family, target)]
                                         for target in (1, 2, 3, 4)], axis=1)
                               for family in FAMILIES], axis=1)
            assert ratios.shape == (20000, 2, 4)
            seed = [calibration['bootstrap']['master_seed'], dataset_index, 0]
            rng = np.random.default_rng(seed)
            assert saved_bootstrap[dataset].shape == (calibration['bootstrap']['repetitions'], 2, 4)
            for row in range(3):
                sample_indices = rng.integers(0, len(ratios), size=len(ratios))
                independently_sorted = np.sort(ratios[sample_indices], axis=0)
                np.testing.assert_array_equal(independently_sorted[19000], saved_bootstrap[dataset][row])
                n_calibration_bootstrap_checks += 8

    assert summary['n_reused_signal'] == 0
    assert summary['n_new_by_stage']['calibration'] == new_counts['calibration'] == 40000
    assert summary['n_new_by_stage']['validation'] == new_counts['validation'] == 20000
    assert summary['n_new_by_stage'].get('signal_extension', 0) == new_counts.get('extend', 0)
    assert summary['n_new_by_stage']['signal'] == new_counts['signal']
    covariance_rows = {
        (row['dataset'], row['stage']): row for row in summary['covariance_diagnostics']
    }
    assert len(covariance_rows) == len(summary['covariance_diagnostics'])
    assert set(covariance_rows) == set(batches_by_dataset_stage)
    for (dataset, stage), n in batches_by_dataset_stage.items():
        row = covariance_rows[(dataset, stage)]
        assert row['n'] == n
        for family in FAMILIES:
            assert row[family+'_unavailable'] == covariance_missing[(dataset, stage, family)]
    assert summary.get('needs_extension') == []
    assert len(summary['curves']) == 8
    n_rates = n_roots = 0
    for curve in summary['curves']:
        dataset, target = curve['dataset'], curve['target']
        nominal = manifest['datasets'][dataset]['nominal_MDC'][target-1]*1000
        np.testing.assert_allclose(curve['nominal_MDC_mBq_m3'], nominal, rtol=0, atol=0)
        for family in FAMILIES:
            method = curve['methods'][family]
            cutoff = calibrated[(dataset, family, target)]
            np.testing.assert_allclose(method['calibration']['c'], cutoff, rtol=0, atol=0)
            saved_calibration = calibration['datasets'][dataset]['methods'][family][target-1]
            for name in ('n', 'k_order', 'ci95_c'):
                np.testing.assert_array_equal(method['calibration'][name], saved_calibration[name])
            validation = null_scores[(dataset, 'validation', family, target)]
            assert len(validation) == manifest['validation_n_per_dataset'] == 10000
            verify_rate(method['validation']['standard'], validation, 1.0)
            verify_rate(method['validation']['calibrated'], validation, cutoff)
            n_rates += 2
            points = method['formal_points']
            factors = [p['factor'] for p in points]
            assert factors == sorted(set(factors))
            expected_factors = sorted(k[2] for k in signal_scores
                                      if k[0] == dataset and k[1] == target and k[3] == family)
            assert factors == expected_factors
            for point in points:
                scores = signal_scores[(dataset, target, point['factor'], family)]
                np.testing.assert_allclose(point['activity_mBq_m3'], point['factor']*nominal, rtol=1e-13)
                verify_rate(point['standard'], scores, 1.0)
                verify_rate(point['calibrated'], scores, cutoff)
                if cutoff > 1:
                    assert point['calibrated']['k'] <= point['standard']['k']
                elif cutoff < 1:
                    assert point['calibrated']['k'] >= point['standard']['k']
                else:
                    assert point['calibrated']['k'] == point['standard']['k']
                n_rates += 2
            calibrated_root = linear_root(points, 'calibrated')
            assert calibrated_root is not None
            np.testing.assert_allclose(method['calibrated_ld_mBq_m3'], calibrated_root, rtol=1e-12)
            standard_root = linear_root(points, 'standard')
            assert standard_root is not None
            np.testing.assert_allclose(method['standard_ld_mBq_m3'], standard_root, rtol=1e-12)
            if cutoff > 1:
                assert calibrated_root >= standard_root-1e-12
            elif cutoff < 1:
                assert calibrated_root <= standard_root+1e-12
            else:
                np.testing.assert_allclose(calibrated_root, standard_root, rtol=1e-12)
            historical = next(c for c in old_summary['curves']
                              if c['dataset'] == dataset and c['target'] == target)
            if 'historical_standard_ld_mBq_m3' in method:
                np.testing.assert_allclose(method['historical_standard_ld_mBq_m3'],
                    historical['methods'][family]['standard_ld_mBq_m3'], rtol=0, atol=0)
            np.testing.assert_allclose(method['historical_calibrated_ld_mBq_m3'],
                historical['methods'][family]['calibrated_ld_mBq_m3'], rtol=0, atol=0)
            n_roots += 2
            for name in ('standard_ci95_mBq_m3', 'conditional_ci95_mBq_m3', 'total_ci95_mBq_m3'):
                interval = method.get(name)
                if interval is not None:
                    assert len(interval) == 2 and all(np.isfinite(interval))
                    assert interval[0] <= interval[1]
            score_points = [(p['factor'], signal_scores[(dataset, target, p['factor'], family)])
                            for p in points]
            standard_grid = independent_grid_bounds(score_points, validation, 1, 1, nominal, False)
            assert_endpoints(method['standard_grid_bound95_mBq_m3'], standard_grid)
            conditional_grid = independent_grid_bounds(score_points, validation, cutoff, cutoff, nominal, False)
            total_grid = independent_grid_bounds(score_points, validation,
                *saved_calibration['ci975_c'], nominal, True)
            assert_endpoints(method['conditional_grid_bound95_mBq_m3'], conditional_grid)
            assert_endpoints(method['total_grid_bound95_mBq_m3'], total_grid)
        paired = curve.get('paired_ld_difference')
        if paired is not None:
            expected_difference = (curve['methods']['wls']['calibrated_ld_mBq_m3']
                                   - curve['methods']['poisson']['calibrated_ld_mBq_m3'])
            np.testing.assert_allclose(paired['point'], expected_difference, rtol=1e-12, atol=1e-15)

    assert len(errors) == len(summary.get('errors', []))
    bootstrap_audit = []
    with np.load(ROOT/'calibration_bootstrap.npz') as stored_bootstrap:
        values = {name: stored_bootstrap[name] for name in stored_bootstrap.files}
    for dataset, target in [('may22', 2), ('may26', 4)]:
        curve = next(c for c in summary['curves'] if c['dataset'] == dataset and c['target'] == target)
        bootstrap_audit.append(independent_bootstrap_curve(curve, signal_scores, values,
            manifest['bootstrap']['master_seed']))

    result = dict(status='passed', new_vectors=sum(new_counts.values()), new_by_stage=dict(new_counts),
        reused_signal_vectors=0, all_new_batch_keys_match_plan=True,
        unique_new_seeds=len(new_seeds), old_new_master_seeds_disjoint=True,
        calibration_validation_seeds_disjoint=True, all_raw_poisson_counts_reproduced=True,
        all_full_models_rank=17, all_generating_null_activities_zero=True,
        source_background_from_DETBK_only=True, all_production_hashes_unchanged=True,
        calibrated_cutoffs_from_calibration_only=True, calibration_sha256=sha256(ROOT/'calibration.json'),
        calibration_order_statistics=calibration_audit, refitted_models=n_refits,
        paired_calibration_bootstrap_quantiles_checked=n_calibration_bootstrap_checks,
        paired_power_bootstrap_rebuilds=bootstrap_audit,
        lc_fixed_iterative_agreements=n_fixed_iterative,
        reported_rate_checks=n_rates, independent_ld_interpolations=n_roots,
        missing_decisions=missing_decisions, recorded_errors=errors,
        free_covariance_unavailable=[dict(dataset=key[0], stage=key[1], family=key[2], n=count)
                                    for key, count in sorted(covariance_missing.items())],
        cutoff_direction_invariants_checked=True, all_grid_bounds_independently_recomputed=True,
        independent_input_and_formula_checks=input_audit,
        note=('All new raw decisions, cutoffs, CP intervals, grid bounds and Ld point estimates are independently rebuilt; '
              'standard, conditional and total paired-bootstrap intervals are independently rebuilt for two representative curves. '
              'Formula limits use independent scaled QR covariance and brentq root finding. '
              'Ordinary free Poisson covariance may be unavailable even when Lc and fit remain valid.'))
    destination = Path(output) if output is not None else ROOT/'verification.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'calibration_order_statistics'},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable', action='store_true', help='All new raw observations are self-contained')
    parser.add_argument('--inputs-only', action='store_true', help='Verify frozen inputs and formula limits before MC finishes')
    parser.add_argument('--output', type=Path, help='Write audit outside the archive directory')
    args = parser.parse_args()
    if args.inputs_only:
        result = verify_inputs()[2]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        verify(output=args.output, portable=args.portable)
