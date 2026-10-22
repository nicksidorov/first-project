"""Independent audit of the per-observation MDC histogram experiment.

Does not import hist_compute, analyze_histograms, or xemat.detection. All raw
Poisson vectors are regenerated. The original null fits are compared in full.
First and last rows of every batch are refitted; MDC is then checked using
QR-based Fisher information and a bracketed scalar root, not fixed-point/SVD.
Only this auditor's JSON result is written. Locked inputs and raw data are read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.stats import beta as beta_distribution, norm

DATES = ('may22', 'may26')
FAMILIES = ('wls', 'poisson')
POSITIVE_METHOD = {'wls': 'nnls', 'poisson': 'poisson_nonnegative'}
ISOTOPES = {1: 'Xe135', 2: 'Xe133', 3: 'Xe131m', 4: 'Xe133m'}
Z95 = float(norm.ppf(.95))
ROOT_DEFAULT = Path(__file__).resolve().parent
SOURCE_DEFAULT = Path('/workspace/scratch/may-clean-mdc-eff221-20261022')
REPO_DEFAULT = Path('/workspace/first-project')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact_array(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def near(actual, expected, label, rtol=2e-12, atol=1e-14):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=False, err_msg=label)


def same_tree(actual, expected, label):
    """Tight numeric comparison, exact keys/types/counts and categorical fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key, value in expected.items():
            same_tree(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for index, value in enumerate(expected):
            same_tree(actual[index], value, f'{label}[{index}]')
    elif isinstance(expected, float):
        near(actual, expected, label, rtol=2e-11, atol=2e-12)
    else:
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def expected_plan():
    jobs = []
    for chunk in range(10):
        for dataset in DATES:
            jobs.append(dict(dataset=dataset, stage='null', target=0,
                             family='both', n=1000, chunk=chunk))
    for chunk in range(3):
        for target in range(1, 5):
            for dataset in DATES:
                for family in FAMILIES:
                    jobs.append(dict(dataset=dataset, stage='signal_ld', target=target,
                                     family=family, n=1000, chunk=chunk))
    return jobs


def stem(job):
    return '{dataset}_{stage}_i{target}_{family}_c{chunk}'.format(**job)


def qr_covariance(S, mu):
    """Inverse expected information via scaled QR and triangular solves."""
    require(np.all(np.asarray(mu) > 0), 'QR covariance needs strictly positive means')
    weighted = S / np.sqrt(mu)[:, None]
    scales = np.linalg.norm(weighted, axis=0)
    require(np.all(scales > 0), 'zero sensitivity column')
    normalized = weighted / scales
    _, R = np.linalg.qr(normalized, mode='reduced')
    require(R.shape == (S.shape[1], S.shape[1]), 'QR is not full column rank')
    require(np.all(np.abs(np.diag(R)) > np.finfo(float).eps * max(S.shape)),
            'rank deficient information')
    R_inv = solve_triangular(R, np.eye(R.shape[0]))
    inverse = R_inv @ R_inv.T
    return inverse / scales[:, None] / scales[None, :]


def independent_limits(S, b, positive, target):
    """Remove only target from the positive pilot and solve L=LC+z*sigma(L)."""
    nuisance = np.array(positive, dtype=float, copy=True)
    require(np.all(np.isfinite(nuisance)) and np.all(nuisance >= 0), 'invalid positive pilot')
    nuisance[target] = 0.
    mu0 = np.asarray(b) + S @ nuisance
    sigma0 = np.sqrt(qr_covariance(S, mu0)[target, target])
    LC = Z95 * sigma0

    def sigma(activity):
        return np.sqrt(qr_covariance(S, mu0 + activity * S[:, target])[target, target])

    def residual(activity):
        return activity - LC - Z95 * sigma(activity)

    upper = 2 * LC
    for _ in range(100):
        if residual(upper) > 0:
            break
        upper *= 2
    else:
        raise AssertionError('failed to bracket independent MDC root')
    MDC = brentq(residual, 0., upper, xtol=1e-15, rtol=2e-13)
    return {'LC': LC, 'MDC': MDC, 'sigma0': sigma0, 'sigma_at_MDC': sigma(MDC)}


def validate_inputs(root, source, repo, JointModel, portable=False):
    lock = read(root / 'run_lock.json')
    require(set(lock) == {'manifest.json', 'plan.json', 'technical_plan.json', 'hist_compute.py'}, 'lock file set')
    for name, digest in lock.items():
        require(sha(root / name) == digest, 'locked file changed: ' + name)
    manifest = read(root / 'manifest.json')
    old = previous_summary = None
    if not portable:
        require(sha(source / 'manifest.json') == manifest['source_manifest_sha256'], 'old manifest SHA')
        require(sha(source / 'summary.json') == manifest['source_summary_sha256'], 'old summary SHA')
        old = read(source / 'manifest.json')
        previous_summary = read(source / 'summary.json')
        require(manifest['production_sha256'] == old['production_sha256'], 'production hash provenance')
    for name, digest in manifest['production_sha256'].items():
        require(sha(repo / name) == digest, 'production source changed: ' + name)
    require(read(root / 'plan.json') == expected_plan(), 'formal plan differs from independently specified plan')
    technical = [dict(dataset=d, stage='technical', target=4, family=f, n=3, chunk=0)
                 for d in DATES for f in FAMILIES]
    require(read(root / 'technical_plan.json') == technical, 'technical plan')
    expected = dict(coordinates='[a_V; gamma; zeta]', raw_activity_unit='Bq/m3',
                    alpha=.05, beta_ii=.05, mode='iterative', null_n_per_date=10000,
                    signal_n_per_date_isotope_method=3000, new_signal_vectors=48000,
                    reused_null_vectors=20000, signal_master_seed=2026102501,
                    signal_seed_fields=['master', 'date_index', 'target', 'family_index', 'chunk'],
                    technical_seed_extra=99, technical_vectors_excluded=12,
                    systematics=False, poisson_blocks=['SAMPLE', 'GASBK', 'DETBK'],
                    n_observations=21, n_estimated_parameters=17)
    for key, value in expected.items():
        require(manifest[key] == value, 'manifest.' + key)
    require(manifest['mdc_formula'] == 'Unmodified production iterative MDC, not multiplied by c', 'MDC definition')
    if old is not None:
        require(manifest['efficiencies'] == old['efficiency_update'], 'efficiency provenance')
    efficiency = manifest['efficiencies']
    require(efficiency['detector'] == 221 and efficiency['isotope_order'] == list(ISOTOPES.values()), 'efficiency order')
    for kind in ('sample', 'gasbk'):
        require(efficiency[kind] == [.241169, .367784, .233937, .233937], 'new efficiencies ' + kind)
    models, nominal_checks = {}, []
    require(set(manifest['datasets']) == set(DATES), 'dataset keys')
    for date in DATES:
        data = manifest['datasets'][date]
        if old is not None:
            original = old['datasets'][date]
            for current_key, old_key in [('model', 'model'), ('truth0', 'truth0'), ('nominal_MDC_Bq_m3', 'nominal_MDC')]:
                require(data[current_key] == original[old_key], date + ' source ' + current_key)
            for key in ('effective_air_volume_m3', 'time_parameters'):
                require(data[key] == original['metadata'][key], date + ' ' + key)
        model = JointModel(**data['model'])
        require(model.M.shape == (21, 17) and model.n_sample_parameters == 5 and model.n_detector_parameters == 7, date + ' model dimensions')
        require(model.has_detector_background, date + ' DETBK model')
        require(np.all(model.b == 0), date + ' additive background must be zero')
        truth0 = np.asarray(data['truth0'])
        exact_array(truth0[:10], np.zeros(10), date + ' empty sample and memory')
        require(np.all(truth0[10:] > 0), date + ' DETBK truth')
        models[date] = model
        require(set(data['settings']) == set(map(str, range(1, 5))), date + ' targets')
        for target, isotope in ISOTOPES.items():
            curve = None
            if previous_summary is not None:
                curve, = [c for c in previous_summary['curves'] if c['dataset'] == date and c['target'] == target]
            setting = data['settings'][str(target)]
            require(setting['isotope'] == isotope, date + ' isotope')
            if curve is not None:
                require(setting['formula_MDC_Bq_m3'] == curve['nominal_MDC_mBq_m3'] / 1000, date + ' exact nominal conversion')
            nominal = independent_limits(model.physical_matrix, model.b, truth0, target)['MDC']
            near(setting['formula_MDC_Bq_m3'], nominal, date + ' known-background nominal root', rtol=3e-8)
            nominal_checks.append(dict(dataset=date, target=target, independent_MDC_Bq_m3=nominal))
            for family in FAMILIES:
                method = setting['methods'][family]
                require(method['signal_activity_Bq_m3'] > 0 and method['c'] > 0, date + ' fixed signal and cutoff')
                if curve is not None:
                    require(method['signal_activity_Bq_m3'] == curve['methods'][family]['calibrated_ld_mBq_m3'] / 1000,
                            date + ' exact calibrated Ld / 1000')
                    require(method['c'] == curve['methods'][family]['calibration']['c'], date + ' exact fixed c')
    expected_sources = {f'{d}_validation_i0_f00000_c{c}' for d in DATES for c in range(10)}
    require(set(manifest['null_source_sha256']) == expected_sources, 'null source set')
    for name, extensions in manifest['null_source_sha256'].items():
        require(set(extensions) == {'json', 'npz'}, 'null source extensions')
        if not portable:
            for ext, digest in extensions.items():
                require(sha(source / 'points' / (name + '.' + ext)) == digest, 'null source SHA ' + name + '.' + ext)
    return manifest, models, nominal_checks


def array_contract(meta, a, label):
    n, p = meta['n'], len(meta['targets'])
    expected_keys = {'indices', 'targets', 'x'}
    require(a['x'].shape == (n, 21) and np.issubdtype(a['x'].dtype, np.integer) and np.all(a['x'] >= 0), label + ' raw Poisson observations')
    exact_array(a['indices'], np.arange(n), label + ' row indices')
    exact_array(a['targets'], meta['targets'], label + ' target indices')
    for f in meta['families']:
        for suffix in ('free', 'positive'):
            key = f + '_' + suffix
            expected_keys.add(key)
            require(a[key].shape == (n, 17), label + ' ' + key + ' shape')
            require(not np.any(np.isinf(a[key])), label + ' infinite fit')
            finite = np.isfinite(a[key])
            require(np.all(np.all(finite, axis=1) | np.all(~finite, axis=1)), label + ' partial fit row')
        positive = a[f + '_positive']
        require(np.all(positive[np.isfinite(positive)] >= 0), label + ' physical nonnegativity')
        for suffix in ('LC', 'MDC', 'sigma0', 'sigma_at_MDC', 'iterations', 'converged', 'decision_standard', 'decision_calibrated'):
            key = f + '_' + suffix
            expected_keys.add(key)
            require(a[key].shape == (n, p), label + ' ' + key + ' shape')
            require(not np.any(np.isinf(a[key])), label + ' infinite diagnostic')
        key = f + '_free_covariance_available'
        expected_keys.add(key)
        require(a[key].shape == (n,) and a[key].dtype == np.bool_, label + ' covariance flags')
        require(a[f + '_converged'].dtype == np.bool_, label + ' convergence flags')
        require(np.issubdtype(a[f + '_iterations'].dtype, np.integer), label + ' iteration dtype')
        require(np.all((a[f + '_iterations'] >= 0) & (a[f + '_iterations'] <= 200)), label + ' iteration range')
        conv = a[f + '_converged']
        require(np.all(np.isnan(a[f + '_MDC'][~conv])), label + ' unavailable MDC retained as NaN')
        require(np.all(np.isfinite(a[f + '_MDC'][conv]) & (a[f + '_MDC'][conv] > 0)), label + ' converged MDC')
        require(np.all(np.isnan(a[f + '_sigma_at_MDC'][~conv])), label + ' unavailable final sigma')
        near(a[f + '_MDC'][conv], (a[f + '_LC'] + Z95 * a[f + '_sigma_at_MDC'])[conv],
             label + ' unmodified formula equation (no c)', rtol=3e-8)
        finite_lc = np.isfinite(a[f + '_LC'])
        require(np.array_equal(finite_lc, np.isfinite(a[f + '_sigma0'])), label + ' sigma0 availability')
        near(a[f + '_LC'][finite_lc], Z95 * a[f + '_sigma0'][finite_lc], label + ' LC = z*sigma0')
        require(meta['valid_MDC'][f] == int(conv.sum()), label + ' valid MDC metadata')
        for suffix in ('decision_standard', 'decision_calibrated'):
            require(np.all(np.isin(a[f + '_' + suffix], [-1, 0, 1])), label + ' decision domain')
        for row in range(n):
            failures = [e for e in meta['errors'] if e['family'] == f and e['row'] == row]
            free_ok = np.all(np.isfinite(a[f + '_free'][row]))
            positive_ok = np.all(np.isfinite(a[f + '_positive'][row]))
            if not free_ok:
                require(any(e['phase'] == 'free' for e in failures), label + ' unlogged free failure')
            if free_ok and not positive_ok:
                require(any(e['phase'] == 'positive' for e in failures), label + ' unlogged positive failure')
            if not free_ok or not positive_ok:
                require(np.all(~finite_lc[row]) and np.all(~conv[row]), label + ' limits after failed fit')
            for column, target in enumerate(meta['targets']):
                if positive_ok and free_ok and not conv[row, column]:
                    require(any(e['phase'] in ('MDC', 'MDC_nonconvergence') and e.get('target') == target for e in failures),
                            label + ' unlogged MDC failure')
                for suffix in ('decision_standard', 'decision_calibrated'):
                    require((a[f + '_' + suffix][row, column] >= 0) == bool(finite_lc[row, column] and free_ok), label + ' decision availability')
    require(set(a) == expected_keys, label + ' NPZ array set')
    for e in meta['errors']:
        require(e['family'] in meta['families'] and 0 <= e['row'] < n and isinstance(e['error'], str), label + ' error record')
        require(e['phase'] in ('free', 'positive', 'MDC', 'MDC_nonconvergence'), label + ' error phase')
        if e['phase'].startswith('MDC'):
            require(e['target'] in meta['targets'], label + ' error target')


def refit_selected(meta, a, model, JointModel, fit, report):
    for row in (0, meta['n'] - 1):
        trial = JointModel(model.M, a['x'][row], model.b, model.n_sample_parameters,
                           model.T_c, model.coordinates, model.n_detector_parameters)
        for family in meta['families']:
            for kind, method in [('free', family), ('positive', POSITIVE_METHOD[family])]:
                saved = a[family + '_' + kind][row]
                if kind == 'positive' and np.all(np.isnan(a[family + '_free'][row])):
                    continue
                report['refit_attempts'] += 1
                try:
                    result = fit(trial, method)
                    success = result.success and result.physical_parameters is not None and np.all(np.isfinite(result.physical_parameters))
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                    success = False
                require(success == bool(np.all(np.isfinite(saved))), stem(meta) + ' refit availability ' + method)
                if success:
                    near(result.physical_parameters, saved, stem(meta) + f' row {row} refit {method}', rtol=1e-8, atol=1e-10)
                    report['refits_confirmed'] += 1
                    if kind == 'free':
                        require((result.physical_covariance is not None) == a[family + '_free_covariance_available'][row], 'refit covariance availability')
            positive = a[family + '_positive'][row]
            for col, target in enumerate(meta['targets']):
                if not a[family + '_converged'][row, col]:
                    report['independent_roots_unavailable'] += 1
                    continue
                expected = independent_limits(model.physical_matrix, model.b, positive, target)
                for key, value in expected.items():
                    near(a[family + '_' + key][row, col], value,
                         stem(meta) + f' row {row} {family} target {target} independent {key}', rtol=3e-8)
                relative = abs(a[family + '_MDC'][row, col] / expected['MDC'] - 1)
                report['max_relative_MDC_root_difference'] = max(report['max_relative_MDC_root_difference'], float(relative))
                report['independent_MDC_roots_checked'] += 1


def audit_batches(root, source, manifest, models, JointModel, fit, portable=False):
    plan = expected_plan()
    expected = {stem(job): job for job in plan}
    for ext in ('json', 'npz'):
        require({p.stem for p in (root / 'points').glob('*.' + ext)} == set(expected), 'complete formal ' + ext + ' batch set')
    report = dict(formal_batches=68, poisson_vectors_checked=0, reused_null_vectors_checked=0,
                  new_signal_vectors_checked=0, reused_fit_vectors_checked=0,
                  refit_attempts=0, refits_confirmed=0, independent_MDC_roots_checked=0,
                  independent_roots_unavailable=0, max_relative_MDC_root_difference=0.)
    grouped, decisions = defaultdict(list), defaultdict(list)
    errors, covariance, sources = [], {}, {}
    all_seeds = set()
    for name in sorted(expected):
        path = root / 'points' / (name + '.json')
        meta = read(path)
        job = expected[name]
        for key, value in job.items():
            require(meta[key] == value, name + ' job field ' + key)
        null = meta['stage'] == 'null'
        dataset, family, n, chunk = meta['dataset'], meta['family'], meta['n'], meta['chunk']
        targets = [1, 2, 3, 4] if null else [meta['target']]
        families = list(FAMILIES) if null else [family]
        require(meta['targets'] == targets and meta['families'] == families, name + ' targets/families')
        require(meta['reused_null_fits'] == null, name + ' reuse flag')
        for key, value in dict(raw_activity_unit='Bq/m3', mode='iterative', alpha=.05, beta_ii=.05).items():
            require(meta[key] == value, name + ' ' + key)
        require(meta['manifest_sha256'] == sha(root / 'manifest.json'), name + ' manifest SHA')
        require(meta['npz_sha256'] == sha(path.with_suffix('.npz')), name + ' NPZ SHA')
        sources[name] = {'json': sha(path), 'npz': meta['npz_sha256']}
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as loaded:
            a = {key: loaded[key] for key in loaded.files}
        array_contract(meta, a, name)
        truth = np.asarray(manifest['datasets'][dataset]['truth0'], dtype=float).copy()
        if null:
            seed = [2026102401, DATES.index(dataset), 0, 0, 11, chunk]
        else:
            truth[meta['target']] = manifest['datasets'][dataset]['settings'][str(meta['target'])]['methods'][family]['signal_activity_Bq_m3']
            seed = [2026102501, DATES.index(dataset), meta['target'], FAMILIES.index(family), chunk]
        require(meta['seed'] == seed and tuple(seed) not in all_seeds, name + ' unique planned seed')
        all_seeds.add(tuple(seed))
        exact_array(meta['truth'], truth, name + ' exact truth')
        mu = models[dataset].physical_matrix @ truth + models[dataset].b
        exact_array(meta['mu'], mu, name + ' expected counts')
        require(np.all(mu > 0), name + ' admissible generator')
        exact_array(a['x'], np.random.default_rng(seed).poisson(mu, size=(n, 21)), name + ' all Poisson x')
        report['poisson_vectors_checked'] += n
        if null:
            source_stem = f'{dataset}_validation_i0_f00000_c{chunk}'
            require(meta['source_stem'] == source_stem, name + ' source stem')
            require(meta['source_sha256'] == manifest['null_source_sha256'][source_stem], name + ' source hashes')
            if not portable:
                source_meta = read(source / 'points' / (source_stem + '.json'))
                require(source_meta['seed'] == seed and source_meta['n'] == n and source_meta['errors'] == [], name + ' source metadata')
                require(source_meta['stage'] == 'validation', name + ' independent-validation source')
                exact_array(source_meta['truth'], truth, name + ' source truth')
                exact_array(source_meta['mu'], mu, name + ' source mu')
                with np.load(source / 'points' / (source_stem + '.npz'), allow_pickle=False) as old:
                    exact_array(a['x'], old['x'], name + ' copied x')
                    for f in families:
                        for suffix in ('free', 'positive', 'free_covariance_available'):
                            exact_array(a[f + '_' + suffix], old[f + '_' + suffix], name + ' copied ' + f + '_' + suffix)
                        near(a[f + '_LC'], old[f + '_LC'], name + ' unchanged LC')
                        exact_array(a[f + '_decision_standard'], old[f + '_decision'], name + ' unchanged standard decisions')
                        report['reused_fit_vectors_checked'] += 2 * n
            report['reused_null_vectors_checked'] += n
        else:
            require('source_stem' not in meta and 'source_sha256' not in meta, name + ' new signal source')
            report['new_signal_vectors_checked'] += n
        scenario = 'null' if null else 'signal'
        errors.extend(dict(source=path.name, **e) for e in meta['errors'])
        for f in families:
            ck = (scenario, f)
            r = covariance.setdefault(ck, dict(scenario=scenario, family=f, n=0, unavailable=0))
            r['n'] += n
            r['unavailable'] += int(np.count_nonzero(~a[f + '_free_covariance_available']))
            for col, target in enumerate(targets):
                key = f'{dataset}_{f}_i{target}_{scenario}'
                grouped[key].append(a[f + '_MDC'][:, col] * 1000)
                LC = a[f + '_LC'][:, col]
                free = a[f + '_free'][:, target]
                available = np.isfinite(LC) & np.isfinite(free)
                c = manifest['datasets'][dataset]['settings'][str(target)]['methods'][f]['c']
                for rule, multiplier in [('standard', 1.), ('calibrated', c)]:
                    decision = a[f + '_decision_' + rule][:, col]
                    expected_decision = np.full(n, -1, dtype=np.int8)
                    expected_decision[available] = free[available] > multiplier * LC[available]
                    exact_array(decision, expected_decision, name + ' ' + f + ' ' + rule + ' decision')
                    decisions[(key, rule)].append(decision)
        refit_selected(meta, a, models[dataset], JointModel, fit, report)
        print('Audited', name, flush=True)
    require(report['poisson_vectors_checked'] == 68000 and report['reused_null_vectors_checked'] == 20000 and report['new_signal_vectors_checked'] == 48000,
            'vector totals')
    require(report['reused_fit_vectors_checked'] == (0 if portable else 80000), 'reused fit vector count')
    arrays = {key: np.concatenate(parts) for key, parts in grouped.items()}
    decision_arrays = {key: np.concatenate(parts) for key, parts in decisions.items()}
    return report, arrays, decision_arrays, errors, list(covariance.values()), sources


def distribution_summary(values):
    """Independent sample summaries; quantiles use explicit linear interpolation."""
    finite = np.sort(np.asarray(values)[np.isfinite(values)])
    n = len(finite)
    require(n >= 2, 'insufficient valid MDC values')
    mean = float(np.sum(finite) / n)
    sd = float(np.sqrt(np.sum((finite - mean)**2) / (n - 1)))

    def quantile(probability):
        position = (n - 1) * probability
        lower = int(np.floor(position))
        upper = min(lower + 1, n - 1)
        return float(finite[lower] + (position - lower) * (finite[upper] - finite[lower]))

    return dict(n_attempted=len(values), n_valid=n, n_missing=len(values)-n,
                mean=mean, median=quantile(.5), sd=sd, mean_mcse=sd / np.sqrt(n),
                q025=quantile(.025), q25=quantile(.25), q75=quantile(.75),
                q975=quantile(.975), min=float(finite[0]), max=float(finite[-1]))


def cp_interval(k, n):
    return [0. if k == 0 else float(beta_distribution.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta_distribution.ppf(.975, k+1, n-k))]


def decision_summary(decisions):
    total = len(decisions)
    valid = decisions[decisions >= 0]
    n, k = len(valid), int(np.count_nonzero(valid == 1))
    if n == 0:
        return dict(n_attempted=total, n_valid=0, n_missing=total, rate=None, k=0, ci95=None)
    return dict(n_attempted=total, n_valid=n, n_missing=total-n, k=k, rate=k/n,
                ci95=cp_interval(k, n), unconditional_rate_bounds=[k/total, (k+total-n)/total],
                conservative_ci95=[cp_interval(k, total)[0], cp_interval(k+total-n, total)[1]])


def audit_summary(root, manifest, arrays, decisions, errors, covariance, sources):
    summary = read(root / 'summary.json')
    require(summary['values_sha256'] == sha(root / 'values.npz'), 'compact values SHA')
    require(summary['manifest_sha256'] == sha(root / 'manifest.json'), 'summary manifest SHA')
    require(summary['plan_sha256'] == sha(root / 'plan.json'), 'summary plan SHA')
    require(summary['source_batches'] == sources, 'summary all source batch checksums')
    with np.load(root / 'values.npz', allow_pickle=False) as compact:
        require(set(compact.files) == set(arrays) and len(arrays) == 32, 'compact array keys')
        for key, expected in arrays.items():
            exact_array(compact[key], expected, 'all attempts including NaNs: ' + key)
            require(expected.shape == (10000 if key.endswith('_null') else 3000,), key + ' attempt count')
    require(summary['errors'] == errors, 'summary error accounting')
    require(summary['covariance_diagnostics'] == covariance, 'covariance availability is separately counted')
    total = sum(len(a) for a in arrays.values())
    valid = sum(int(np.isfinite(a).sum()) for a in arrays.values())
    require(total == 208000, '208000 MDC attempts required')
    for key, value in dict(unit='mBq/m3', mdc_mode='standard_iterative_alpha_beta_0.05_without_c',
                           n_reused_null_vectors=20000, n_new_signal_vectors=48000,
                           n_mdc_attempts=total, n_mdc_valid=valid, n_mdc_missing=total-valid,
                           technical_vectors_excluded=12, signal_Ld_uncertainty_resampled=False,
                           thresholds_retuned=False, systematics=False).items():
        require(summary[key] == value, 'summary.' + key)
    curves = []
    for dataset in DATES:
        for family in FAMILIES:
            for target, isotope in ISOTOPES.items():
                setting = manifest['datasets'][dataset]['settings'][str(target)]
                curve = dict(dataset=dataset, family=family, target=target, isotope=isotope,
                             nominal_MDC_mBq_m3=setting['formula_MDC_Bq_m3']*1000,
                             signal_A_mBq_m3=setting['methods'][family]['signal_activity_Bq_m3']*1000,
                             cutoff_c=setting['methods'][family]['c'])
                for scenario in ('null', 'signal'):
                    key = f'{dataset}_{family}_i{target}_{scenario}'
                    values = arrays[key]
                    result = distribution_summary(values)
                    result['detection'] = {rule: decision_summary(decisions[(key, rule)]) for rule in ('standard', 'calibrated')}
                    finite = values[np.isfinite(values)]
                    result['fraction_below_nominal_formula'] = int(np.count_nonzero(finite < curve['nominal_MDC_mBq_m3'])) / len(finite)
                    result['fraction_below_signal_Ld95'] = int(np.count_nonzero(finite < curve['signal_A_mBq_m3'])) / len(finite)
                    curve[scenario] = result
                curves.append(curve)
    same_tree(summary['curves'], curves, 'all independently recomputed curve summaries')
    return dict(compact_arrays_checked=32, mdc_attempts_checked=total, valid_MDC=valid,
                unavailable_MDC=total-valid, error_records=len(errors),
                summary_sha256=sha(root / 'summary.json'), values_sha256=sha(root / 'values.npz'),
                covariance_diagnostics=covariance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('XEMAT_PREVIOUS_STUDY', SOURCE_DEFAULT)))
    parser.add_argument('--repo', type=Path, default=Path(os.environ.get('XEMAT_REPO', REPO_DEFAULT)))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--portable', action='store_true', help='Explicitly omit old-study provenance and full null-source comparisons; still regenerate all x and perform selected refits and independent MDC roots')
    args = parser.parse_args()
    root, source, repo = args.root.resolve(), args.source.resolve(), args.repo.resolve()
    sys.path.insert(0, str(repo))
    from xemat.model import JointModel
    from xemat.solvers import fit
    manifest, models, nominal_checks = validate_inputs(root, source, repo, JointModel, portable=args.portable)
    batch_report, arrays, decisions, errors, covariance, sources = audit_batches(root, source, manifest, models, JointModel, fit, portable=args.portable)
    summary_report = audit_summary(root, manifest, arrays, decisions, errors, covariance, sources)
    result = dict(status='passed', audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  audit_file_sha256=sha(Path(__file__)), root=str(root), source=str(source), repo=str(repo),
                  manifest_sha256=sha(root / 'manifest.json'), run_lock_sha256=sha(root / 'run_lock.json'),
                  production_sha256=manifest['production_sha256'], nominal_root_checks=nominal_checks,
                  selected_rows_per_batch=[0, 999], independent_method='scaled QR Fisher inverse + brentq MDC root; alpha=beta=0.05; positive pilot; no c',
                  source_validation_mode='portable_without_old_sources' if args.portable else 'full_source_audit',
                  unavailable_source_checks=([
                      'Original old-study manifest and summary hashes and exact setting provenance',
                      'Original null-source JSON/NPZ hashes',
                      'All reused null fits/covariance flags/LC/decisions compared with old source arrays'
                  ] if args.portable else []),
                  **batch_report, **summary_report)
    output = args.output or root / ('audit_portable.json' if args.portable else 'audit.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
