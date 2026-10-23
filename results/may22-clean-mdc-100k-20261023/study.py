"""Frozen clean-background study, May 22 only, large independent MC samples.

The production solvers and detection formula are unmodified. Every generated
observation is retained, including failures; no replacement draws are made.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('XEMAT_REPO', '/workspace/first-project'))
sys.path.insert(0, str(REPO))
import numpy as np
from scipy.stats import binomtest
from xemat.model import JointModel
from xemat.detection import compute_detection_limits
from xemat.solvers import fit

FAMILIES = {'wls': 'nnls', 'poisson': 'poisson_nonnegative'}
ISOTOPES = {1: 'Xe135', 2: 'Xe133', 3: 'Xe131m', 4: 'Xe133m'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def name_of(job):
    stem = f"{job['dataset']}_{job['stage']}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job['chunk']}"
    if job['stage'] == 'hist_signal':
        stem += '_' + job['family']
    return stem


def checked_fit(model, method):
    result = fit(model, method)
    if not result.success:
        raise RuntimeError(result.message)
    if result.physical_parameters is None or not np.all(np.isfinite(result.physical_parameters)):
        raise ValueError('Invalid physical parameters')
    return result


def run_point(dataset, target, factor, n, chunk=0, stage='calibration',
              family='both', activity_Bq_m3=None, mode=None):
    began = time.monotonic()
    job = dict(dataset=dataset, target=target, factor=factor, n=n, chunk=chunk, stage=stage)
    if family != 'both':
        job['family'] = family
    if activity_Bq_m3 is not None:
        job['activity_Bq_m3'] = activity_Bq_m3
    if mode is not None:
        job['mode'] = mode
    manifest = read(ROOT/'manifest.json')
    d = manifest['datasets'][dataset]
    model = JointModel(**d['model'])
    S = model.physical_matrix
    truth = np.array(d['truth0'], dtype=float)
    if target:
        truth[target] = (factor*d['nominal_MDC'][target-1]
                         if activity_Bq_m3 is None else activity_Bq_m3)
    elif factor != 0 or activity_Bq_m3 is not None:
        raise ValueError('Null stage must have zero activity')
    mu = S @ truth + model.b
    assert np.all(mu > 0)
    code = int(round(factor*10000))
    assert np.isclose(code/10000, factor, atol=1e-12, rtol=0)
    seed = [manifest['master_seed'], 0, target, code, manifest['stage_codes'][stage], chunk]
    if stage == 'hist_signal':
        assert family in FAMILIES and activity_Bq_m3 is not None
        seed.append(list(FAMILIES).index(family))
    elif family != 'both':
        raise ValueError('Main experiments require paired methods')
    mode = mode or ('iterative' if stage in ('validation', 'hist_signal') else 'fixed')
    assert mode in ('fixed', 'iterative')
    indices = list(ISOTOPES) if target == 0 else [target]
    families = list(FAMILIES) if family == 'both' else [family]
    counts = np.random.default_rng(seed).poisson(mu, size=(n, 21))
    stem = name_of(job)
    folder = ROOT/('technical' if stage == 'technical' else 'points')
    folder.mkdir(exist_ok=True)
    if (folder/(stem+'.npz')).exists() or (folder/(stem+'.json')).exists():
        raise FileExistsError(stem)
    arrays = dict(x=counts, indices=np.arange(n), targets=np.array(indices, dtype=int))
    for f in families:
        for key in ('free', 'positive'):
            arrays[f+'_'+key] = np.full((n, 17), np.nan)
            arrays[f+'_'+key+'_success'] = np.zeros(n, dtype=bool)
        arrays[f+'_LC'] = np.full((n, len(indices)), np.nan)
        arrays[f+'_decision'] = np.full((n, len(indices)), -1, dtype=np.int8)
        arrays[f+'_free_covariance_available'] = np.zeros(n, dtype=bool)
        if mode == 'iterative':
            for key in ('MDC', 'sigma0', 'sigma_at_MDC'):
                arrays[f+'_'+key] = np.full((n, len(indices)), np.nan)
            arrays[f+'_iterations'] = np.zeros((n, len(indices)), dtype=np.int32)
            arrays[f+'_converged'] = np.zeros((n, len(indices)), dtype=bool)
    errors = []
    for row, x in enumerate(counts):
        trial = JointModel(model.M, x, model.b, model.n_sample_parameters,
                           model.T_c, model.coordinates, model.n_detector_parameters)
        for f in families:
            phase = 'free'
            try:
                free = checked_fit(trial, f)
                arrays[f+'_free'][row] = free.physical_parameters
                arrays[f+'_free_success'][row] = True
                arrays[f+'_free_covariance_available'][row] = free.physical_covariance is not None
                phase = 'positive'
                positive = checked_fit(trial, FAMILIES[f])
                arrays[f+'_positive'][row] = positive.physical_parameters
                arrays[f+'_positive_success'][row] = True
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                errors.append(dict(row=row, family=f, phase=phase, error=f'{type(exc).__name__}: {exc}'))
                continue
            for col, index in enumerate(indices):
                try:
                    result = compute_detection_limits(S, model.b, free.physical_parameters,
                        positive.physical_parameters, component_indices=[index], mode=mode)
                    arrays[f+'_LC'][row, col] = result.L_C[0]
                    arrays[f+'_decision'][row, col] = result.detected[0]
                    if mode == 'iterative':
                        for key, result_key in (('MDC', 'MDC'), ('sigma0', 'sigma0'),
                            ('sigma_at_MDC', 'sigma_at_MDC'), ('iterations', 'iterations'),
                            ('converged', 'converged')):
                            arrays[f+'_'+key][row, col] = getattr(result, result_key)[0]
                        if not result.converged[0]:
                            errors.append(dict(row=row, family=f, phase='MDC_nonconvergence',
                                target=index, error='; '.join(result.warnings)))
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    errors.append(dict(row=row, family=f, phase='LC_MDC', target=index,
                                       error=f'{type(exc).__name__}: {exc}'))
                    if mode == 'iterative':
                        # A failure while iterating MDC must not silently remove
                        # an otherwise available Lc or detection decision.
                        try:
                            lc_only = compute_detection_limits(S, model.b, free.physical_parameters,
                                positive.physical_parameters, component_indices=[index], mode='fixed')
                            arrays[f+'_LC'][row, col] = lc_only.L_C[0]
                            arrays[f+'_decision'][row, col] = lc_only.detected[0]
                        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as lc_exc:
                            errors.append(dict(row=row, family=f, phase='LC_recovery', target=index,
                                               error=f'{type(lc_exc).__name__}: {lc_exc}'))
    metadata = {**job, 'family': family, 'families': families, 'targets': indices,
        'mode': mode, 'seed': seed, 'truth': truth.tolist(), 'mu': mu.tolist(),
        'raw_activity_unit': 'Bq/m3', 'alpha': .05, 'beta_ii': .05,
        'manifest_sha256': sha(ROOT/'manifest.json'), 'errors': errors,
        'elapsed_seconds': time.monotonic()-began, 'rates': {}}
    for f in families:
        metadata['rates'][f] = []
        for col in range(len(indices)):
            values = arrays[f+'_decision'][:, col]
            valid = values >= 0
            total = int(valid.sum()); k = int(values[valid].sum())
            ci = binomtest(k, total).proportion_ci(.95, method='exact') if total else None
            metadata['rates'][f].append(dict(n_valid=total, n_missing=n-total, k=k,
                rate=k/total if total else None, ci95=[ci.low, ci.high] if ci else None))
    temp = folder/(stem+'.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, folder/(stem+'.npz'))
    metadata['npz_sha256'] = sha(folder/(stem+'.npz'))
    write(folder/(stem+'.json'), metadata)
    print('DONE', stem, 'n', n, 'seconds', round(metadata['elapsed_seconds'], 1),
          'errors', len(errors), 'mode', mode, flush=True)
    return stem


def execute(plan_path, workers):
    plan_path = Path(plan_path).resolve()
    locked = read(ROOT/'run_lock.json')
    for name, digest in locked.items():
        assert sha(ROOT/name) == digest, name
    if plan_path.name not in locked or plan_path.parent != ROOT:
        # Later adaptive plans are separately locked before sampling. They do
        # not alter the initial protocol or its checksum file.
        extra = read(plan_path.with_suffix('.lock.json'))
        assert extra['plan_sha256'] == sha(plan_path)
        assert extra['manifest_sha256'] == sha(ROOT/'manifest.json')
    manifest = read(ROOT/'manifest.json')
    for name, digest in manifest['production_sha256'].items():
        assert sha(REPO/name) == digest, name
    jobs = read(plan_path)
    assert len({name_of(job) for job in jobs}) == len(jobs)
    pending = []
    for job in jobs:
        folder = ROOT/('technical' if job['stage'] == 'technical' else 'points')
        stem = name_of(job)
        metadata = folder/(stem+'.json'); raw = folder/(stem+'.npz')
        if metadata.exists() and raw.exists():
            saved = read(metadata)
            assert all(saved[k] == v for k, v in job.items()), stem
            assert saved['manifest_sha256'] == sha(ROOT/'manifest.json'), stem
            assert saved['npz_sha256'] == sha(raw), stem
            continue
        pending.append(job)
    print('PLAN', plan_path.name, 'pending_batches', len(pending),
          'pending_vectors', sum(j['n'] for j in pending), 'workers', workers, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_point, **job) for job in pending]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    execute(args.plan, args.workers)
