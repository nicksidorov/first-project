"""May22 detector221: 100000 paired Xe133m null experiments with new epsilon.

All generated observations and all fit outcomes are retained. Each family
uses its unchanged free fit, unchanged nonnegative auxiliary fit, and its own
LC. No additional null fit, LD, MDC, Wald test, multiplier or power calculation.

Run all batches: python study.py --workers 8
Run first FORMAL batch: python study.py --chunks 0 --workers 1
Resume by repeating the same command. Finished batches must pass hashes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'kernel_snapshot'
sys.path.insert(0, str(REPO))

import numpy as np
from xemat.model import JointModel
from xemat.solvers import fit
from lc_only import critical_level

N = 100_000
BATCH_SIZE = 1000
TARGET_INDEX = 4
ISOTOPES = ['Xe135', 'Xe133', 'Xe131m', 'Xe133m']
FAMILIES = {'wls': 'nnls', 'poisson': 'poisson_nonnegative'}
FIT_STATUS = {-2: 'not attempted', -1: 'exception', 0: 'solver returned success=False',
              1: 'successful finite complete physical estimate',
              2: 'success but missing/nonfinite/wrong-size physical estimate'}
LC_STATUS = {-2: 'not attempted: auxiliary estimate unavailable', -1: 'LC exception',
             0: 'returned LC or sigma0 is nonfinite/negative', 1: 'available LC'}
DECISION_STATUS = {-1: 'unavailable; never count as nondetection',
                   0: 'available: free concentration <= LC',
                   1: 'available: free concentration > LC'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def get_lock():
    scenario = read(ROOT/'scenario.json')
    if (scenario['n_generated'] != N or scenario['batch_size'] != BATCH_SIZE or
            scenario['n_batches'] != N // BATCH_SIZE or scenario['target_index'] != TARGET_INDEX):
        raise RuntimeError('Scenario differs from the fixed 100000-draw Xe133m plan')
    current = dict(study_sha256=sha(__file__), lc_only_sha256=sha(ROOT/'lc_only.py'),
        scenario_sha256=sha(ROOT/'scenario.json'), scenario_arrays_sha256=sha(ROOT/'scenario_arrays.npz'),
        core_sha256={str(p.relative_to(REPO)): sha(p) for p in sorted(REPO.rglob('*.py'))},
        n=N, batch_size=BATCH_SIZE, target_index=TARGET_INDEX, families=FAMILIES,
        fit_calls='fit(trial, method), all default weights/tolerances/optimizer settings',
        lc_call='critical_level(S,b,aux,target_index=4): zero target; production model_covariance; norm.ppf(.95)*sigma0',
        no_MDC_calculation=True, no_LD_calculation=True, no_null_fit=True,
        no_threshold_multiplier=True, no_replacement=True,
        covariance_unavailability_does_not_invalidate_decision=True)
    path = ROOT/'runner_lock.json'
    if path.exists():
        if read(path) != current:
            raise RuntimeError('Frozen scenario or runner/kernel changed; resume refused')
    else:
        write(path, current)
    return current


def seed_for(chunk):
    return [int(read(ROOT/'scenario.json')['master_seed']), int(chunk)]


def batch_paths(chunk):
    stem = f'batch_{chunk:03d}'
    return ROOT/'points'/(stem+'.npz'), ROOT/'points'/(stem+'.json')


def finished(chunk, lock=None):
    raw, meta = batch_paths(chunk)
    if not raw.exists() and not meta.exists():
        return False
    if not raw.exists() or not meta.exists():
        raise RuntimeError(f'Incomplete checkpoint {chunk}; do not redraw or replace automatically')
    saved = read(meta)
    checks = (saved['chunk'] == chunk, saved['n_generated'] == BATCH_SIZE,
              saved['seed'] == seed_for(chunk), saved['npz_sha256'] == sha(raw),
              saved['runner_lock_sha256'] == sha(ROOT/'runner_lock.json'))
    if not all(checks):
        raise RuntimeError(f'Checkpoint verification failed: {chunk}')
    return True


def run_batch(chunk):
    started = time.monotonic()
    lock = get_lock()
    if finished(chunk, lock):
        return dict(chunk=chunk, already_finished=True)
    with np.load(ROOT/'scenario_arrays.npz', allow_pickle=False) as source:
        M, original_x, b, T, S, truth, mu0 = (
            source[key].copy() for key in ('M','x','b','T_V','S','z_truth','mu0'))
    model = JointModel(M, original_x, b, 5, T, 'beta_gamma_zeta', 7)
    np.testing.assert_array_equal(S, model.physical_matrix)
    np.testing.assert_allclose(mu0, S @ truth + b, rtol=1e-13, atol=1e-12)
    assert S.shape == (21,17) and np.all(mu0 > 0) and truth[TARGET_INDEX] == 0
    n = BATCH_SIZE
    seed = seed_for(chunk)
    counts = np.random.default_rng(seed).poisson(mu0, size=(n,21))
    arrays = dict(x=counts, replicate_id=np.arange(chunk*n,(chunk+1)*n,dtype=np.int64),
        chunk=np.full(n,chunk,dtype=np.int16), row_in_batch=np.arange(n,dtype=np.int16),
        isotope_order=np.array(ISOTOPES), target_isotope=np.array('Xe133m'),
        target_index=np.array(TARGET_INDEX), concentration_units=np.array('mBq/m3'),
        physical_coordinate_label=np.array('[a_V; gamma; zeta]'),
        physical_units=np.array(['Bq/m3 (legacy auxiliary Rn)']+['Bq/m3']*4+['expected reference GASBK counts']*5+['expected DETBK counts']*7),
        seed=np.asarray(seed,dtype=np.int64))
    for family in FAMILIES:
        for phase in ('free','auxiliary'):
            prefix = f'{family}_{phase}'
            arrays[prefix+'_physical'] = np.full((n,17),np.nan)
            arrays[prefix+'_status'] = np.full(n,-2,dtype=np.int8)
            arrays[prefix+'_diagnostic_code'] = np.zeros(n,dtype=np.int32)
            arrays[prefix+'_iterations'] = np.full(n,-1,dtype=np.int32)
            arrays[prefix+'_objective'] = np.full(n,np.nan)
            arrays[prefix+'_rank'] = np.full(n,-1,dtype=np.int16)
            arrays[prefix+'_condition_number'] = np.full(n,np.nan)
        arrays[family+'_free_covariance_available'] = np.zeros(n,dtype=bool)
        arrays[family+'_concentrations'] = np.full((n,4),np.nan)
        arrays[family+'_LC'] = np.full(n,np.nan)
        arrays[family+'_sigma0'] = np.full(n,np.nan)
        arrays[family+'_LC_status'] = np.full(n,-2,dtype=np.int8)
        arrays[family+'_LC_diagnostic_code'] = np.zeros(n,dtype=np.int32)
        arrays[family+'_decision'] = np.full(n,-1,dtype=np.int8)

    catalogue = [dict(message='No diagnostic')]
    lookup = {json.dumps(catalogue[0],sort_keys=True,ensure_ascii=False): 0}
    errors = []

    def code(diagnostic):
        key = json.dumps(diagnostic,sort_keys=True,ensure_ascii=False)
        if key not in lookup:
            lookup[key] = len(catalogue)
            catalogue.append(diagnostic)
        return lookup[key]

    def error(row,family,phase,message,**extra):
        errors.append(dict(replicate_id=int(arrays['replicate_id'][row]),row_in_batch=row,
                           family=family,phase=phase,message=message,**extra))

    def fitting(row,trial,family,phase,method):
        prefix = f'{family}_{phase}'
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = fit(trial,method)  # EXACT existing defaults, including observed WLS weights.
            physical = result.physical_parameters
            valid = (physical is not None and np.shape(physical) == (17,)
                     and bool(np.all(np.isfinite(physical))))
            if physical is not None and np.shape(physical) == (17,):
                arrays[prefix+'_physical'][row] = physical
            if phase == 'free':
                arrays[family+'_concentrations'][row] = arrays[prefix+'_physical'][row,1:5]*1000
                K = result.physical_covariance
                arrays[family+'_free_covariance_available'][row] = (
                    K is not None and np.shape(K) == (17,17) and np.all(np.isfinite(K)))
            status = 1 if result.success and valid else (0 if not result.success else 2)
            arrays[prefix+'_status'][row] = status
            arrays[prefix+'_iterations'][row] = int(result.iterations)
            arrays[prefix+'_objective'][row] = result.objective
            arrays[prefix+'_rank'][row] = int(result.rank)
            arrays[prefix+'_condition_number'][row] = result.condition_number
            diagnostic = dict(method=method,success=bool(result.success),message=str(result.message),
                warnings=list(result.warnings),python_warnings=[f'{w.category.__name__}: {w.message}' for w in caught])
            if not valid:
                diagnostic['invalid_physical_estimate'] = True
            arrays[prefix+'_diagnostic_code'][row] = code(diagnostic)
            if status != 1:
                error(row,family,phase,str(result.message),status=status,invalid_physical_estimate=not valid)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            arrays[prefix+'_status'][row] = -1
            arrays[prefix+'_diagnostic_code'][row] = code(dict(method=method,exception=message,
                python_warnings=[f'{w.category.__name__}: {w.message}' for w in caught]))
            error(row,family,phase,message,status=-1)

    for row,x in enumerate(counts):
        trial = JointModel(M,x,b,5,T,'beta_gamma_zeta',7)
        for family,auxiliary_method in FAMILIES.items():
            # Independent attempts: a failure in one stage must not erase the
            # other stage or other family's results. Failed draws are not retried.
            fitting(row,trial,family,'free',family)
            fitting(row,trial,family,'auxiliary',auxiliary_method)
            if arrays[family+'_auxiliary_status'][row] == 1:
                try:
                    lc = critical_level(S,b,arrays[family+'_auxiliary_physical'][row],
                                        target_index=TARGET_INDEX,alpha=.05)
                    arrays[family+'_LC'][row] = 1000*lc.L_C
                    arrays[family+'_sigma0'][row] = 1000*lc.sigma0
                    valid = bool(np.isfinite(lc.L_C) and lc.L_C >= 0 and
                                 np.isfinite(lc.sigma0) and lc.sigma0 >= 0)
                    arrays[family+'_LC_status'][row] = int(valid)
                    if not valid:
                        message = 'Returned LC or sigma0 is nonfinite/negative'
                        arrays[family+'_LC_diagnostic_code'][row] = code(dict(message=message))
                        error(row,family,'LC',message,status=0)
                except Exception as exc:
                    message = f'{type(exc).__name__}: {exc}'
                    arrays[family+'_LC_status'][row] = -1
                    arrays[family+'_LC_diagnostic_code'][row] = code(dict(exception=message))
                    error(row,family,'LC',message,status=-1)
            if arrays[family+'_free_status'][row] == 1 and arrays[family+'_LC_status'][row] == 1:
                # Target is fresh Xe133m (physical index4, concentration index3).
                # Covariance of the free estimator is not needed for this decision.
                arrays[family+'_decision'][row] = int(
                    arrays[family+'_concentrations'][row,3] > arrays[family+'_LC'][row])

    summaries = {}
    for family in FAMILIES:
        decision = arrays[family+'_decision']
        no_cov = ~arrays[family+'_free_covariance_available']
        summaries[family] = dict(
            free_status_counts={str(k):int(v) for k,v in Counter(arrays[family+'_free_status'].tolist()).items()},
            auxiliary_status_counts={str(k):int(v) for k,v in Counter(arrays[family+'_auxiliary_status'].tolist()).items()},
            n_available=int(np.sum(decision >= 0)),n_false_detections=int(np.sum(decision == 1)),
            n_failed_decisions=int(np.sum(decision < 0)),
            n_successful_free_covariance_unavailable=int(np.sum(no_cov & (arrays[family+'_free_status'] == 1))),
            n_covariance_unavailable_but_decision_available=int(np.sum(no_cov & (decision >= 0))))
    raw,meta = batch_paths(chunk)
    raw.parent.mkdir(exist_ok=True)
    temporary = raw.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    os.replace(temporary,raw)
    metadata = dict(chunk=chunk,n_generated=n,seed=seed,bit_generator='PCG64 via numpy.random.default_rng(seed)',
        sampling='rng.poisson(mu0,size=(1000,21)); same x for both families',
        runner_lock_sha256=sha(ROOT/'runner_lock.json'),npz_sha256=sha(raw),
        elapsed_seconds=time.monotonic()-started,diagnostic_catalogue=catalogue,errors=errors,summaries=summaries,
        fit_status_legend=FIT_STATUS,lc_status_legend=LC_STATUS,decision_legend=DECISION_STATUS)
    write(meta,metadata)
    print(f'DONE batch {chunk:03d}: n={n}, seconds={metadata["elapsed_seconds"]:.1f}, '
          f'errors={len(errors)}, failed_decisions={sum(s["n_failed_decisions"] for s in summaries.values())}',flush=True)
    return dict(chunk=chunk,elapsed_seconds=metadata['elapsed_seconds'],error_records=len(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--chunks',help='Comma-separated FORMAL chunk indices 0..99; default all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    chunks = list(range(N//BATCH_SIZE)) if args.chunks is None else [int(x) for x in args.chunks.split(',')]
    if not chunks or len(set(chunks)) != len(chunks) or any(x < 0 or x >= N//BATCH_SIZE for x in chunks):
        parser.error('Invalid or duplicate formal chunk indices')
    lock = get_lock()
    pending = [chunk for chunk in chunks if not finished(chunk,lock)]
    print(f'PLAN {len(pending)} pending formal batches, {len(pending)*BATCH_SIZE} observations, '
          f'workers={args.workers}; fixed 100000-draw plan; no MDC',flush=True)
    if args.workers == 1:
        for chunk in pending:
            run_batch(chunk)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_batch,chunk) for chunk in pending]
            for future in as_completed(futures):
                future.result()


if __name__ == '__main__':
    main()
