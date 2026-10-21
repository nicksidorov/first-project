"""Sampling distribution of reported Xe133m MDC; production XeMat is unchanged.

Run with single-threaded BLAS. Truth coordinates are [a_V; gamma; zeta],
with a_V in Bq/m3. The application supplies the fitting and iterative MDC.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('XEMAT_REPO', '/workspace/first-project'))
sys.path.insert(0, str(REPO))

import numpy as np
import scipy
from scipy.stats import norm

from xemat.detection import compute_detection_limits
from xemat.model import JointModel
from xemat.solvers import fit

PREVIOUS = Path('/workspace/scratch/may-xe133m-ld-power')
FAMILIES = {'wls': 'nnls', 'poisson': 'poisson_nonnegative'}
TARGET = 4
N = 10000
CHUNK = 1000
SEED = 2026102112


def checked_fit(model, method):
    value = fit(model, method)
    if not value.success:
        raise RuntimeError(f'{method}: {value.message}')
    if value.physical_parameters is None or not np.all(np.isfinite(value.physical_parameters)):
        raise ValueError(f'{method}: invalid physical parameters')
    return value


def prepare():
    destination = ROOT / 'manifest.json'
    if destination.exists():
        raise FileExistsError(destination)
    previous = json.loads((PREVIOUS/'manifest.json').read_text())
    powers = json.loads((PREVIOUS/'power_summary.json').read_text())
    for filename, digest in previous['production_sha256'].items():
        assert hashlib.sha256((REPO/filename).read_bytes()).hexdigest() == digest
    for name, data in previous['datasets'].items():
        model = JointModel(**data['model'])
        positive = checked_fit(model, 'poisson_nonnegative').physical_parameters
        np.testing.assert_allclose(positive, data['positive'], rtol=1e-10, atol=1e-10)
        data['reference_ld'] = {}
        data['original_reported'] = {}
        for family, positive_method in FAMILIES.items():
            ref = powers['ld'][name]['methods'][family+'_LC']
            boot = powers['paired_bootstrap']['datasets'][name]['methods'][family+'_LC']
            data['reference_ld'][family] = dict(
                estimate=ref['ld_interpolation']['estimate_mBq_m3']/1000,
                approximate_bootstrap_ci95=np.array(boot['approximate_ci95_mBq_m3']).tolist(),
                approximate_bootstrap_ci95_unit='mBq/m3',
                grid_bound95_mBq_m3=ref['grid_bound95_mBq_m3'])
            free = checked_fit(model, family)
            pilot = checked_fit(model, positive_method)
            reported = compute_detection_limits(model.physical_matrix, model.b,
                free.physical_parameters, pilot.physical_parameters, component_indices=[TARGET])
            assert reported.converged[0]
            data['original_reported'][family] = dict(MDC=float(reported.MDC[0]),
                LC=float(reported.L_C[0]), activity=float(free.physical_parameters[TARGET]))
        # Both fitting families see the very same counts in both scenarios.
        # The common signal level is the previously estimated Poisson-Lc Ld95.
        data['signal_activity'] = data['reference_ld']['poisson']['estimate']
        truth0 = np.array(data['truth0'])
        oracle = compute_detection_limits(model.physical_matrix, model.b, truth0, truth0,
                                          component_indices=[TARGET])
        data['oracle_formula_MDC'] = float(oracle.MDC[0])
        data['oracle_formula_LC'] = float(oracle.L_C[0])
        assert np.isclose(data['oracle_formula_MDC'], data['reference_MDC'], rtol=1e-10)
    manifest = dict(scope='Distribution of reported iterative MDC for the existing Lc procedure',
        datasets=previous['datasets'], target=TARGET, unit='Bq/m3; multiply by 1000 for mBq/m3',
        alpha=.05, beta_ii=.05, repetitions_per_scenario=N, chunk_size=CHUNK, seed=SEED,
        scenario_names=['null','signal'], families=FAMILIES,
        generator='Same ML+ mixtures as the power study; 21 independent Poisson observations including DETBK',
        signal_definition='Common signal at prior Poisson-Lc Ld95; WLS evaluated on identical counts, compared to its own reference Ld',
        generation_independent_of_previous_power_study=True, reused_trials=0,
        mdc_mode='iterative', calibration=previous['calibration'],
        systematic_uncertainties=False, git_head=previous['git_head'],
        production_sha256=previous['production_sha256'],
        prior_manifest_sha256=hashlib.sha256((PREVIOUS/'manifest.json').read_bytes()).hexdigest(),
        prior_power_summary_sha256=hashlib.sha256((PREVIOUS/'power_summary.json').read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        stopping_rule='All four declared series of 10000 completed; no outcome-dependent stopping or replacement of failures')
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (ROOT/'previous_power_summary.json').write_bytes((PREVIOUS/'power_summary.json').read_bytes())
    print('Manifest fixed', {k:v['signal_activity'] for k,v in manifest['datasets'].items()},flush=True)


def evaluate(dataset, scenario, start, stop, validation=False):
    begun = time.monotonic()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    data = manifest['datasets'][dataset]
    model = JointModel(**data['model'])
    activity = 0.0 if scenario == 'null' else data['signal_activity']
    truth = np.array(data['truth0']); truth[TARGET] = activity
    mu = model.physical_matrix @ truth + model.b
    assert np.all(mu > 0) and np.all(truth >= 0)
    seed = [SEED, list(manifest['datasets']).index(dataset), ['null','signal'].index(scenario), int(validation)]
    counts = np.random.default_rng(seed).poisson(mu, size=(N, len(mu)))[start:stop]
    stem = f'{dataset}_{scenario}_{start:05d}_{stop:05d}'
    folder = ROOT/('validation' if validation else 'chunks')
    folder.mkdir(exist_ok=True)
    if (folder/(stem+'.npz')).exists():
        raise FileExistsError(stem)
    n, p = len(counts), len(truth)
    arrays = dict(x=counts, indices=np.arange(start,stop))
    for family in FAMILIES:
        for key in ('LC','MDC','sigma0','sigma_end','equation_relative_residual'):
            arrays[family+'_'+key] = np.full(n,np.nan)
        for key in ('free','positive'):
            arrays[family+'_'+key] = np.full((n,p),np.nan)
        arrays[family+'_detected'] = np.full(n,-1,dtype=np.int8)
        arrays[family+'_iterations'] = np.zeros(n,dtype=np.int16)
        arrays[family+'_free_covariance_available'] = np.zeros(n,dtype=bool)
    errors = []
    for row, x in enumerate(counts):
        trial = JointModel(model.M, x, model.b, model.n_sample_parameters,
                           model.T_c, model.coordinates, model.n_detector_parameters)
        for family, positive_method in FAMILIES.items():
            phase = 'free'
            try:
                free = checked_fit(trial,family)
                arrays[family+'_free'][row] = free.physical_parameters
                arrays[family+'_free_covariance_available'][row] = free.physical_covariance is not None
                phase = 'positive'
                positive = checked_fit(trial,positive_method)
                arrays[family+'_positive'][row] = positive.physical_parameters
                phase = 'MDC'
                limits = compute_detection_limits(model.physical_matrix, model.b,
                    free.physical_parameters, positive.physical_parameters,
                    component_indices=[TARGET], mode='iterative', alpha=.05, beta_ii=.05)
                arrays[family+'_LC'][row] = limits.L_C[0]
                arrays[family+'_detected'][row] = int(limits.detected[0])
                arrays[family+'_iterations'][row] = limits.iterations[0]
                if not limits.converged[0] or not np.isfinite(limits.MDC[0]):
                    raise ValueError('Iterative MDC did not converge')
                assert limits.MDC[0] > limits.L_C[0] > 0
                for key, value in [('MDC', limits.MDC[0]),('sigma0',limits.sigma0[0]),
                                   ('sigma_end',limits.sigma_at_MDC[0])]:
                    arrays[family+'_'+key][row] = value
                arrays[family+'_equation_relative_residual'][row] = abs(
                    limits.MDC[0] - limits.L_C[0] - norm.ppf(.95)*limits.sigma_at_MDC[0])/limits.MDC[0]
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                errors.append(dict(index=int(start+row),family=family,phase=phase,
                                   error=f'{type(exc).__name__}: {exc}'))
    metadata = dict(dataset=dataset, scenario=scenario, start=start, stop=stop,
        n_attempted=n, activity=activity, seed=seed, truth=truth.tolist(), mu=mu.tolist(),
        validation=validation, errors=errors, elapsed_seconds=time.monotonic()-begun,
        n_MDC_valid={f:int(np.isfinite(arrays[f+'_MDC']).sum()) for f in FAMILIES},
        production_sha256=manifest['production_sha256'])
    np.savez_compressed(folder/(stem+'.npz'),**arrays)
    (folder/(stem+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print('DONE',stem,'validation',validation,'seconds',round(metadata['elapsed_seconds'],1),
          'errors',len(errors),flush=True)
    return stem


def run(validation=False, workers=4):
    jobs=[]
    for dataset in ('may22','may26'):
        for scenario in ('null','signal'):
            for start in range(0,100 if validation else N,100 if validation else CHUNK):
                stop = start+(100 if validation else CHUNK)
                folder=ROOT/('validation' if validation else 'chunks')
                stem=f'{dataset}_{scenario}_{start:05d}_{stop:05d}'
                if (folder/(stem+'.npz')).exists() and (folder/(stem+'.json')).exists():
                    print('Already completed',stem,flush=True)
                else:
                    jobs.append((dataset,scenario,start,stop,validation))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate,*job) for job in jobs]
        for future in as_completed(futures):
            future.result()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','validate','run'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    else: run(args.action=='validate',args.workers)
