"""Freeze model, preselected power grid and independent random streams."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/scratch/may-clean-mdc-eff221-20261022')
REPO = Path('/workspace/first-project')
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from xemat.model import JointModel
from xemat.detection import compute_detection_limits

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    path = ROOT/name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    old = json.loads((SOURCE/'manifest.json').read_text())
    source = old['datasets']['may22']
    for name, expected in old['production_sha256'].items():
        assert sha(REPO/name) == expected, name
    model = JointModel(**source['model'])
    truth = np.array(source['truth0'])
    formula = compute_detection_limits(model.physical_matrix, model.b, truth, truth, mode='iterative')
    assert np.all(formula.converged)
    for key, field in [('nominal_LC', 'L_C'), ('nominal_MDC', 'MDC'),
                       ('nominal_sigma0', 'sigma0'), ('nominal_sigma_at_MDC', 'sigma_at_MDC')]:
        np.testing.assert_allclose(getattr(formula, field), source[key], rtol=1e-12, atol=1e-15)
    data = {key: source[key] for key in ('model', 'truth0', 'metadata', 'calibration_matrices',
                                        'input_paths', 'input_sha256')}
    data.update(mu0=(model.physical_matrix@truth+model.b).tolist(),
        nominal_LC=formula.L_C.tolist(), nominal_MDC=formula.MDC.tolist(),
        nominal_sigma0=formula.sigma0.tolist(), nominal_sigma_at_MDC=formula.sigma_at_MDC.tolist(),
        iterations=formula.iterations.tolist())
    grid = {1: [.96, 1., 1.04, 1.08], 2: [.96, 1., 1.04, 1.08],
            3: [1.02, 1.06, 1.10, 1.14], 4: [1.02, 1.06, 1.10, 1.14]}
    manifest = dict(scope='Clean sample, May 22 only, enlarged independent Monte Carlo; no Wald',
        datasets={'may22': data}, alpha=.05, beta_ii=.05, master_seed=2026102701,
        coordinates='[a_V; gamma; zeta]', unit='Bq/m3', isotopes=['Xe135','Xe133','Xe131m','Xe133m'],
        families=['wls','poisson'], production_sha256=old['production_sha256'],
        production_commit=old['production_commit'], efficiency_update=old['efficiency_update'],
        source_manifest_sha256=sha(SOURCE/'manifest.json'), source_summary_sha256=sha(SOURCE/'summary.json'),
        source_usage='Frozen physical model and pilot choice of power grid only; no old observations reused',
        systematics=False, numpy=np.__version__, scipy=scipy.__version__,
        generator=old['generator'], sampling=old['sampling'],
        lc_rule='A_hat_free > Lc, with per-trial corresponding nonnegative pilot and target mean removal',
        mc_mdc_mode='iterative only for validation histograms and later fixed-Ld signal histograms',
        calibration_rule='c=R_(ceil((n+1)*.95)), R=A_hat_free/Lc; strict R>c; freeze before validation',
        calibration_n_per_dataset=100000, validation_n_per_dataset=100000,
        signal_n_per_point=20000, histogram_signal_n_per_isotope_method=20000,
        histogram_null='All 100000 validation vectors and both methods, all four target MDCs',
        histogram_signal='New independent observations at fixed newly estimated calibrated Ld95 for each target/method',
        signal_reuse='Both free estimator families and both detection rules share each power-grid observation',
        signal_plan={str(k): v for k,v in grid.items()},
        extension_policy=dict(max_extra_points_per_isotope=4, n_per_point=20000, outward_step=.04,
            trigger='First .95 crossing or 95% bootstrap interval is not bracketed; no cutoff retuning',
            rule='Freeze each new activity, seed and reason before generation; retain all prior points',
            max_points_including_null=9),
        bootstrap=dict(repetitions=2000, master_seed=2026102709, paired=True,
            total='Paired calibration rows and paired signal decisions; one c_b across each whole curve',
            conditional='Freeze c_hat and resample paired signal decisions'),
        background_note='DETBK mean fixed at chosen generator; all DETBK observations fluctuate and background is refitted',
        missing_policy='Keep failed attempts and errors; never replace draws; no calibration freeze if any planned ratio missing',
        stage_codes=dict(calibration=10,validation=11,signal=12,extend=13,hist_signal=15,technical=99),
        seed_rule='[master,0,target,round(10000*factor),stage_code,chunk]; hist_signal appends family index 0/1',
        technical_vectors_excluded=128, planned_at_utc=datetime.now(timezone.utc).isoformat())
    calibration=[dict(dataset='may22',target=0,factor=0.,n=1000,chunk=c,stage='calibration') for c in range(100)]
    validation=[dict(dataset='may22',target=0,factor=0.,n=1000,chunk=c,stage='validation') for c in range(100)]
    signal=[dict(dataset='may22',target=i,factor=f,n=1000,chunk=c,stage='signal')
            for c in range(20) for i, factors in grid.items() for f in factors]
    technical=[dict(dataset='may22', target=t, factor=f, n=32, chunk=c,
                    stage='technical', mode=mode)
               for c,(t,f,mode) in enumerate([(0,0.,'fixed'),(0,0.,'iterative'),
                                            (4,1.10,'fixed'),(4,1.10,'iterative')])]
    for name, value in [('manifest.json',manifest),('calibration_plan.json',calibration),
                        ('validation_plan.json',validation),('signal_plan.json',signal),
                        ('technical_plan.json',technical)]:
        write(name,value)
    names=['manifest.json','calibration_plan.json','validation_plan.json','signal_plan.json',
           'technical_plan.json','study.py','prepare.py']
    write('run_lock.json',{name:sha(ROOT/name) for name in names})
    print('Frozen: calibration100000; validation100000; power320000; technical128 excluded.')

if __name__ == '__main__':
    main()
