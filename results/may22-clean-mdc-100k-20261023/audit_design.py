"""Independent preflight of the locked design; no simulation or fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get('XEMAT_REPO', '/workspace/first-project'))
SOURCE = Path('/workspace/scratch/may-clean-mdc-eff221-20261022')
sys.path.insert(0, str(REPO))
from xemat.model import JointModel

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_limits(S, b, truth, target):
    nuisance = np.array(truth, dtype=float); nuisance[target] = 0
    mu = b + S@nuisance
    def sigma(activity):
        weighted = S/np.sqrt(mu+activity*S[:,target])[:,None]
        scales = np.linalg.norm(weighted,axis=0)
        _, R = np.linalg.qr(weighted/scales,mode='reduced')
        inverse = solve_triangular(R,np.eye(len(scales)))
        return float(np.linalg.norm(inverse[target])/scales[target])
    z = norm.ppf(.95); lc = z*sigma(0)
    def residual(a): return a-lc-z*sigma(a)
    upper = 2*lc
    while residual(upper) <= 0: upper *= 2
    ld = brentq(residual,0,upper,xtol=1e-15,rtol=2e-13)
    return dict(target=target,LC=lc,MDC=ld,sigma0=sigma(0),sigma_at_MDC=sigma(ld))

def main():
    manifest=read(ROOT/'manifest.json'); lock=read(ROOT/'run_lock.json')
    for name,digest in lock.items(): assert sha(ROOT/name)==digest,name
    assert set(lock)=={'manifest.json','calibration_plan.json','validation_plan.json',
                      'signal_plan.json','technical_plan.json','study.py','prepare.py',
                      'statistical_design.md','statistical_design_review.md'}
    for name,digest in manifest['production_sha256'].items(): assert sha(REPO/name)==digest,name
    original=read(SOURCE/'manifest.json')
    assert sha(SOURCE/'manifest.json')==manifest['source_manifest_sha256']
    assert sha(SOURCE/'summary.json')==manifest['source_summary_sha256']
    assert manifest['master_seed']==2026102701 and manifest['bootstrap']['master_seed']==2026102709
    assert manifest['master_seed']!=original['master_seed']
    assert list(manifest['datasets'])==['may22']
    assert manifest['families']==['wls','poisson']
    assert manifest['alpha']==manifest['beta_ii']==.05 and not manifest['systematics']
    assert manifest['calibration_n_per_dataset']==manifest['validation_n_per_dataset']==100000
    assert manifest['signal_n_per_point']==manifest['histogram_signal_n_per_isotope_method']==20000
    assert manifest['bootstrap']['repetitions']==2000 and manifest['bootstrap']['paired']
    assert manifest['stage_codes']==dict(calibration=10,validation=11,signal=12,extend=13,hist_signal=15,technical=99)
    extension=manifest['extension_policy']
    assert extension['max_extra_points_per_isotope']==4
    assert extension['n_per_point']==20000 and extension['outward_step']==.04
    assert extension['max_points_including_null']==9
    grid={'1':[.96,1.,1.04,1.08],'2':[.96,1.,1.04,1.08],
          '3':[1.02,1.06,1.10,1.14],'4':[1.02,1.06,1.10,1.14]}
    assert manifest['signal_plan']==grid
    data=manifest['datasets']['may22']; old=original['datasets']['may22']
    for key in ('model','truth0','metadata','calibration_matrices','input_paths','input_sha256'):
        assert data[key]==old[key],key
    model=JointModel(**data['model']); S=model.physical_matrix
    assert S.shape==(21,17) and model.physical_coordinate_label=='[a_V; gamma; zeta]'
    np.testing.assert_array_equal(data['truth0'],[0.]*10+[522.,854.,494.,66.,64.,89.,97.])
    np.testing.assert_array_equal(model.b,np.zeros(21))
    np.testing.assert_array_equal(data['mu0'],S@data['truth0']+model.b)
    for name in ('sample_efficiencies','gasbk_efficiencies'):
        np.testing.assert_array_equal(data['metadata'][name],[.241169,.367784,.233937,.233937])
    checks=[]
    for i in range(1,5):
        row=independent_limits(S,model.b,data['truth0'],i)
        for key,field in [('LC','nominal_LC'),('MDC','nominal_MDC'),('sigma0','nominal_sigma0'),('sigma_at_MDC','nominal_sigma_at_MDC')]:
            np.testing.assert_allclose(row[key],data[field][i-1],rtol=3e-8,atol=1e-14)
        checks.append(row)
    counts={}; all_seeds=set(); names=set()
    for stage,count in [('calibration',100000),('validation',100000),('signal',320000),('technical',128)]:
        jobs=read(ROOT/(stage+'_plan.json'))
        assert sum(j['n'] for j in jobs)==count
        keys=[]
        for j in jobs:
            assert j['dataset']=='may22' and j['stage']==stage
            assert j['n']==(32 if stage=='technical' else 1000)
            factor_code=round(10000*j['factor']); assert abs(factor_code/10000-j['factor'])<1e-12
            seed=(manifest['master_seed'],0,j['target'],factor_code,manifest['stage_codes'][stage],j['chunk'])
            assert seed not in all_seeds; all_seeds.add(seed)
            stem=f"may22_{stage}_i{j['target']}_f{factor_code:05d}_c{j['chunk']}"
            assert stem not in names; names.add(stem)
            keys.append((j['target'],j['factor'],j['chunk']))
        if stage in ('calibration','validation'):
            assert set(keys)=={(0,0.,c) for c in range(100)}
        elif stage=='signal':
            assert set(keys)=={(i,f,c) for i in range(1,5) for f in grid[str(i)] for c in range(20)}
        else:
            assert len(jobs)==4
            assert [j['mode'] for j in jobs]==['fixed','iterative','fixed','iterative']
        counts[stage]=dict(vectors=count,batches=len(jobs))
    result=dict(status='passed',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(ROOT/'manifest.json'),run_lock_sha256=sha(ROOT/'run_lock.json'),
        audit_file_sha256=sha(Path(__file__)),method='Independent locked plan comparison and scaled QR + brentq formula roots',
        source_model_exact_match=True,seeds_checked=len(all_seeds),stage_counts=counts,
        calibration_order_rank=95001,expected_training_exceedances_without_ties=4999,
        nominal_formula_checks=checks,planned_formal_vectors_before_histograms=520000,
        planned_histogram_signal_vectors=160000,planned_iterative_MDC_attempts=960000,
        max_signal_points_per_isotope=8,systematics=False,wald=False)
    (ROOT/'design_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
