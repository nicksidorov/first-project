"""Verify raw draws, full-model fits, nominal roots and reported probabilities."""
from pathlib import Path
import hashlib
import json
import os
import sys
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('XEMAT_REPO','/workspace/first-project'))
sys.path.insert(0,str(REPO))
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm,beta
from xemat.model import JointModel
from xemat.detection import compute_detection_limits
from xemat.solvers import fit
from analysis_tools import load_groups,crossing


def verify():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    summary=json.loads((ROOT/'summary.json').read_text())
    for name,digest in manifest['production_sha256'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest
    expected=[]
    for filename in ('initial_plan.json','confirmation_plan.json','extension_plan.json'):
        p=ROOT/filename
        if p.exists():expected.extend(json.loads(p.read_text()))
    plan_keys={(v['dataset'],v['stage'],v['target'],v['factor'],v['chunk'],v['n']) for v in expected}
    assert len(plan_keys)==len(expected)
    files=sorted((ROOT/'points').glob('*.json'));assert len(files)==len(expected)
    roots=[];refits=0;fixed_iterative=0;max_root_error=0
    for dataset,data in manifest['datasets'].items():
        model=JointModel(**data['model']);S=model.physical_matrix;truth=np.array(data['truth0']);mu0=np.array(data['mu0'])
        assert np.all(truth[:10]==0)
        np.testing.assert_equal(truth[10:],model.x[14:])
        np.testing.assert_allclose(S@truth+model.b,mu0)
        assert np.linalg.matrix_rank(S/np.linalg.norm(S,axis=0))==17
        for i in range(1,5):
            def variance(mu):
                whitened=S/np.sqrt(mu)[:,None];signal=whitened[:,i];nuisance=np.delete(whitened,i,axis=1)
                nuisance/=np.linalg.norm(nuisance,axis=0)
                coeff,_,rank,_=np.linalg.lstsq(nuisance,signal,rcond=None);assert rank==16
                residual=signal-nuisance@coeff
                return 1/(residual@residual)
            lc=norm.ppf(.95)*np.sqrt(variance(mu0));mdc=data['nominal_MDC'][i-1]
            root=brentq(lambda L:L-lc-norm.ppf(.95)*np.sqrt(variance(mu0+L*S[:,i])),lc,2*mdc,xtol=1e-15)
            np.testing.assert_allclose(root,mdc,rtol=2e-8)
            np.testing.assert_allclose(lc,data['nominal_LC'][i-1],rtol=1e-10)
            error=abs(root-mdc)/root;max_root_error=max(max_root_error,float(error))
            roots.append(dict(dataset=dataset,target=i,relative_difference=float(error)))
    total=0;seed_set=set();missing=0;error_log=[]
    for path in files:
        meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));total+=meta['n']
        key=(meta['dataset'],meta['stage'],meta['target'],meta['factor'],meta['chunk'],meta['n'])
        assert key in plan_keys
        seed=tuple(meta['seed']);assert seed not in seed_set;seed_set.add(seed)
        model=JointModel(**manifest['datasets'][meta['dataset']]['model']);truth=np.array(manifest['datasets'][meta['dataset']]['truth0'])
        if meta['target']:truth[meta['target']]=meta['factor']*manifest['datasets'][meta['dataset']]['nominal_MDC'][meta['target']-1]
        np.testing.assert_equal(truth,meta['truth'])
        np.testing.assert_allclose(model.physical_matrix@truth+model.b,meta['mu'],rtol=1e-12)
        expected_counts=np.random.default_rng(seed).poisson(meta['mu'],size=(meta['n'],21))
        assert np.array_equal(a['x'],expected_counts)
        indices=np.array(meta['targets'])
        for f in ('wls','poisson'):
            decision=a[f+'_decision'];valid=decision>=0
            derived=a[f+'_free'][:,indices]>a[f+'_LC']
            assert np.array_equal(derived[valid],decision[valid]);missing+=int(np.sum(~valid))
        error_log.extend(meta['errors'])
        # Refit one independent interior row per file with all 17 parameters.
        row=meta['n']//3
        trial=JointModel(model.M,a['x'][row],model.b,model.n_sample_parameters,model.T_c,model.coordinates,model.n_detector_parameters)
        for f,pilot in [('wls','nnls'),('poisson','poisson_nonnegative')]:
            saved={}
            for label,method in [('free',f),('positive',pilot)]:
                value=fit(trial,method);assert value.success
                np.testing.assert_allclose(value.physical_parameters,a[f+'_'+label][row],rtol=1e-8,atol=1e-10)
                saved[label]=value.physical_parameters;refits+=1
            limits=compute_detection_limits(model.physical_matrix,model.b,saved['free'],saved['positive'],component_indices=meta['targets'],mode='iterative')
            np.testing.assert_allclose(limits.L_C,a[f+'_LC'][row],rtol=1e-13)
            assert np.array_equal(limits.detected,a[f+'_decision'][row]);fixed_iterative+=len(indices)
    assert total==summary['n_vectors']
    # Separate beta-quantile interval calculation and direct interpolation arithmetic.
    for curve in summary['curves']:
        for family,method in curve['methods'].items():
            for point in method['formal_points']+[method['null'],method['nominal_formula_power']]:
                k=point['k'];n=point['n_valid'];lo=0 if k==0 else beta.ppf(.025,k,n-k+1);hi=1 if k==n else beta.ppf(.975,k+1,n-k)
                np.testing.assert_allclose(point['ci95'],[lo,hi],rtol=1e-10,atol=1e-12)
            pts=method['formal_points']
            pair=next((a,b) for a,b in zip(pts[:-1],pts[1:]) if a['rate']<=.95<=b['rate'] and b['rate']>a['rate'])
            a,b=pair
            root=(a['activity_mBq_m3']*(b['rate']-.95)+b['activity_mBq_m3']*(.95-a['rate']))/(b['rate']-a['rate'])
            np.testing.assert_allclose(root,method['ld_mBq_m3'],rtol=1e-12)
    output=dict(status='passed',vectors=total,files=len(files),unique_seeds=len(seed_set),
        production_hashes_unchanged=True,source_background_from_DETBK_only=True,all_seeds_reproduced=True,
        full_matrix_rank=17,refitted_vectors=len(files),refitted_models=refits,
        lc_fixed_iterative_agreements=fixed_iterative,
        nominal_independent_roots=roots,max_nominal_root_relative_difference=max_root_error,
        missing_decisions=missing,recorded_errors=error_log,intervals_and_interpolations_recomputed=True)
    (ROOT/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='nominal_independent_roots'},ensure_ascii=False,indent=2))


if __name__=='__main__':verify()
