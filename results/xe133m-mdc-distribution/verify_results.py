"""Independent numerical checks of saved counts and iterative MDC solutions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('XEMAT_REPO','/workspace/first-project'))
sys.path.insert(0,str(REPO))

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from xemat.model import JointModel
from xemat.detection import compute_detection_limits
from xemat.solvers import fit


def scalar_variance(matrix,mu):
    """Schur complement by residual projection, not production SVD inversion."""
    whitened=matrix/np.sqrt(mu)[:,None]
    target=whitened[:,4]
    nuisance=np.delete(whitened,4,axis=1)
    nuisance=nuisance/np.linalg.norm(nuisance,axis=0)
    coefficient,_,rank,_=np.linalg.lstsq(nuisance,target,rcond=None)
    assert rank==16
    residual=target-nuisance@coefficient
    return 1/(residual@residual)


def main(validation):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['production_sha256'].items():
        assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest
    folder=ROOT/('validation' if validation else 'chunks')
    files=sorted(folder.glob('*.npz'))
    assert len(files)==(4 if validation else 40),(len(files),validation)
    seen={}
    checks=[]
    maximum_relative_error=0.0
    maximum_residual=0.0
    failures=[]
    fit_spot_checks=0
    for path in files:
        meta=json.loads(path.with_suffix('.json').read_text())
        data=manifest['datasets'][meta['dataset']]
        model=JointModel(**data['model'])
        truth=np.array(data['truth0']);truth[4]=meta['activity']
        mu=model.physical_matrix@truth+model.b
        np.testing.assert_allclose(mu,meta['mu'],rtol=1e-13)
        a=np.load(path)
        indices=a['indices']
        assert np.array_equal(indices,np.arange(meta['start'],meta['stop']))
        key=meta['dataset']+'_'+meta['scenario']
        seen.setdefault(key,[]).extend(indices.tolist())
        expected=np.random.default_rng(meta['seed']).poisson(mu,size=(10000,21))[indices]
        assert np.array_equal(expected,a['x'])
        failures.extend(meta['errors'])
        for family in manifest['families']:
            mdc=a[family+'_MDC']; lc=a[family+'_LC']
            valid=np.isfinite(mdc)
            assert int(valid.sum())==meta['n_MDC_valid'][family]
            assert np.all(mdc[valid]>lc[valid])
            decision=a[family+'_detected']
            known=decision>=0
            assert np.array_equal(decision[known],a[family+'_free'][known,4]>lc[known])
            residual=a[family+'_equation_relative_residual'][valid]
            if len(residual):maximum_residual=max(maximum_residual,float(np.max(residual)))
            # Check endpoints and both MDC tails in every saved chunk.
            candidates=np.flatnonzero(valid)
            sample=np.unique(np.r_[candidates[0],candidates[-1],candidates[np.argmin(mdc[valid])],candidates[np.argmax(mdc[valid])]]) if len(candidates) else []
            for i in sample:
                positive=a[family+'_positive'][i].copy();positive[4]=0
                mu0=model.physical_matrix@positive+model.b
                z=norm.ppf(.95)
                expected_lc=z*np.sqrt(scalar_variance(model.physical_matrix,mu0))
                np.testing.assert_allclose(lc[i],expected_lc,rtol=1e-10)
                def equation(value):
                    return value-expected_lc-z*np.sqrt(scalar_variance(model.physical_matrix,mu0+value*model.physical_matrix[:,4]))
                root=brentq(equation,expected_lc,2*mdc[i],xtol=1e-15,rtol=1e-13)
                relative=abs(root-mdc[i])/root
                assert relative<2e-8,(path,family,i,relative)
                maximum_relative_error=max(maximum_relative_error,float(relative))
                checks.append(dict(file=path.name,family=family,index=int(indices[i]),relative_difference=float(relative)))
        # Independently rerun all four fits on a non-tail row in every chunk.
        row=len(indices)//3
        trial=JointModel(model.M,a['x'][row],model.b,model.n_sample_parameters,
                         model.T_c,model.coordinates,model.n_detector_parameters)
        for family,positive_method in manifest['families'].items():
            fitted={}
            for label,method in [('free',family),('positive',positive_method)]:
                value=fit(trial,method)
                assert value.success
                np.testing.assert_allclose(value.physical_parameters,a[family+'_'+label][row],rtol=1e-8,atol=1e-10)
                fitted[label]=value.physical_parameters
                fit_spot_checks+=1
            full=compute_detection_limits(model.physical_matrix,model.b,fitted['free'],fitted['positive'],component_indices=[1,2,3,4],mode='iterative')
            np.testing.assert_allclose(full.MDC[-1],a[family+'_MDC'][row],rtol=1e-12)
            assert full.converged[-1]
    expected_n=100 if validation else 10000
    assert len(seen)==4
    for key,indices in seen.items():
        assert sorted(indices)==list(range(expected_n)),key
    assert maximum_residual<1e-8
    output=dict(status='passed',validation=validation,raw_vectors=sum(map(len,seen.values())),
        number_of_chunks=len(files),production_hashes_unchanged=True,
        all_counts_reproduced_from_seeds=True,all_indices_disjoint_and_complete=True,
        independent_projection_brentq_checks=len(checks),maximum_relative_root_difference=maximum_relative_error,
        maximum_equation_relative_residual=maximum_residual,refitted_parameter_vectors=fit_spot_checks,
        full_four_isotope_call_matches_target_only=True,recorded_errors=failures,checks=checks)
    out=ROOT/('validation_review.json' if validation else 'verification.json')
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k!='checks'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation',action='store_true')
    main(p.parse_args().validation)
