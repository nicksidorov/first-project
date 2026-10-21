"""Nominal detection on pure DETBK background, with the full XeMat model.

No parameters are removed from estimation. Only generating sample and gas
activities are zero. Detector background is estimated anew in every fit.
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
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('XEMAT_REPO','/workspace/first-project'))
sys.path.insert(0,str(REPO))
import numpy as np
import scipy
from scipy.stats import binomtest
from xemat.model import JointModel
from xemat.detection import compute_detection_limits
from xemat.solvers import fit

OLD=Path('/workspace/scratch/may-xe133m-ld-power')
FAMILIES={'wls':'nnls','poisson':'poisson_nonnegative'}
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}
MASTER=2026102201


def checked_fit(model,method):
    result=fit(model,method)
    if not result.success:raise RuntimeError(result.message)
    if result.physical_parameters is None or not np.all(np.isfinite(result.physical_parameters)):
        raise ValueError('Invalid physical parameters')
    return result


def prepare():
    assert not (ROOT/'manifest.json').exists()
    old=json.loads((OLD/'manifest.json').read_text())
    review=json.loads((OLD/'inputs_independent_review.json').read_text())
    for key,hash_value in old['production_sha256'].items():
        assert hashlib.sha256((REPO/key).read_bytes()).hexdigest()==hash_value
    datasets={}
    for name,entry in old['datasets'].items():
        source=json.loads(Path(entry['source']).read_text())
        assert source['use_fixed_roi'] and source['use_fixed_efficiencies']
        model=JointModel(**source['model']);metadata=source['metadata']
        audit=review[Path(entry['source']).parent.name]
        for key,path in audit['paths'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==audit['input_hashes'][key]
        truth=np.zeros(17);truth[10:]=model.x[14:]
        np.testing.assert_equal(truth[10:],[metadata['detector_roi_counts'][str(i)] for i in [1,2,3,5,6,7,8]])
        S=model.physical_matrix
        assert S.shape==(21,17)
        np.testing.assert_equal(S[14:,10:],np.eye(7))
        np.testing.assert_equal(S[14:,:10],np.zeros((7,10)))
        np.testing.assert_equal(S[7:,:5],np.zeros((14,5)))
        mu=S@truth+model.b
        np.testing.assert_allclose(mu[:7],metadata['kappa_s']*truth[10:])
        np.testing.assert_allclose(mu[7:14],metadata['kappa_g']*truth[10:])
        limits=compute_detection_limits(S,model.b,truth,truth,mode='iterative')
        assert np.all(limits.converged)
        # All fitted parameters remain in the Fisher covariance, even at zero truth.
        datasets[name]=dict(model=source['model'],truth0=truth.tolist(),mu0=mu.tolist(),
            nominal_LC=limits.L_C.tolist(),nominal_MDC=limits.MDC.tolist(),
            nominal_sigma0=limits.sigma0.tolist(),nominal_sigma_at_MDC=limits.sigma_at_MDC.tolist(),
            iterations=limits.iterations.tolist(),metadata=metadata,
            input_paths=audit['paths'],input_sha256=audit['input_hashes'],
            calibration_matrices=source['calibration_matrices'],
            source=entry['source'],source_sha256=hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest())
    manifest=dict(scope='Nominal MDC on pure detector background; four isotopes injected one at a time',
        datasets=datasets,isotopes=ISOTOPES,families=FAMILIES,master_seed=MASTER,
        alpha=.05,beta_ii=.05,unit='Bq/m3; output tables and plots use mBq/m3',
        coordinates='[a_V(5); gamma(5); zeta(7)]',
        generator='zeta equals original DETBK ROI counts; all other truth parameters zero except the injected fresh isotope',
        sampling='All 21 SAMPLE/GASBK/DETBK counts independently Poisson; all 17 parameters refitted',
        lc_rule='Unchanged production Lc, nonnegative pilot with target subtraction, free estimate > Lc',
        mc_mdc_mode='fixed: only Lc used in power simulation; iterative nominal MDC calculated separately',
        initial_plan=dict(null_n=5000,screen_factors=[.8,1.,1.2,1.4],screen_n=500),
        confirmation='After screening choose three common factors per isotope/dataset, 3000 independent repetitions each; lock entire plan before confirmation',
        extension_policy='If confirmed points do not bracket .95 for either estimator, declare and run independent additional points; never discard existing ones',
        production_sha256=old['production_sha256'],production_commit=old['git_head'],
        numpy=np.__version__,scipy=scipy.__version__,systematics=False,
        background_note='Original real SAMPLE/GASBK counts are not used to estimate the generating background')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    jobs=[]
    for name in datasets:
        for j in range(5):jobs.append(dict(dataset=name,target=0,factor=0,n=1000,chunk=j,stage='null'))
        for target in ISOTOPES:
            for factor in manifest['initial_plan']['screen_factors']:
                jobs.append(dict(dataset=name,target=target,factor=factor,n=500,chunk=0,stage='screen'))
    (ROOT/'initial_plan.json').write_text(json.dumps(jobs,indent=2))
    print(json.dumps({k:dict(LC_mBq=np.array(v['nominal_LC']).__mul__(1000).tolist(),
                            MDC_mBq=np.array(v['nominal_MDC']).__mul__(1000).tolist()) for k,v in datasets.items()},indent=2))


def run_point(dataset,target,factor,n,chunk=0,stage='confirm'):
    began=time.monotonic()
    manifest=json.loads((ROOT/'manifest.json').read_text());data=manifest['datasets'][dataset]
    model=JointModel(**data['model']);truth=np.array(data['truth0'])
    if target:truth[target]=factor*data['nominal_MDC'][target-1]
    mu=model.physical_matrix@truth+model.b
    assert np.all(mu>0)
    code=int(round(factor*10000));assert np.isclose(code/10000,factor,atol=1e-12,rtol=0)
    seed=[MASTER,list(manifest['datasets']).index(dataset),target,code,
          {'null':0,'screen':1,'confirm':2,'validate':3,'extend':4}[stage],chunk]
    counts=np.random.default_rng(seed).poisson(mu,size=(n,21))
    indices=list(ISOTOPES) if target==0 else [target]
    stem=f'{dataset}_{stage}_i{target}_f{code:05d}_c{chunk}'
    folder=ROOT/('validation' if stage=='validate' else 'points');folder.mkdir(exist_ok=True)
    if (folder/(stem+'.npz')).exists():raise FileExistsError(stem)
    arrays=dict(x=counts,indices=np.arange(n),targets=np.array(indices,dtype=int))
    for family in FAMILIES:
        for key in ('free','positive'):arrays[family+'_'+key]=np.full((n,17),np.nan)
        arrays[family+'_LC']=np.full((n,len(indices)),np.nan)
        arrays[family+'_decision']=np.full((n,len(indices)),-1,dtype=np.int8)
        arrays[family+'_free_covariance_available']=np.zeros(n,dtype=bool)
    errors=[]
    for row,x in enumerate(counts):
        trial=JointModel(model.M,x,model.b,model.n_sample_parameters,model.T_c,
                         model.coordinates,model.n_detector_parameters)
        for family,pilot_method in FAMILIES.items():
            phase='free'
            try:
                free=checked_fit(trial,family)
                arrays[family+'_free'][row]=free.physical_parameters
                arrays[family+'_free_covariance_available'][row]=free.physical_covariance is not None
                phase='positive';positive=checked_fit(trial,pilot_method)
                arrays[family+'_positive'][row]=positive.physical_parameters
                phase='LC'
                detection=compute_detection_limits(model.physical_matrix,model.b,
                    free.physical_parameters,positive.physical_parameters,component_indices=indices,mode='fixed')
                arrays[family+'_LC'][row]=detection.L_C
                arrays[family+'_decision'][row]=detection.detected.astype(np.int8)
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                errors.append(dict(row=row,family=family,phase=phase,error=f'{type(exc).__name__}: {exc}'))
    metadata=dict(dataset=dataset,target=target,targets=indices,factor=factor,n=n,stage=stage,
        chunk=chunk,seed=seed,truth=truth.tolist(),mu=mu.tolist(),errors=errors,elapsed_seconds=time.monotonic()-began)
    metadata['rates']={}
    for family in FAMILIES:
        metadata['rates'][family]=[]
        for col in range(len(indices)):
            v=arrays[family+'_decision'][:,col];valid=v>=0;total=int(valid.sum());k=int(v[valid].sum())
            ci=binomtest(k,total).proportion_ci(.95,method='exact') if total else None
            metadata['rates'][family].append(dict(n_valid=total,n_missing=n-total,k=k,
                rate=k/total if total else None,ci95=[ci.low,ci.high] if ci else None))
    np.savez_compressed(folder/(stem+'.npz'),**arrays)
    (folder/(stem+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print('DONE',stem,'n',n,'seconds',round(metadata['elapsed_seconds'],1),'errors',len(errors),
          'rates',{f:[round(v['rate'],4) if v['rate'] is not None else None for v in metadata['rates'][f]] for f in FAMILIES},flush=True)
    return stem


def plan_run(path,workers):
    jobs=json.loads(Path(path).read_text())
    pending=[]
    for job in jobs:
        stem=f"{job['dataset']}_{job.get('stage','confirm')}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job.get('chunk',0)}"
        folder=ROOT/('validation' if job.get('stage')=='validate' else 'points')
        if (folder/(stem+'.npz')).exists() and (folder/(stem+'.json')).exists():continue
        pending.append(job)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_point,**job) for job in pending]
        for future in as_completed(futures):future.result()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--plan');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:plan_run(args.plan,args.workers)
