"""Calibrating nominal Lc on pure DETBK background with the full XeMat model.

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

FAMILIES={"wls":"nnls","poisson":"poisson_nonnegative"}
ISOTOPES={1:"Xe135",2:"Xe133",3:"Xe131m",4:"Xe133m"}
MASTER=2026102301


def checked_fit(model,method):
    result=fit(model,method)
    if not result.success:raise RuntimeError(result.message)
    if result.physical_parameters is None or not np.all(np.isfinite(result.physical_parameters)):
        raise ValueError('Invalid physical parameters')
    return result


def run_point(dataset,target,factor,n,chunk=0,stage='calibration'):
    began=time.monotonic()
    manifest=json.loads((ROOT/'manifest.json').read_text());data=manifest['datasets'][dataset]
    model=JointModel(**data['model']);truth=np.array(data['truth0'])
    if target:truth[target]=factor*data['nominal_MDC'][target-1]
    mu=model.physical_matrix@truth+model.b
    assert np.all(mu>0)
    code=int(round(factor*10000));assert np.isclose(code/10000,factor,atol=1e-12,rtol=0)
    seed=[MASTER,list(manifest['datasets']).index(dataset),target,code,
          {'calibration':10,'validation':11,'extend':12,'tech':13}[stage],chunk]
    counts=np.random.default_rng(seed).poisson(mu,size=(n,21))
    indices=list(ISOTOPES) if target==0 else [target]
    stem=f'{dataset}_{stage}_i{target}_f{code:05d}_c{chunk}'
    folder=ROOT/('technical' if stage=='tech' else 'points');folder.mkdir(exist_ok=True)
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
        stem=f"{job['dataset']}_{job.get('stage','calibration')}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job.get('chunk',0)}"
        folder=ROOT/('technical' if job.get('stage')=='tech' else 'points')
        if (folder/(stem+'.npz')).exists() and (folder/(stem+'.json')).exists():continue
        pending.append(job)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_point,**job) for job in pending]
        for future in as_completed(futures):future.result()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    plan_run(args.plan,args.workers)
