"""Sampling distributions of the standard iterative, observation-based MDC.

The calibrated Ld95 is a fixed generating activity in the signal series.
It does not multiply the formula: compute_detection_limits uses alpha=beta=.05
and the unchanged production iterative procedure in every trial.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('XEMAT_REPO','/workspace/first-project'))
SOURCE=Path(os.environ.get('XEMAT_PREVIOUS_STUDY','/workspace/scratch/may-clean-mdc-eff221-20261022'))
sys.path.insert(0,str(REPO))
import numpy as np
import scipy
from xemat.model import JointModel
from xemat.solvers import fit
from xemat.detection import compute_detection_limits

MASTER=2026102501
DATES=('may22','may26')
FAMILIES={'wls':'nnls','poisson':'poisson_nonnegative'}
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare():
    """Lock all means, targets, seeds and source hashes before new sampling."""
    for name in ('manifest.json','plan.json','technical_plan.json','run_lock.json'):
        if (ROOT/name).exists():raise FileExistsError(f'Refusing to replace planned file: {name}')
    old=read(SOURCE/'manifest.json');summary=read(SOURCE/'summary.json')
    datasets={}
    for dataset in DATES:
        original=old['datasets'][dataset]
        settings={}
        for target,isotope in ISOTOPES.items():
            c=next(c for c in summary['curves'] if c['dataset']==dataset and c['target']==target)
            settings[str(target)]={'isotope':isotope,'formula_MDC_Bq_m3':c['nominal_MDC_mBq_m3']/1000,
                'methods':{f:{'signal_activity_Bq_m3':c['methods'][f]['calibrated_ld_mBq_m3']/1000,
                              'c':c['methods'][f]['calibration']['c']} for f in FAMILIES}}
        datasets[dataset]={'model':original['model'],'truth0':original['truth0'],
            'nominal_MDC_Bq_m3':original['nominal_MDC'],'settings':settings,
            'effective_air_volume_m3':original['metadata']['effective_air_volume_m3'],
            'time_parameters':original['metadata']['time_parameters']}
    sources={}
    for dataset in DATES:
        for chunk in range(10):
            stem=f'{dataset}_validation_i0_f00000_c{chunk}'
            sources[stem]={extension:sha(SOURCE/'points'/(stem+'.'+extension)) for extension in ('json','npz')}
    manifest={'scope':'Distribution of standard iterative plug-in MDC at A=0 and fixed calibrated Ld95',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'datasets':datasets,
        'efficiencies':old['efficiency_update'],'production_sha256':old['production_sha256'],
        'production_commit':old['production_commit'],
        'source_manifest_sha256':sha(SOURCE/'manifest.json'),'source_summary_sha256':sha(SOURCE/'summary.json'),
        'null_source_sha256':sources,'coordinates':'[a_V; gamma; zeta]',
        'raw_activity_unit':'Bq/m3','alpha':.05,'beta_ii':.05,'mode':'iterative',
        'mdc_formula':'Unmodified production iterative MDC, not multiplied by c',
        'signal_truth':'Fixed calibrated_ld_mBq_m3 / 1000 for the specific date, isotope and estimator',
        'null_reuse':'10000 independent-validation vectors per date, complete free and positive fits reused without refitting',
        'null_n_per_date':10000,'signal_n_per_date_isotope_method':3000,
        'new_signal_vectors':48000,'reused_null_vectors':20000,
        'signal_master_seed':MASTER,
        'signal_seed_fields':['master','date_index','target','family_index','chunk'],
        'technical_seed_extra':99,'technical_vectors_excluded':12,
        'systematics':False,'poisson_blocks':['SAMPLE','GASBK','DETBK'],
        'n_observations':21,'n_estimated_parameters':17,
        'missing_policy':'Keep NaN/nonconvergence and explicit errors; never replace failures or regenerate observations',
        'numpy':np.__version__,'scipy':scipy.__version__}
    plan=[]
    for chunk in range(10):
        for dataset in DATES:plan.append({'dataset':dataset,'stage':'null','target':0,'family':'both','n':1000,'chunk':chunk})
    for chunk in range(3):
        for target in ISOTOPES:
            for dataset in DATES:
                for family in FAMILIES:plan.append({'dataset':dataset,'stage':'signal_ld','target':target,'family':family,'n':1000,'chunk':chunk})
    technical=[{'dataset':dataset,'stage':'technical','target':4,'family':family,'n':3,'chunk':0}
               for dataset in DATES for family in FAMILIES]
    write(ROOT/'manifest.json',manifest);write(ROOT/'plan.json',plan);write(ROOT/'technical_plan.json',technical)
    write(ROOT/'run_lock.json',{name:sha(ROOT/name) for name in ('manifest.json','plan.json','technical_plan.json','hist_compute.py')})
    print('Prepared and locked: 20000 reused null vectors; 48000 new signal vectors; 12 excluded technical vectors.',flush=True)


def checked_fit(model,method):
    result=fit(model,method)
    if not result.success:raise RuntimeError(result.message)
    if result.physical_parameters is None or not np.all(np.isfinite(result.physical_parameters)):
        raise ValueError('Physical estimates are unavailable or nonfinite')
    return result


def name_of(job):
    return f"{job['dataset']}_{job['stage']}_i{job['target']}_{job['family']}_c{job['chunk']}"


def allocate(n,targets,families):
    a={'indices':np.arange(n),'targets':np.asarray(targets,dtype=int)}
    for family in families:
        for key in ('free','positive'):a[family+'_'+key]=np.full((n,17),np.nan)
        for key in ('LC','MDC','sigma0','sigma_at_MDC'):a[family+'_'+key]=np.full((n,len(targets)),np.nan)
        a[family+'_iterations']=np.zeros((n,len(targets)),dtype=np.int32)
        a[family+'_converged']=np.zeros((n,len(targets)),dtype=bool)
        a[family+'_free_covariance_available']=np.zeros(n,dtype=bool)
        for key in ('decision_standard','decision_calibrated'):
            a[family+'_'+key]=np.full((n,len(targets)),-1,dtype=np.int8)
    return a


def run_job(dataset,stage,target,family,n,chunk):
    began=time.monotonic();job=dict(dataset=dataset,stage=stage,target=target,family=family,n=n,chunk=chunk)
    name=name_of(job);folder=ROOT/('technical' if stage=='technical' else 'points');folder.mkdir(exist_ok=True)
    if (folder/(name+'.npz')).exists() or (folder/(name+'.json')).exists():raise FileExistsError(name)
    manifest=read(ROOT/'manifest.json');d=manifest['datasets'][dataset];model=JointModel(**d['model'])
    truth=np.asarray(d['truth0'],dtype=float).copy();targets=list(ISOTOPES) if stage=='null' else [target]
    families=list(FAMILIES) if family=='both' else [family]
    arrays=allocate(n,targets,families);source=None
    if stage=='null':
        stem=f'{dataset}_validation_i0_f00000_c{chunk}'
        expected=manifest['null_source_sha256'][stem]
        for ext,digest in expected.items():assert sha(SOURCE/'points'/(stem+'.'+ext))==digest
        source_meta=read(SOURCE/'points'/(stem+'.json'))
        assert source_meta['n']==n and source_meta['stage']=='validation' and not source_meta['errors']
        with np.load(SOURCE/'points'/(stem+'.npz')) as loaded:source={key:loaded[key].copy() for key in loaded.files}
        arrays['x']=source['x'];seed=source_meta['seed']
        mu=model.physical_matrix@truth+model.b
        np.testing.assert_array_equal(arrays['x'],np.random.default_rng(seed).poisson(mu,size=(n,21)))
        for f in families:
            for key in ('free','positive','free_covariance_available'):arrays[f+'_'+key]=source[f+'_'+key].copy()
    else:
        truth[target]=d['settings'][str(target)]['methods'][family]['signal_activity_Bq_m3']
        mu=model.physical_matrix@truth+model.b
        seed=[MASTER,DATES.index(dataset),target,list(FAMILIES).index(family),chunk]
        if stage=='technical':seed.append(99)
        arrays['x']=np.random.default_rng(seed).poisson(mu,size=(n,21))
    assert np.all(mu>0)
    errors=[]
    for row,x in enumerate(arrays['x']):
        trial=JointModel(model.M,x,model.b,model.n_sample_parameters,model.T_c,model.coordinates,model.n_detector_parameters)
        for f in families:
            if stage!='null':
                phase='free'
                try:
                    free=checked_fit(trial,f);arrays[f+'_free'][row]=free.physical_parameters
                    arrays[f+'_free_covariance_available'][row]=free.physical_covariance is not None
                    phase='positive';positive=checked_fit(trial,FAMILIES[f]);arrays[f+'_positive'][row]=positive.physical_parameters
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    errors.append({'row':row,'family':f,'phase':phase,'error':f'{type(exc).__name__}: {exc}'})
                    continue
            for column,index in enumerate(targets):
                try:
                    result=compute_detection_limits(model.physical_matrix,model.b,arrays[f+'_free'][row],
                        arrays[f+'_positive'][row],component_indices=[index],alpha=.05,beta_ii=.05,mode='iterative')
                    for key,result_key in [('LC','L_C'),('MDC','MDC'),('sigma0','sigma0'),('sigma_at_MDC','sigma_at_MDC'),
                                           ('iterations','iterations'),('converged','converged')]:
                        arrays[f+'_'+key][row,column]=getattr(result,result_key)[0]
                    arrays[f+'_decision_standard'][row,column]=result.detected[0]
                    c=d['settings'][str(index)]['methods'][f]['c']
                    arrays[f+'_decision_calibrated'][row,column]=arrays[f+'_free'][row,index]>c*result.L_C[0]
                    if stage=='null':
                        np.testing.assert_allclose(result.L_C[0],source[f+'_LC'][row,column],rtol=2e-12,atol=1e-15)
                        assert arrays[f+'_decision_standard'][row,column]==source[f+'_decision'][row,column]
                    if not result.converged[0]:
                        errors.append({'row':row,'family':f,'phase':'MDC_nonconvergence','target':index,
                                       'iterations':int(result.iterations[0]),'error':'; '.join(result.warnings)})
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    errors.append({'row':row,'family':f,'phase':'MDC','target':index,'error':f'{type(exc).__name__}: {exc}'})
    meta={**job,'targets':targets,'families':families,'seed':seed,'truth':truth.tolist(),'mu':mu.tolist(),
          'raw_activity_unit':'Bq/m3','mode':'iterative','alpha':.05,'beta_ii':.05,
          'manifest_sha256':sha(ROOT/'manifest.json'),'errors':errors,'elapsed_seconds':time.monotonic()-began,
          'reused_null_fits':stage=='null'}
    if stage=='null':meta['source_stem']=stem;meta['source_sha256']=expected
    meta['valid_MDC']={f:int(np.count_nonzero(np.isfinite(arrays[f+'_MDC'])&arrays[f+'_converged'])) for f in families}
    temp=folder/(name+'.tmp.npz');np.savez_compressed(temp,**arrays);os.replace(temp,folder/(name+'.npz'))
    meta['npz_sha256']=sha(folder/(name+'.npz'));write(folder/(name+'.json'),meta)
    print('DONE',name,'n',n,'seconds',round(meta['elapsed_seconds'],1),'errors',len(errors),'valid_MDC',meta['valid_MDC'],flush=True)
    return name


def execute(plan_path,workers):
    for name,digest in read(ROOT/'run_lock.json').items():assert sha(ROOT/name)==digest,name
    manifest=read(ROOT/'manifest.json')
    for name,digest in manifest['production_sha256'].items():assert sha(REPO/name)==digest,name
    jobs=read(plan_path);pending=[]
    for job in jobs:
        folder=ROOT/('technical' if job['stage']=='technical' else 'points');stem=name_of(job)
        metadata=folder/(stem+'.json');raw=folder/(stem+'.npz')
        if metadata.exists() and raw.exists():
            meta=read(metadata)
            assert all(meta[k]==v for k,v in job.items()) and meta['manifest_sha256']==sha(ROOT/'manifest.json')
            assert meta['npz_sha256']==sha(raw)
            continue
        pending.append(job)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_job,**job) for job in pending]
        for future in as_completed(futures):future.result()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.prepare:prepare()
    if args.plan:execute(args.plan,args.workers)
