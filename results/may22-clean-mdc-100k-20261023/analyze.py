"""Independent calibration/validation and paired power-curve inference.

The original Lc is recomputed by the unchanged production core in study.py.
Here only the dimensionless cutoff R=free activity/Lc is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')
DATES=('may22',)
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}
B=2000
BOOTSTRAP_MASTER=2026102709
CALIBRATION_N=100000
VALIDATION_N=100000
SIGNAL_N_PER_POINT=20000
INITIAL_SIGNAL_N=320000


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def cp_rate(r,cutoff,confidence=.95):
    """Keep missing observations in explicit bounds; never turn them into negatives."""
    r=np.asarray(r);valid=np.isfinite(r);N=len(r);n=int(valid.sum());k=int(np.sum(r[valid]>cutoff))
    if n==0:raise RuntimeError('No valid ratios')
    ci=binomtest(k,n).proportion_ci(confidence,method='exact')
    low=binomtest(k,N).proportion_ci(confidence,method='exact').low
    high=binomtest(k+N-n,N).proportion_ci(confidence,method='exact').high
    return dict(n=N,n_valid=n,missing=N-n,k=k,rate=k/n,ci95=[float(ci.low),float(ci.high)],
                bounds=[k/N,(k+N-n)/N],conservative_ci=[float(low),float(high)])


def raw_batches(folder,stages):
    for path in sorted((folder/'points').glob('*.json')):
        if not any('_'+stage+'_' in path.stem for stage in stages):continue
        meta=json.loads(path.read_text())
        if meta['stage'] not in stages:continue
        assert meta['dataset'] in DATES
        assert meta['manifest_sha256']==hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()
        assert meta['npz_sha256']==hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        with np.load(path.with_suffix('.npz')) as data:
            assert data['x'].shape==(meta['n'],21)
            np.testing.assert_array_equal(data['targets'],meta['targets'])
            np.testing.assert_array_equal(data['indices'],np.arange(meta['n']))
            free=np.stack([data[f+'_free'][:,np.array(meta['targets'])] for f in FAMILIES],axis=1)
            lc=np.stack([data[f+'_LC'] for f in FAMILIES],axis=1)
            valid=np.isfinite(free)&np.isfinite(lc)&(lc>0)
            ratio=np.divide(free,lc,out=np.full(free.shape,np.nan),where=valid)
            for j,f in enumerate(FAMILIES):
                decision=data[f+'_decision'];good=decision>=0
                assert np.array_equal((ratio[:,j,:]>1)[good],decision[good])
        yield path,meta,ratio


def signal_groups():
    """All signal scores were generated and refitted under the new calibration."""
    grouped={};errors=[];n_signal=n_extension=0
    plan=json.loads((ROOT/'signal_plan.json').read_text())
    expected={(m['dataset'],m['target'],m['factor'],m['chunk'],m['n']) for m in plan};seen=set()
    extension_path=ROOT/'extension_plan.json'
    extension_plan=json.loads(extension_path.read_text()) if extension_path.exists() else []
    if extension_plan:
        extension_lock=json.loads(extension_path.with_suffix('.lock.json').read_text())
        assert extension_lock['plan_sha256']==hashlib.sha256(extension_path.read_bytes()).hexdigest()
        assert extension_lock['manifest_sha256']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    expected_extension={(m['dataset'],m['target'],m['factor'],m['chunk'],m['n']) for m in extension_plan};seen_extension=set()
    assert len(expected_extension)==len(extension_plan)
    for path,m,r in raw_batches(ROOT,('signal','extend')):
        assert m['target'] in ISOTOPES and m['targets']==[m['target']]
        if m['stage']=='signal':
            batch=(m['dataset'],m['target'],m['factor'],m['chunk'],m['n'])
            assert batch in expected and batch not in seen;seen.add(batch)
            n_signal+=m['n']
        else:
            batch=(m['dataset'],m['target'],m['factor'],m['chunk'],m['n'])
            assert batch in expected_extension and batch not in seen_extension;seen_extension.add(batch)
            n_extension+=m['n']
        key=(m['dataset'],m['target'],m['factor'])
        g=grouped.setdefault(key,dict(dataset=m['dataset'],target=m['target'],factor=m['factor'],n=0,
            ratios=[],sources=[],n_old=0,n_new=0,n_signal=0,n_extension=0))
        g['ratios'].append(r[:,:,0]);g['n']+=m['n'];g['n_new']+=m['n']
        g['n_signal' if m['stage']=='signal' else 'n_extension']+=m['n']
        g['sources'].append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            npz_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),seed=m['seed']))
        errors.extend(dict(error=e,source=str(path)) for e in m['errors'])
    assert seen==expected,'Formal signal plan incomplete'
    assert seen_extension==expected_extension,'Extension plan incomplete'
    assert n_signal==INITIAL_SIGNAL_N
    for g in grouped.values():
        assert g['n']==SIGNAL_N_PER_POINT, 'Each independent signal point requires 20000 rows'
        g['ratios']=np.concatenate(g['ratios'])
    return [grouped[k] for k in sorted(grouped)],errors,n_signal,n_extension


def validation_data():
    if (ROOT/'portable_validation.npz').exists() and not (ROOT/'points').exists():
        with np.load(ROOT/'portable_validation.npz') as a:
            return {d:a[d] for d in DATES},json.loads((ROOT/'portable_validation_index.json').read_text())['errors']
    plan=json.loads((ROOT/'validation_plan.json').read_text())
    expected={(v['dataset'],v['chunk'],v['n']) for v in plan};seen=set();ratios={d:[] for d in DATES};errors=[]
    for path,m,r in raw_batches(ROOT,('validation',)):
        key=(m['dataset'],m['chunk'],m['n']);assert key in expected and key not in seen;seen.add(key)
        assert m['target']==0 and m['targets']==[1,2,3,4]
        ratios[m['dataset']].append(r)
        errors.extend(dict(error=e,source=str(path)) for e in m['errors'])
    assert seen==expected,'Independent validation series incomplete'
    ratios={d:np.concatenate(v) for d,v in ratios.items()}
    assert all(v.shape==(VALIDATION_N,2,4) for v in ratios.values())
    return ratios,errors


def crossing(x,y,power=.95):
    if not len(x) or not np.all(np.isfinite(y)):return None
    # The first grid point already above the target means a left-censored
    # first crossing, even if later Monte Carlo noise produces a downward dip.
    if y[0]>power:return None
    if y[0]==power:return float(x[0])
    for a,b,pa,pb in zip(x[:-1],x[1:],y[:-1],y[1:]):
        if pa<=power<=pb and pb>pa:return float(a+(power-pa)/(pb-pa)*(b-a))
    return None


def bootstrap_power(ratios,cutoffs,rng):
    """Paired row-bootstrap via joint multinomial outcomes at each cutoff.

    With varying cutoffs the classifications differ in each bootstrap replicate.
    The conditional multinomial draw is exactly the distribution of resampling
    the original paired rows after classification at that replicate's c_b.
    """
    n=len(ratios);cutoffs=np.asarray(cutoffs)
    if cutoffs.ndim==1:cutoffs=np.broadcast_to(cutoffs,(B,2))
    assert cutoffs.shape==(B,2)
    valid=np.isfinite(ratios)
    w=ratios[None,:,0]>cutoffs[:,None,0]
    p=ratios[None,:,1]>cutoffs[:,None,1]
    if np.all(valid):
        nw=w.sum(axis=1);np_=p.sum(axis=1);both=(w&p).sum(axis=1)
        counts=np.stack([n-nw-np_+both,np_-both,nw-both,both],axis=1)
        draws=rng.multinomial(n,counts/n)
        return np.stack([(draws[:,2]+draws[:,3])/n,(draws[:,1]+draws[:,3])/n],axis=1)
    states=np.array([(a,b) for a in (-1,0,1) for b in (-1,0,1)])
    wd=np.where(valid[None,:,0],w.astype(int),-1);pd=np.where(valid[None,:,1],p.astype(int),-1)
    counts=np.stack([((wd==a)&(pd==b)).sum(axis=1) for a,b in states],axis=1)
    draws=rng.multinomial(n,counts/n);values=[]
    for j in range(2):
        denom=draws[:,states[:,j]>=0].sum(axis=1)
        values.append(np.divide(draws[:,states[:,j]==1].sum(axis=1),denom,
            out=np.full(B,np.nan),where=denom>0))
    return np.stack(values,axis=1)


def bootstrap_roots(x,y):
    values=[];ambiguous=0
    for p in y:
        value=crossing(x,p)
        if value is None:
            if np.all(np.isfinite(p)) and p[0]>.95:value=-np.inf
            elif np.all(p<.95):value=np.inf
            else:value=np.nan;ambiguous+=1
        values.append(value)
    a=np.array(values)
    if ambiguous:ci=[None,None]
    else:
        q=np.quantile(a,[.025,.975],method='inverted_cdf')
        ci=[float(v) if np.isfinite(v) else None for v in q]
    return a,dict(n_repetitions=B,n_crossings=int(np.isfinite(a).sum()),n_below_grid=int(np.isneginf(a).sum()),
        n_above_grid=int(np.isposinf(a).sum()),n_ambiguous=ambiguous,ci95_factor=ci)


def conservative_difference_interval(a,b,nominal):
    """Interval-censored root differences; don't drop out-of-grid replicates."""
    finite=np.isfinite(a)&np.isfinite(b)
    lower=np.full(B,-np.inf);upper=np.full(B,np.inf)
    lower[finite]=(a[finite]-b[finite])*nominal;upper[finite]=lower[finite]
    ql=np.quantile(lower,.025,method='inverted_cdf');qu=np.quantile(upper,.975,method='inverted_cdf')
    return [float(ql) if np.isfinite(ql) else None,float(qu) if np.isfinite(qu) else None],int(finite.sum())


def grid_bounds(points,null_ratio,c_low,c_high,nominal,total=False):
    confidence=1-.025/(2*9) if total else 1-.05/9
    checks=[(g['factor'],g['ratios']) for g in points]
    if null_ratio is not None:checks=[(0.,null_ratio)]+checks
    lo=0.;hi=None
    for factor,r in checks:
        liberal=cp_rate(r,c_low,confidence)['conservative_ci']
        strict=cp_rate(r,c_high,confidence)['conservative_ci']
        if liberal[1]<.95:lo=max(lo,factor)
        if strict[0]>.95:hi=factor if hi is None else min(hi,factor)
    return [lo*nominal,None if hi is None else hi*nominal]


def covariance_diagnostics(preview=False):
    """Unavailable regular free-fit covariance does not mean Lc is missing."""
    grouped={}
    for path in sorted((ROOT/'points').glob('*.json')):
        if preview and '_validation_' in path.stem:continue
        m=json.loads(path.read_text())
        if m['stage'] not in ('calibration','validation','signal','extend'):continue
        key=(m['dataset'],m['stage'])
        row=grouped.setdefault(key,dict(dataset=m['dataset'],stage=m['stage'],n=0,wls_unavailable=0,poisson_unavailable=0))
        row['n']+=m['n']
        with np.load(path.with_suffix('.npz')) as data:
            for family in FAMILIES:row[family+'_unavailable']+=int(np.sum(~data[family+'_free_covariance_available']))
    return [grouped[key] for key in sorted(grouped)]


def evaluate(preview=False):
    manifest=json.loads((ROOT/'manifest.json').read_text());cal=json.loads((ROOT/'calibration.json').read_text())
    assert tuple(manifest['datasets'])==DATES
    assert manifest['calibration_n_per_dataset']==CALIBRATION_N
    assert manifest['validation_n_per_dataset']==VALIDATION_N
    for name,digest in json.loads((ROOT/'run_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    frozen=json.loads((ROOT/'cutoff.json').read_text())
    # Refuse any alteration of coefficients or their bootstrap after freezing.
    assert frozen['frozen'] and frozen['validation_not_used'] and cal['validation_not_used']
    for name,key in [('calibration.json','calibration_sha256'),('calibration_bootstrap.npz','bootstrap_sha256')]:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen[key]
    assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()==cal['source_sha256']['manifest.json']
    boot=np.load(ROOT/'calibration_bootstrap.npz')
    groups,errors,n_signal,n_extension=signal_groups()
    val=None
    if not preview:
        val,validation_errors=validation_data();errors+=validation_errors
    curves=[];need=[];bootstrap_arrays={}
    for ds_index,dataset in enumerate(DATES):
        for target,isotope in ISOTOPES.items():
            pts=sorted([g for g in groups if g['dataset']==dataset and g['target']==target],key=lambda g:g['factor'])
            assert 4<=len(pts)<=8 and all(g['n']==SIGNAL_N_PER_POINT for g in pts)
            nominal=manifest['datasets'][dataset]['nominal_MDC'][target-1]*1000
            methods_cal=[cal['datasets'][dataset]['methods'][f][target-1] for f in FAMILIES]
            c=np.array([v['c'] for v in methods_cal]);x=np.array([g['factor'] for g in pts])
            cb=boot[dataset][:,:,target-1];assert cb.shape==(B,2)
            roots_by_mode={};meta_by_mode={}
            for mode,cutoffs,code in [('standard',np.ones(2),3),('conditional',c,1),('total',cb,2)]:
                rng=np.random.default_rng([BOOTSTRAP_MASTER,ds_index,target,code])
                powers=np.stack([bootstrap_power(g['ratios'],cutoffs,rng) for g in pts],axis=1)
                roots_by_mode[mode]=[];meta_by_mode[mode]=[]
                for j in range(2):
                    roots,metadata=bootstrap_roots(x,powers[:,:,j]);roots_by_mode[mode].append(roots);meta_by_mode[mode].append(metadata)
                    bootstrap_arrays[f'{dataset}_i{target}_{FAMILIES[j]}_{mode}']=roots
            methods={}
            for j,f in enumerate(FAMILIES):
                calibration=methods_cal[j]
                formal=[dict(factor=g['factor'],activity_mBq_m3=g['factor']*nominal,
                    standard=cp_rate(g['ratios'][:,j],1),calibrated=cp_rate(g['ratios'][:,j],c[j])) for g in pts]
                standard=crossing(x,[p['standard']['rate'] for p in formal])
                adjusted=crossing(x,[p['calibrated']['rate'] for p in formal])
                validation=None if val is None else dict(standard=cp_rate(val[dataset][:,j,target-1],1),calibrated=cp_rate(val[dataset][:,j,target-1],c[j]))
                m=dict(calibration=calibration,validation=validation,formal_points=formal,
                    standard_ld_mBq_m3=None if standard is None else standard*nominal,
                    calibrated_ld_mBq_m3=None if adjusted is None else adjusted*nominal,
                    bootstrap={mode:meta_by_mode[mode][j] for mode in roots_by_mode},
                    empirical_reversals=int(np.sum(np.diff([p['calibrated']['rate'] for p in formal])<0)))
                for mode in roots_by_mode:
                    m[mode+'_ci95_mBq_m3']=[None if v is None else v*nominal for v in meta_by_mode[mode][j]['ci95_factor']]
                nuisance_null=None if val is None else val[dataset][:,j,target-1]
                single=[dict(factor=g['factor'],ratios=g['ratios'][:,j]) for g in pts]
                m['standard_grid_bound95_mBq_m3']=grid_bounds(single,nuisance_null,1,1,nominal)
                m['conditional_grid_bound95_mBq_m3']=grid_bounds(single,nuisance_null,c[j],c[j],nominal)
                m['total_grid_bound95_mBq_m3']=grid_bounds(single,nuisance_null,*calibration['ci975_c'],nominal,total=True)
                if adjusted is None or standard is None or any(v is None for mode in ('standard','conditional','total') for v in m[mode+'_ci95_mBq_m3']):
                    direction=[]
                    ys=[p['calibrated']['rate'] for p in formal]
                    if adjusted is None:direction.append('lower' if ys[0]>.95 else 'upper')
                    if standard is None:direction.append('lower' if formal[0]['standard']['rate']>.95 else 'upper')
                    for mode in ('standard','conditional','total'):
                        ci=m[mode+'_ci95_mBq_m3']
                        if ci[0] is None:direction.append('lower')
                        if ci[1] is None:direction.append('upper')
                    need.append(dict(dataset=dataset,target=target,family=f,directions=sorted(set(direction)),
                        min_factor=float(x[0]),max_factor=float(x[-1]),reason='Point estimate or bootstrap interval not covered by the independent signal grid'))
                methods[f]=m
            paired={}
            if all(methods[f]['calibrated_ld_mBq_m3'] is not None for f in FAMILIES):
                paired['point']=methods['wls']['calibrated_ld_mBq_m3']-methods['poisson']['calibrated_ld_mBq_m3']
                for mode in ('conditional','total'):
                    ci,finite=conservative_difference_interval(*roots_by_mode[mode],nominal)
                    paired[mode+'_ci95']=ci;paired[mode+'_n_finite']=finite
            curves.append(dict(dataset=dataset,target=target,isotope=isotope,
                nominal_LC_mBq_m3=manifest['datasets'][dataset]['nominal_LC'][target-1]*1000,
                nominal_MDC_mBq_m3=nominal,methods=methods,paired_ld_difference=paired))
    bootstrap_name='power_preview_bootstrap.npz' if preview else 'analysis_bootstrap.npz'
    np.savez_compressed(ROOT/bootstrap_name,**bootstrap_arrays)
    summary=dict(scope=manifest['scope'],unit='mBq/m3',n_new_by_stage=dict(calibration=CALIBRATION_N,validation=0 if preview else VALIDATION_N,signal=n_signal,signal_extension=n_extension),
        n_reused_signal=0,technical_vectors_excluded=manifest['technical_vectors_excluded'],curves=curves,errors=errors,needs_extension=need,
        manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
        cutoff_sha256=hashlib.sha256((ROOT/'cutoff.json').read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_file=bootstrap_name,bootstrap_sha256=hashlib.sha256((ROOT/bootstrap_name).read_bytes()).hexdigest(),
        covariance_diagnostics=covariance_diagnostics(preview),
        note='Conditional CI: fixed fitted cutoff; total CI: uncertainty in ideal population-quantile cutoff and signal MC; bootstrap conditions on chosen grid and omits interpolation-shape/systematic errors')
    write_json(ROOT/('power_preview.json' if preview else 'summary.json'),summary)
    print('Signal vectors',n_signal,'new baseline +',n_extension,'new extension; errors',len(errors),'extension requests',len(need))
    for c in curves:print(c['dataset'],c['isotope'],{f:dict(c=c['methods'][f]['calibration']['c'],Ld=c['methods'][f]['calibrated_ld_mBq_m3']) for f in FAMILIES})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    evaluate(preview=args.preview)
