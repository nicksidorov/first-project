"""Distributions of per-observation iterative MDC; no new fits or threshold tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
DATES=('may22','may26')
FAMILIES=('wls','poisson')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((ROOT/name).read_text())
def key(m):return tuple(m[k] for k in ('dataset','stage','target','family','n','chunk'))


def stats(values):
    a=np.asarray(values);v=a[np.isfinite(a)];n=len(v)
    if n<2:raise RuntimeError('Insufficient valid MDC values')
    q=np.quantile(v,[.025,.25,.5,.75,.975])
    return dict(n_attempted=len(a),n_valid=n,n_missing=len(a)-n,mean=float(v.mean()),
        median=float(q[2]),sd=float(v.std(ddof=1)),mean_mcse=float(v.std(ddof=1)/np.sqrt(n)),
        q025=float(q[0]),q25=float(q[1]),q75=float(q[3]),q975=float(q[4]),
        min=float(v.min()),max=float(v.max()))


def detection_rate(a):
    a=np.asarray(a);ok=a>=0;n=int(ok.sum());k=int(a[ok].sum());N=len(a)
    if not n:return dict(n_attempted=N,n_valid=0,n_missing=N,rate=None,k=0,ci95=None)
    ci=binomtest(k,n).proportion_ci(.95,method='exact')
    lo=binomtest(k,N).proportion_ci(.95,method='exact').low
    hi=binomtest(k+N-n,N).proportion_ci(.95,method='exact').high
    return dict(n_attempted=N,n_valid=n,n_missing=N-n,k=k,rate=k/n,ci95=[ci.low,ci.high],
                unconditional_rate_bounds=[k/N,(k+N-n)/N],conservative_ci95=[lo,hi])


def main():
    manifest=read('manifest.json');plan=read('plan.json');expected={key(j) for j in plan}
    assert len(expected)==len(plan)==68
    for name,digest in read('run_lock.json').items():assert sha(ROOT/name)==digest,name
    grouped={};decisions={};seen=set();errors=[];cov={};counts={'null':0,'signal_ld':0};sources={}
    for path in sorted((ROOT/'points').glob('*.json')):
        meta=json.loads(path.read_text());k=key(meta)
        assert k in expected and k not in seen,k
        seen.add(k);counts[meta['stage']]+=meta['n']
        assert meta['manifest_sha256']==sha(ROOT/'manifest.json')
        assert meta['npz_sha256']==sha(path.with_suffix('.npz'))
        sources[path.stem]={'json':sha(path),'npz':meta['npz_sha256']}
        errors.extend(dict(source=path.name,**e) for e in meta['errors'])
        scenario='null' if meta['stage']=='null' else 'signal'
        with np.load(path.with_suffix('.npz')) as a:
            for family in meta['families']:
                ck=(scenario,family)
                r=cov.setdefault(ck,dict(scenario=scenario,family=family,n=0,unavailable=0))
                r['n']+=meta['n'];r['unavailable']+=int((~a[family+'_free_covariance_available']).sum())
                for col,target in enumerate(meta['targets']):
                    name=f"{meta['dataset']}_{family}_i{target}_{scenario}"
                    raw=a[family+'_MDC'][:,col];conv=a[family+'_converged'][:,col]
                    assert np.all(np.isnan(raw[~conv]))
                    assert np.all(np.isfinite(raw[conv])&(raw[conv]>0))
                    grouped.setdefault(name,[]).append(raw*1000)
                    for rule in ('standard','calibrated'):
                        values=a[family+'_decision_'+rule][:,col]
                        assert np.all(np.isin(values,[-1,0,1]))
                        decisions.setdefault((name,rule),[]).append(values)
    assert seen==expected,'Incomplete histogram plan'
    assert counts=={'null':20000,'signal_ld':48000},counts
    arrays={k:np.concatenate(v) for k,v in grouped.items()};curves=[]
    for dataset in DATES:
        for family in FAMILIES:
            for target in range(1,5):
                setting=manifest['datasets'][dataset]['settings'][str(target)]
                c=dict(dataset=dataset,family=family,target=target,isotope=setting['isotope'],
                    nominal_MDC_mBq_m3=setting['formula_MDC_Bq_m3']*1000,
                    signal_A_mBq_m3=setting['methods'][family]['signal_activity_Bq_m3']*1000,
                    cutoff_c=setting['methods'][family]['c'])
                for scenario,n in [('null',10000),('signal',3000)]:
                    name=f'{dataset}_{family}_i{target}_{scenario}'
                    assert len(arrays[name])==n
                    result=stats(arrays[name])
                    result['detection']={rule:detection_rate(np.concatenate(decisions[(name,rule)]))
                                         for rule in ('standard','calibrated')}
                    v=arrays[name][np.isfinite(arrays[name])]
                    result['fraction_below_nominal_formula']=float(np.mean(v<c['nominal_MDC_mBq_m3']))
                    result['fraction_below_signal_Ld95']=float(np.mean(v<c['signal_A_mBq_m3']))
                    c[scenario]=result
                curves.append(c)
    np.savez_compressed(ROOT/'values.npz',**arrays)
    summary=dict(unit='mBq/m3',mdc_mode='standard_iterative_alpha_beta_0.05_without_c',
        signal_definition='Fixed previously calibrated Ld95 for each date, isotope and method',
        n_reused_null_vectors=counts['null'],n_new_signal_vectors=counts['signal_ld'],
        n_mdc_attempts=sum(len(v) for v in arrays.values()),
        n_mdc_valid=sum(int(np.isfinite(v).sum()) for v in arrays.values()),
        n_mdc_missing=sum(int((~np.isfinite(v)).sum()) for v in arrays.values()),
        technical_vectors_excluded=manifest['technical_vectors_excluded'],
        curves=curves,errors=errors,covariance_diagnostics=list(cov.values()),
        manifest_sha256=sha(ROOT/'manifest.json'),plan_sha256=sha(ROOT/'plan.json'),
        values_sha256=sha(ROOT/'values.npz'),source_batches=sources,
        interpretation='q025/q975 describe the central 95% of per-trial MDC estimates, not a confidence interval for true Ld95; no conditioning on detection',
        signal_Ld_uncertainty_resampled=False,thresholds_retuned=False,systematics=False)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Vectors:',counts,'MDC attempts:',summary['n_mdc_attempts'],'missing:',summary['n_mdc_missing'],'errors:',len(errors))
    for c in curves:print(c['dataset'],c['family'],c['isotope'],'formula',c['nominal_MDC_mBq_m3'],
        'Ld95',c['signal_A_mBq_m3'],'median0',c['null']['median'],'medianLd',c['signal']['median'])


if __name__=='__main__':main()
