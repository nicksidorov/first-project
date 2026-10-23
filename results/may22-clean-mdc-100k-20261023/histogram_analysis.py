"""Summarize ordinary observation-based MDC at A=0 and fixed calibrated Ld95.

The validation rows are reused without selection or refitting. Signal rows
were generated after freezing Ld95. No threshold or power curve is retuned.
Raw MDC is in Bq/m3; the compact arrays and report summaries are in mBq/m3.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    a=np.asarray(values);v=a[np.isfinite(a)];n=len(v)
    if n<2:
        raise RuntimeError('Insufficient valid MDC values')
    q=np.quantile(v,[.025,.25,.5,.75,.975])
    return dict(n_attempted=len(a),n_valid=n,n_missing=len(a)-n,mean=float(v.mean()),
        median=float(q[2]),sd=float(v.std(ddof=1)),mean_mcse=float(v.std(ddof=1)/np.sqrt(n)),
        q025=float(q[0]),q25=float(q[1]),q75=float(q[3]),q975=float(q[4]),
        min=float(v.min()),max=float(v.max()))


def detection_rate(a):
    a=np.asarray(a);ok=a>=0;n=int(ok.sum());k=int(a[ok].sum());N=len(a)
    if not n:
        return dict(n_attempted=N,n_valid=0,n_missing=N,rate=None,k=0,ci95=None)
    ci=binomtest(k,n).proportion_ci(.95,method='exact')
    lo=binomtest(k,N).proportion_ci(.95,method='exact').low
    hi=binomtest(k+N-n,N).proportion_ci(.95,method='exact').high
    return dict(n_attempted=N,n_valid=n,n_missing=N-n,k=k,rate=k/n,ci95=[ci.low,ci.high],
        unconditional_rate_bounds=[k/N,(k+N-n)/N],conservative_ci95=[lo,hi])


def stem(job):
    base=f"{job['dataset']}_{job['stage']}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job['chunk']}"
    return base+'_'+job['family'] if job['stage']=='hist_signal' else base


def main(root=ROOT,histogram_plan='histogram_plan.json'):
    root=Path(root).resolve()
    manifest=read(root/'manifest.json');power=read(root/'summary.json')
    cal=read(root/'calibration.json');cutoff=read(root/'cutoff.json')
    for name,digest in read(root/'run_lock.json').items():
        assert sha(root/name)==digest,name
    assert cutoff['calibration_sha256']==sha(root/'calibration.json')
    assert power['cutoff_sha256']==sha(root/'cutoff.json')
    assert tuple(manifest['datasets'])==('may22',)
    plan_path=root/histogram_plan
    lock=read(plan_path.with_suffix('.lock.json'))
    for key,path in [('plan_sha256',plan_path),('manifest_sha256',root/'manifest.json'),
                     ('summary_sha256',root/'summary.json'),('cutoff_sha256',root/'cutoff.json')]:
        assert lock[key]==sha(path),key
    null_plan=read(root/'validation_plan.json');signal_plan=read(plan_path)
    assert sum(j['n'] for j in null_plan)==100000
    assert sum(j['n'] for j in signal_plan)==160000
    all_jobs=null_plan+signal_plan
    expected={stem(j) for j in all_jobs}
    assert len(expected)==len(all_jobs)
    grouped={};decisions={};errors=[];cov={};sources={}
    counts={'null':0,'signal':0}
    generating={}
    for job in all_jobs:
        name=stem(job);path=root/'points'/(name+'.json');meta=read(path)
        assert all(meta[k]==v for k,v in job.items()),name
        assert meta['manifest_sha256']==sha(root/'manifest.json')
        assert meta['npz_sha256']==sha(path.with_suffix('.npz'))
        assert meta['mode']=='iterative' and meta['alpha']==.05 and meta['beta_ii']==.05
        assert meta['dataset']=='may22'
        scenario='null' if meta['stage']=='validation' else 'signal'
        assert meta['stage'] in ('validation','hist_signal')
        counts[scenario]+=meta['n']
        errors.extend(dict(source=path.name,**e) for e in meta['errors'])
        sources[name]={'json':sha(path),'npz':meta['npz_sha256']}
        if scenario=='null':
            assert meta['targets']==[1,2,3,4] and meta['families']==list(FAMILIES)
            np.testing.assert_array_equal(meta['truth'],manifest['datasets']['may22']['truth0'])
        else:
            target=meta['target'];family=meta['family']
            assert meta['targets']==[target] and meta['families']==[family]
            curve=next(c for c in power['curves'] if c['target']==target)
            estimate=curve['methods'][family]['calibrated_ld_mBq_m3']
            assert estimate is not None and np.isfinite(estimate) and estimate>0
            exact=estimate/1000
            assert meta['activity_Bq_m3']==exact
            truth=np.asarray(manifest['datasets']['may22']['truth0']).copy()
            truth[target]=exact
            np.testing.assert_array_equal(meta['truth'],truth)
            generating[(family,target)]=exact*1000
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
            assert a['x'].shape==(meta['n'],21)
            np.testing.assert_array_equal(a['indices'],np.arange(meta['n']))
            np.testing.assert_array_equal(a['targets'],meta['targets'])
            for family in meta['families']:
                ck=(scenario,family)
                r=cov.setdefault(ck,dict(scenario=scenario,family=family,n=0,unavailable=0))
                r['n']+=meta['n']
                r['unavailable']+=int((~a[family+'_free_covariance_available']).sum())
                for col,target in enumerate(meta['targets']):
                    key=f'may22_{family}_i{target}_{scenario}'
                    raw=a[family+'_MDC'][:,col];conv=a[family+'_converged'][:,col]
                    assert np.all(np.isnan(raw[~conv]))
                    assert np.all(np.isfinite(raw[conv])&(raw[conv]>0))
                    grouped.setdefault(key,[]).append(raw*1000)
                    lc=a[family+'_LC'][:,col];free=a[family+'_free'][:,target]
                    valid=np.isfinite(lc)&(lc>0)&np.isfinite(free)
                    standard=np.where(valid,free>lc,-1).astype(np.int8)
                    np.testing.assert_array_equal(standard,a[family+'_decision'][:,col])
                    coefficient=cal['datasets']['may22']['methods'][family][target-1]['c']
                    calibrated=np.where(valid,free>coefficient*lc,-1).astype(np.int8)
                    for rule,values in [('standard',standard),('calibrated',calibrated)]:
                        decisions.setdefault((key,rule),[]).append(values)
    assert counts=={'null':100000,'signal':160000}
    arrays={key:np.concatenate(values) for key,values in grouped.items()}
    assert len(arrays)==16 and len(generating)==8
    curves=[]
    for family in FAMILIES:
        for target,isotope in ISOTOPES.items():
            c=dict(dataset='may22',family=family,target=target,isotope=isotope,
                nominal_MDC_mBq_m3=manifest['datasets']['may22']['nominal_MDC'][target-1]*1000,
                signal_A_mBq_m3=generating[(family,target)],
                cutoff_c=cal['datasets']['may22']['methods'][family][target-1]['c'])
            for scenario,n in [('null',100000),('signal',20000)]:
                key=f'may22_{family}_i{target}_{scenario}'
                assert len(arrays[key])==n
                result=stats(arrays[key])
                result['detection']={rule:detection_rate(np.concatenate(decisions[(key,rule)]))
                                     for rule in ('standard','calibrated')}
                v=arrays[key][np.isfinite(arrays[key])]
                result['fraction_below_nominal_formula']=float(np.mean(v<c['nominal_MDC_mBq_m3']))
                result['fraction_below_signal_Ld95']=float(np.mean(v<c['signal_A_mBq_m3']))
                c[scenario]=result
            curves.append(c)
    output=root/'histogram_values.npz'
    np.savez_compressed(output,**arrays)
    summary=dict(unit='mBq/m3',mdc_mode='standard_iterative_alpha_beta_0.05_without_c',
        signal_definition='Fixed newly calibrated Ld95 for May22, separately by isotope and method',
        n_reused_null_vectors=counts['null'],n_new_signal_vectors=counts['signal'],
        n_mdc_attempts=sum(len(v) for v in arrays.values()),
        n_mdc_valid=sum(int(np.isfinite(v).sum()) for v in arrays.values()),
        n_mdc_missing=sum(int((~np.isfinite(v)).sum()) for v in arrays.values()),
        technical_vectors_excluded=manifest['technical_vectors_excluded'],
        curves=curves,errors=errors,covariance_diagnostics=list(cov.values()),
        manifest_sha256=sha(root/'manifest.json'),plan_sha256=sha(plan_path),
        null_plan_sha256=sha(root/'validation_plan.json'),power_summary_sha256=sha(root/'summary.json'),
        values_sha256=sha(output),source_batches=sources,
        interpretation='q025/q975 are central 95% of per-trial MDC estimates, not a confidence interval for true Ld95; no conditioning on detection',
        signal_Ld_uncertainty_resampled=False,thresholds_retuned=False,systematics=False)
    (root/'histogram_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('MDC attempts:',summary['n_mdc_attempts'],'valid:',summary['n_mdc_valid'],
          'missing:',summary['n_mdc_missing'],'errors:',len(errors))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--histogram-plan',default='histogram_plan.json')
    args=parser.parse_args()
    main(args.root,args.histogram_plan)
