"""Independent audit of observation-based MDC histograms in the 100k study.

All vectors and compact MDC values are verified. Null selected fits/QR roots
are covered by the mandatory main audit; new histogram selected rows are
refitted and checked by QR + brentq here. No production detection or analysis
helper is imported.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import argparse,hashlib,json
import numpy as np
from scipy.stats import norm
import verify_main as core
from verify_main import cp_interval,independent_limits
ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}
Z95=float(norm.ppf(.95))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def stem(job):
    value=f"{job['dataset']}_{job['stage']}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job['chunk']}"
    return value+'_'+job['family'] if job['stage']=='hist_signal' else value

def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_array(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def near(actual, expected, label, rtol=2e-12, atol=1e-14):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                               equal_nan=False, err_msg=label)


def same_tree(actual, expected, label):
    """Tight numeric comparison, exact keys/types/counts and categorical fields."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ': keys')
        for key, value in expected.items():
            same_tree(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for index, value in enumerate(expected):
            same_tree(actual[index], value, f'{label}[{index}]')
    elif isinstance(expected, float):
        near(actual, expected, label, rtol=2e-11, atol=2e-12)
    else:
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def distribution_summary(values):
    """Independent sample summaries; quantiles use explicit linear interpolation."""
    finite = np.sort(np.asarray(values)[np.isfinite(values)])
    n = len(finite)
    require(n >= 2, 'insufficient valid MDC values')
    mean = float(np.sum(finite) / n)
    sd = float(np.sqrt(np.sum((finite - mean)**2) / (n - 1)))

    def quantile(probability):
        position = (n - 1) * probability
        lower = int(np.floor(position))
        upper = min(lower + 1, n - 1)
        return float(finite[lower] + (position - lower) * (finite[upper] - finite[lower]))

    return dict(n_attempted=len(values), n_valid=n, n_missing=len(values)-n,
                mean=mean, median=quantile(.5), sd=sd, mean_mcse=sd / np.sqrt(n),
                q025=quantile(.025), q25=quantile(.25), q75=quantile(.75),
                q975=quantile(.975), min=float(finite[0]), max=float(finite[-1]))


def decision_summary(decisions):
    total = len(decisions)
    valid = decisions[decisions >= 0]
    n, k = len(valid), int(np.count_nonzero(valid == 1))
    if n == 0:
        return dict(n_attempted=total, n_valid=0, n_missing=total, rate=None, k=0, ci95=None)
    return dict(n_attempted=total, n_valid=n, n_missing=total-n, k=k, rate=k/n,
                ci95=cp_interval(k, n), unconditional_rate_bounds=[k/total, (k+total-n)/total],
                conservative_ci95=[cp_interval(k, total)[0], cp_interval(k+total-n, total)[1]])


def inputs():
    manifest,model,_=core.verify_inputs()
    main_audit=read(ROOT/'verification.json')
    require(main_audit['status']=='passed' and main_audit['audit_scope']=='main','complete main audit')
    for key,name in [('manifest_sha256','manifest.json'),('run_lock_sha256','run_lock.json'),
                     ('summary_sha256','summary.json'),('audit_file_sha256','verify_main.py')]:
        require(main_audit[key]==sha(ROOT/name),'main audit source '+name)
    plan=read(ROOT/'histogram_plan.json');lock=read(ROOT/'histogram_plan.lock.json')
    for key,name in [('plan_sha256','histogram_plan.json'),('manifest_sha256','manifest.json'),
                     ('summary_sha256','summary.json'),('cutoff_sha256','cutoff.json')]:
        require(lock[key]==sha(ROOT/name),'histogram freeze '+name)
    power=read(ROOT/'summary.json');cal=read(ROOT/'calibration.json')
    require(len(plan)==160 and sum(j['n'] for j in plan)==160000,'histogram signal count')
    seen=set();seeds=set()
    for job in plan:
        require(job['stage']=='hist_signal' and job['dataset']=='may22' and job['n']==1000,'histogram plan stage')
        target,family,chunk=job['target'],job['family'],job['chunk']
        require(target in ISOTOPES and family in FAMILIES and 0<=chunk<20,'histogram tuple')
        require((target,family,chunk) not in seen,'duplicate histogram job');seen.add((target,family,chunk))
        curve=next(c for c in power['curves'] if c['target']==target)
        value=curve['methods'][family]['calibrated_ld_mBq_m3']
        require(value is not None and np.isfinite(value) and value>0,'finite generating Ld')
        require(job['activity_Bq_m3']==value/1000,'fixed exact generating Ld')
        code=round(10000*job['factor']);require(abs(job['factor']-code/10000)<1e-12,'seed factor exactness')
        seed=(manifest['master_seed'],0,target,code,15,chunk,FAMILIES.index(family))
        require(seed not in seeds,'duplicate histogram seed');seeds.add(seed)
    require(seen=={(i,f,c) for i in ISOTOPES for f in FAMILIES for c in range(20)},'complete histogram combinations')
    null_plan=read(ROOT/'validation_plan.json')
    require(sum(j['n'] for j in null_plan)==100000 and len(null_plan)==100,'null plan count')
    return manifest,model,power,cal,main_audit,null_plan,plan


def audit_batches(manifest,model,power,cal,main_audit,null_plan,signal_plan):
    arrays=defaultdict(list);decisions=defaultdict(list);errors=[];covariance={};sources={}
    counts={'null':0,'signal':0};n_refits=n_roots=0;maximum_relative=0.;all_seeds=set()
    S=model.physical_matrix
    for job in null_plan+signal_plan:
        name=stem(job);path=ROOT/'points'/(name+'.json');meta=read(path)
        require(all(meta[k]==v for k,v in job.items()),'job metadata '+name)
        require(meta['manifest_sha256']==sha(ROOT/'manifest.json'),'manifest hash '+name)
        require(meta['npz_sha256']==sha(path.with_suffix('.npz')),'NPZ hash '+name)
        require(meta['mode']=='iterative' and meta['alpha']==meta['beta_ii']==.05,'ordinary iterative MDC '+name)
        require(meta['raw_activity_unit']=='Bq/m3','raw unit '+name)
        scenario='null' if meta['stage']=='validation' else 'signal'
        require(meta['stage'] in ('validation','hist_signal'),'stage '+name)
        family=meta['family'];target=meta['target'];n=meta['n'];counts[scenario]+=n
        seed=[manifest['master_seed'],0,target,round(10000*meta['factor']),manifest['stage_codes'][meta['stage']],meta['chunk']]
        if scenario=='signal':seed.append(FAMILIES.index(family))
        require(meta['seed']==seed and tuple(seed) not in all_seeds,'unique expected seed '+name);all_seeds.add(tuple(seed))
        truth=np.asarray(manifest['datasets']['may22']['truth0']).copy()
        if scenario=='signal':truth[target]=job['activity_Bq_m3']
        else:
            expected=main_audit['source_batches'][name]
            require(expected['json']==sha(path) and expected['npz']==sha(path.with_suffix('.npz')),'unchanged audited null data '+name)
        exact_array(meta['truth'],truth,'truth '+name)
        mu=S@truth+model.b;exact_array(meta['mu'],mu,'expected observations '+name)
        expected_targets=[1,2,3,4] if scenario=='null' else [target]
        expected_families=list(FAMILIES) if scenario=='null' else [family]
        require(meta['targets']==expected_targets and meta['families']==expected_families,'targets/families '+name)
        sources[name]={'json':sha(path),'npz':meta['npz_sha256']}
        errors.extend(dict(source=path.name,**e) for e in meta['errors'])
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as loaded:
            a={key:loaded[key] for key in loaded.files}
        exact_array(a['x'],np.random.default_rng(seed).poisson(mu,size=(n,21)),'Poisson x '+name)
        exact_array(a['indices'],np.arange(n),'indices '+name);exact_array(a['targets'],expected_targets,'targets '+name)
        for f in expected_families:
            cv=covariance.setdefault((scenario,f),dict(scenario=scenario,family=f,n=0,unavailable=0))
            cv['n']+=n;cv['unavailable']+=int((~a[f+'_free_covariance_available']).sum())
            for kind in ('free','positive'):
                p=a[f+'_'+kind];success=a[f+'_'+kind+'_success']
                require(p.shape==(n,17) and success.shape==(n,),'fit shape '+name)
                require(np.all(np.isfinite(p[success])) and np.all(np.isnan(p[~success])),'fit availability '+name)
                if kind=='positive':require(np.all(p[success]>=0),'physical nonnegativity '+name)
                if f=='poisson' and np.any(success):
                    means=p[success]@S.T+model.b
                    roundoff=256*np.finfo(float).eps*(np.abs(p[success])@np.abs(S).T+model.b+1)
                    require(np.all(means>=-roundoff) and np.all(means[a['x'][success]>0]>0),'Poisson mean feasibility '+name)
            lc=a[f+'_LC'];mdc=a[f+'_MDC'];conv=a[f+'_converged'];sigma0=a[f+'_sigma0'];sigma_ld=a[f+'_sigma_at_MDC']
            require(lc.shape==mdc.shape==conv.shape==(n,len(expected_targets)),'MDC shapes '+name)
            require(np.all(np.isnan(mdc[~conv])),'retain invalid MDC as NaN '+name)
            require(np.all(np.isfinite(mdc[conv])) and np.all(mdc[conv]>0),'valid MDC '+name)
            near(mdc[conv],(lc+Z95*sigma_ld)[conv],'ordinary MDC equation '+name,rtol=3e-8)
            valid_sigma=np.isfinite(sigma0);near(lc[valid_sigma],Z95*sigma0[valid_sigma],'LC from sigma0 '+name)
            for col,i in enumerate(expected_targets):
                key=f'may22_{f}_i{i}_{scenario}'
                arrays[key].append(mdc[:,col]*1000)
                coefficient=cal['datasets']['may22']['methods'][f][i-1]['c']
                free=a[f+'_free'][:,i];threshold=lc[:,col]
                valid=np.isfinite(free)&np.isfinite(threshold)&(threshold>0)
                standard=np.full(n,-1,dtype=np.int8);calibrated=standard.copy()
                standard[valid]=(free[valid]>threshold[valid]);calibrated[valid]=(free[valid]>coefficient*threshold[valid])
                exact_array(standard,a[f+'_decision'][:,col],'standard decisions '+name)
                decisions[(key,'standard')].append(standard);decisions[(key,'calibrated')].append(calibrated)
        if scenario=='signal':
            for row in (0,n-1):
                trial=core.JointModel(model.M,a['x'][row],model.b,model.n_sample_parameters,
                                      model.T_c,model.coordinates,model.n_detector_parameters)
                positive_ok=True
                for kind,method in [('free',family),('positive',core.PILOTS[family])]:
                    result=core.fit(trial,method);stored_ok=bool(a[family+'_'+kind+'_success'][row])
                    result_ok=bool(result.success and result.physical_parameters is not None
                                   and np.all(np.isfinite(result.physical_parameters)))
                    require(result_ok==stored_ok,'refit convergence '+name)
                    if stored_ok:near(result.physical_parameters,a[family+'_'+kind][row],'refit parameters '+name,rtol=1e-8,atol=1e-10)
                    else:positive_ok=False
                    if kind=='free' and stored_ok:
                        require((result.physical_covariance is not None)==bool(a[family+'_free_covariance_available'][row]),'covariance availability '+name)
                    n_refits+=1
                if positive_ok and a[family+'_converged'][row,0]:
                    limits=independent_limits(S,model.b,a[family+'_positive'][row],target)
                    for field,value in limits.items():near(a[family+'_'+field][row,0],value,'independent '+field+' '+name,rtol=3e-8)
                    maximum_relative=max(maximum_relative,abs(a[family+'_MDC'][row,0]/limits['MDC']-1));n_roots+=1
        print('HIST AUDIT',name,flush=True)
    require(counts=={'null':100000,'signal':160000},'histogram vector totals')
    return ({key:np.concatenate(parts) for key,parts in arrays.items()},
            {key:np.concatenate(parts) for key,parts in decisions.items()},errors,list(covariance.values()),sources,
            dict(poisson_vectors_checked=260000,reused_null_vectors_checked=100000,new_signal_vectors_checked=160000,
                 formal_batches=260,unique_seeds_checked=len(all_seeds),new_signal_refitted_models=n_refits,
                 new_signal_independent_MDC_roots=n_roots,max_relative_MDC_root_difference=maximum_relative,
                 null_fits_and_selected_roots_checked_by='verification.json'))


def audit_summary(manifest,power,cal,arrays,decisions,errors,covariance,sources):
    summary=read(ROOT/'histogram_summary.json')
    for key,name in [('manifest_sha256','manifest.json'),('plan_sha256','histogram_plan.json'),
                     ('null_plan_sha256','validation_plan.json'),('power_summary_sha256','summary.json'),
                     ('values_sha256','histogram_values.npz')]:
        require(summary[key]==sha(ROOT/name),'summary hash '+name)
    require(summary['source_batches']==sources,'summary source batches')
    exact_keys={f'may22_{f}_i{i}_{s}' for f in FAMILIES for i in ISOTOPES for s in ('null','signal')}
    require(set(arrays)==exact_keys,'compact keys')
    with np.load(ROOT/'histogram_values.npz',allow_pickle=False) as compact:
        require(set(compact.files)==exact_keys,'saved compact keys')
        for key,values in arrays.items():
            exact_array(compact[key],values,'all MDC attempts '+key)
            require(values.shape==((100000,) if key.endswith('_null') else (20000,)),'per-scenario attempt count')
    require(summary['errors']==errors,'all histogram errors')
    require(summary['covariance_diagnostics']==covariance,'covariance diagnostics')
    total=sum(len(v) for v in arrays.values());valid=sum(int(np.isfinite(v).sum()) for v in arrays.values())
    require(total==960000,'MDC attempt count')
    for key,value in dict(unit='mBq/m3',mdc_mode='standard_iterative_alpha_beta_0.05_without_c',
        n_reused_null_vectors=100000,n_new_signal_vectors=160000,n_mdc_attempts=total,n_mdc_valid=valid,n_mdc_missing=total-valid,
        technical_vectors_excluded=128,signal_Ld_uncertainty_resampled=False,thresholds_retuned=False,systematics=False).items():
        require(summary[key]==value,'summary.'+key)
    curves=[]
    for family in FAMILIES:
        for target,isotope in ISOTOPES.items():
            power_curve=next(c for c in power['curves'] if c['target']==target)
            curve=dict(dataset='may22',family=family,target=target,isotope=isotope,
                nominal_MDC_mBq_m3=manifest['datasets']['may22']['nominal_MDC'][target-1]*1000,
                signal_A_mBq_m3=power_curve['methods'][family]['calibrated_ld_mBq_m3']/1000*1000,
                cutoff_c=cal['datasets']['may22']['methods'][family][target-1]['c'])
            for scenario in ('null','signal'):
                key=f'may22_{family}_i{target}_{scenario}';values=arrays[key]
                result=distribution_summary(values)
                result['detection']={rule:decision_summary(decisions[(key,rule)]) for rule in ('standard','calibrated')}
                finite=values[np.isfinite(values)]
                result['fraction_below_nominal_formula']=float(np.count_nonzero(finite<curve['nominal_MDC_mBq_m3'])/len(finite))
                result['fraction_below_signal_Ld95']=float(np.count_nonzero(finite<curve['signal_A_mBq_m3'])/len(finite))
                curve[scenario]=result
            curves.append(curve)
    same_tree(summary['curves'],curves,'independent distribution summaries')
    return dict(compact_arrays_checked=16,mdc_attempts_checked=total,valid_MDC=valid,unavailable_MDC=total-valid,
                error_records=len(errors),covariance_diagnostics=covariance)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    manifest,model,power,cal,main_audit,null_plan,signal_plan=inputs()
    arrays,decisions,errors,cov,sources,batch_report=audit_batches(manifest,model,power,cal,main_audit,null_plan,signal_plan)
    summary_report=audit_summary(manifest,power,cal,arrays,decisions,errors,cov,sources)
    result=dict(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(ROOT/'manifest.json'),run_lock_sha256=sha(ROOT/'run_lock.json'),
        summary_sha256=sha(ROOT/'summary.json'),histogram_summary_sha256=sha(ROOT/'histogram_summary.json'),
        histogram_values_sha256=sha(ROOT/'histogram_values.npz'),histogram_plan_sha256=sha(ROOT/'histogram_plan.json'),
        histogram_plan_lock_sha256=sha(ROOT/'histogram_plan.lock.json'),audit_file_sha256=sha(Path(__file__)),
        audit_helper_sha256=sha(ROOT/'verify_main.py'),main_verification_sha256=sha(ROOT/'verification.json'),
        selected_signal_refit_rows_per_batch=[0,999],independent_method='Regenerated all x, all MDC values and summaries; selected new fits and scaled QR + brentq roots',
        **batch_report,**summary_report)
    output=args.output or ROOT/'histogram_verification.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

