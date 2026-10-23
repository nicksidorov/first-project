"""Freeze permitted adaptive power extensions or exact-Ld histogram jobs.

This module does not generate observations. Each newly requested set of jobs
is saved with source hashes before the runner is invoked. Frozen c is never
changed. Original observations are never replaced.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def base_lock(root,summary_name):
    manifest=read(root/'manifest.json')
    for name,digest in read(root/'run_lock.json').items():
        assert sha(root/name)==digest,name
    cutoff=read(root/'cutoff.json')
    assert cutoff['frozen'] and cutoff['validation_not_used']
    assert cutoff['calibration_sha256']==sha(root/'calibration.json')
    assert cutoff['bootstrap_sha256']==sha(root/'calibration_bootstrap.npz')
    summary=read(root/summary_name)
    assert summary['cutoff_sha256']==sha(root/'cutoff.json')
    return manifest,summary,dict(manifest_sha256=sha(root/'manifest.json'),
        summary_file=summary_name,summary_sha256=sha(root/summary_name),
        cutoff_sha256=sha(root/'cutoff.json'),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        planner_sha256=sha(Path(__file__)))


def extensions(root=ROOT,summary_name='summary.json'):
    root=Path(root).resolve()
    manifest,summary,lock=base_lock(root,summary_name)
    requests=summary['needs_extension']
    if not requests:
        print('No extension is required.')
        return None
    original=read(root/'signal_plan.json')
    cumulative=root/'extension_plan.json'
    prior=read(cumulative) if cumulative.exists() else []
    if prior:
        old_lock=read(cumulative.with_suffix('.lock.json'))
        assert old_lock['plan_sha256']==sha(cumulative)
        assert old_lock['manifest_sha256']==lock['manifest_sha256']
    new=[];reasons=[];exhausted=[]
    for target in range(1,5):
        relevant=[r for r in requests if r['target']==target]
        if not relevant:continue
        factors={j['factor'] for j in original+prior if j['target']==target}
        prior_factors={j['factor'] for j in prior if j['target']==target}
        budget=4-len(prior_factors)
        directions=sorted({d for r in relevant for d in r['directions']})
        for direction in directions:
            assert direction in ('lower','upper')
            if budget<=0:
                exhausted.append(dict(target=target,direction=direction))
                continue
            factor=round(min(factors)-.04 if direction=='lower' else max(factors)+.04,4)
            if factor<=0:
                exhausted.append(dict(target=target,direction=direction,reason='nonpositive activity'))
                continue
            assert factor not in factors
            for chunk in range(20):
                new.append(dict(dataset='may22',stage='extend',target=target,
                    factor=factor,n=1000,chunk=chunk))
            reasons.append(dict(target=target,direction=direction,factor=factor,requests=relevant))
            factors.add(factor);budget-=1
    if not new:
        print('Extension budget exhausted; retain one-sided/unavailable intervals:',exhausted)
        return None
    round_index=1
    while (root/f'extension_round_{round_index:02d}.json').exists():round_index+=1
    destination=root/f'extension_round_{round_index:02d}.json'
    write_new(destination,new)
    round_lock={**lock,'plan_sha256':sha(destination),'round':round_index,
        'reason':'Point crossing or main 95% bootstrap interval is outside the frozen grid',
        'requests':reasons,'exhausted_requests':exhausted,'vectors':sum(j['n'] for j in new)}
    write_new(destination.with_suffix('.lock.json'),round_lock)
    all_jobs=prior+new
    # The cumulative index is only for the analyzer. Immutable round files
    # retain every prospective plan and its justification.
    cumulative.write_text(json.dumps(all_jobs,ensure_ascii=False,indent=2),encoding='utf-8')
    index_lock={**lock,'plan_sha256':sha(cumulative),
        'rounds':[{ 'file':p.name,'sha256':sha(p),'lock_sha256':sha(p.with_suffix('.lock.json'))}
                  for p in sorted(root.glob('extension_round_*.json'))
                  if not p.name.endswith('.lock.json')]}
    cumulative.with_suffix('.lock.json').write_text(json.dumps(index_lock,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen',destination.name,'vectors',round_lock['vectors'])
    return destination


def histograms(root=ROOT):
    root=Path(root).resolve()
    manifest,summary,lock=base_lock(root,'summary.json')
    assert summary['n_new_by_stage']['validation']==100000
    destination=root/'histogram_plan.json'
    if destination.exists() or destination.with_suffix('.lock.json').exists():
        raise FileExistsError('Histogram generating truths have already been frozen')
    jobs=[];truths=[]
    for curve in summary['curves']:
        assert curve['dataset']=='may22'
        for family in FAMILIES:
            activity=curve['methods'][family]['calibrated_ld_mBq_m3']
            if activity is None or not math.isfinite(activity) or activity<=0:
                raise ValueError('All calibrated Ld95 point estimates must be finite and positive')
            exact=activity/1000
            truths.append(dict(target=curve['target'],isotope=curve['isotope'],family=family,
                activity_Bq_m3=exact))
            for chunk in range(20):
                jobs.append(dict(dataset='may22',stage='hist_signal',target=curve['target'],
                    factor=0.0,n=1000,chunk=chunk,family=family,activity_Bq_m3=exact))
    assert len(jobs)==160 and len(truths)==8
    write_new(destination,jobs)
    write_new(destination.with_suffix('.lock.json'),{
        **lock,'plan_sha256':sha(destination),'vectors':160000,'truths':truths,
        'thresholds_retuned':False,'signal_Ld_uncertainty_resampled':False,
        'rule':'Exact calibrated Ld95 from frozen summary / 1000; no rounding',
        'main_interval_extension_requests':summary['needs_extension']})
    print('Frozen histogram_plan.json, 160000 new vectors at exact calibrated Ld95.')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('extensions','histograms'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--summary',default='summary.json')
    args=parser.parse_args()
    if args.action=='extensions':extensions(args.root,args.summary)
    else:histograms(args.root)
