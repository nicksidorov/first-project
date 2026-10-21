"""Aggregation, interval arithmetic and selection of independent confirmation grid."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import binomtest
from scipy.optimize import isotonic_regression

ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')


def batches():
    """Read local complete trials or the portable compact archive."""
    paths=sorted((ROOT/'points').glob('*.json'))
    if paths:
        for path in paths:
            if not path.with_suffix('.npz').exists():continue
            with np.load(path.with_suffix('.npz')) as data:
                yield path,json.loads(path.read_text()),{k:data[k] for k in data.files}
    else:
        index=json.loads((ROOT/'trials_compact_index.json').read_text())
        with np.load(ROOT/'trials_compact.npz') as archive:
            counts=archive['x'];decisions={f:archive[f+'_decision'] for f in FAMILIES}
            for meta in index['blocks']:
                rows=slice(meta['start'],meta['stop']);cols=np.array(meta['targets'])-1
                data={'x':counts[rows]}
                for f in FAMILIES:data[f+'_decision']=decisions[f][rows][:,cols]
                yield ROOT/meta['file'],meta,data


def load_groups():
    grouped={};seeds=set();errors=[]
    for path,m,a in batches():
        seed=tuple(m['seed']);assert seed not in seeds;seeds.add(seed)
        assert a['x'].shape==(m['n'],21)
        key=(m['dataset'],m['stage'],m['target'],m['factor'])
        entry=grouped.setdefault(key,dict(dataset=m['dataset'],stage=m['stage'],target=m['target'],
            targets=m['targets'],factor=m['factor'],n=0,paths=[],chunks=[],decisions={f:[] for f in FAMILIES}))
        assert m['chunk'] not in entry['chunks'];entry['chunks'].append(m['chunk'])
        entry['n']+=m['n'];entry['paths'].append(str(path))
        for f in FAMILIES:
            decision=a[f+'_decision'];assert np.all(np.isin(decision,[-1,0,1]))
            assert decision.shape==(m['n'],len(m['targets']))
            for j in range(len(m['targets'])):
                v=decision[:,j];valid=v>=0;r=m['rates'][f][j]
                assert valid.sum()==r['n_valid'] and v[valid].sum()==r['k']
            entry['decisions'][f].append(decision.copy())
        errors.extend(dict(e,file=path.name) for e in m['errors'])
    result=[]
    for group in grouped.values():
        group['decisions']={f:np.concatenate(v) for f,v in group['decisions'].items()}
        result.append(group)
    return sorted(result,key=lambda g:(g['dataset'],g['target'],g['stage'],g['factor'])),errors


def rate(v,confidence=.95):
    v=np.asarray(v);valid=v>=0;n=int(valid.sum());N=len(v);k=int(v[valid].sum());missing=N-n
    if n==0:return dict(n=N,n_valid=0,missing=N,k=0,rate=None,ci95=[0.,1.],bounds=[0.,1.])
    ci=binomtest(k,n).proportion_ci(confidence,method='exact')
    # Conservative intervals allowing every completion of missing decisions.
    low=binomtest(k,N).proportion_ci(confidence,method='exact').low
    high=binomtest(k+missing,N).proportion_ci(confidence,method='exact').high
    return dict(n=N,n_valid=n,missing=missing,k=k,rate=k/n,ci95=[float(ci.low),float(ci.high)],
                bounds=[k/N,(k+missing)/N],conservative_ci=[float(low),float(high)])


def crossing(x,y,power=.95):
    for i in range(len(x)-1):
        if y[i]<=power<=y[i+1] and y[i+1]>y[i]:
            return float(x[i]+(power-y[i])/(y[i+1]-y[i])*(x[i+1]-x[i]))
    return None


def select_confirmation():
    assert not (ROOT/'confirmation_plan.json').exists()
    groups,errors=load_groups();assert not errors,errors
    assert len(groups)==34,len(groups)  # 32 screen points + two pooled null points
    jobs=[];selection=[]
    for dataset in ('may22','may26'):
        for target in (1,2,3,4):
            points=sorted([g for g in groups if g['dataset']==dataset and g['target']==target and g['stage']=='screen'],key=lambda g:g['factor'])
            assert len(points)==4 and all(g['n']==500 for g in points)
            factors=[g['factor'] for g in points];estimates={};powers={}
            for family in FAMILIES:
                y=[rate(g['decisions'][family][:,0])['rate'] for g in points]
                ymono=isotonic_regression(y).x
                center=crossing(factors,ymono)
                if center is None:raise RuntimeError(f'Screening does not bracket .95: {dataset} {target} {family} {y}')
                estimates[family]=center;powers[family]=y
            lo=min(estimates.values())-.08;hi=max(estimates.values())+.08
            selected=[round(lo,4),round((lo+hi)/2,4),round(hi,4)]
            for factor in selected:
                for chunk in range(3):
                    jobs.append(dict(dataset=dataset,target=target,factor=factor,n=1000,chunk=chunk,stage='confirm'))
            selection.append(dict(dataset=dataset,target=target,screen_factors=factors,screen_powers=powers,
                screen_crossings=estimates,confirmation_factors=selected,n_per_point=3000))
    (ROOT/'confirmation_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    (ROOT/'confirmation_plan.json').write_text(json.dumps(jobs,indent=2))
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__=='__main__':select_confirmation()
