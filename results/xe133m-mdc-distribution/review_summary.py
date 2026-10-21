"""Check exported summary against compact raw trials with separate arithmetic."""
from pathlib import Path
import json
import math
import statistics
import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'summary.json').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
a=np.load(ROOT/'trials_compact.npz')
assert a['x'].shape==(40000,21)
assert np.all(a['x']>=0) and np.all(a['x']==np.floor(a['x']))
checks=[]

def close(x,y):
    assert math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-12),(x,y)

def quantile(v,q):
    s=sorted(v);z=(len(s)-1)*q;lo=int(math.floor(z));hi=int(math.ceil(z))
    return s[lo]+(s[hi]-s[lo])*(z-lo)

for c in summary['cells']:
    ds=['may22','may26'].index(c['dataset']);sc=['null','signal'].index(c['scenario'])
    take=(a['dataset_index']==ds)&(a['scenario_index']==sc)
    assert take.sum()==10000
    assert np.array_equal(a['index'][take],np.arange(10000))
    f=c['family'];values=a[f+'_MDC'][take]*1000
    values=values[np.isfinite(values)].tolist()
    close(statistics.mean(values),c['mean']);close(statistics.median(values),c['median'])
    close(statistics.stdev(values),c['sd'])
    for q,k in [(.025,0),(.975,1)]:close(quantile(values,q),c['central95'][k])
    reference=manifest['datasets'][c['dataset']]['reference_ld'][f]['estimate']*1000
    close(reference,c['reference_ld'])
    close(math.fsum(v-reference for v in values)/len(values),c['bias'])
    close(math.sqrt(math.fsum((v-reference)**2 for v in values)/len(values)),c['rmse'])
    assert sum(v<reference for v in values)==c['below_reference']['k']
    decision=a[f+'_detected'][take];known=decision>=0
    assert np.array_equal(decision[known],a[f+'_free_activity'][take][known]>a[f+'_LC'][take][known])
    k=int(decision[known].sum());n=int(known.sum())
    assert (k,n)==(c['detection']['k'],c['detection']['n'])
    exact=[0.0 if k==0 else beta.ppf(.025,k,n-k+1),
           1.0 if k==n else beta.ppf(.975,k+1,n-k)]
    for x,y in zip(exact,c['detection']['ci95']):close(x,y)
    close(c['mean_minus_oracle']+c['oracle_minus_reference'],c['bias'])
    checks.append(dict(dataset=c['dataset'],scenario=c['scenario'],family=f,status='passed'))
for cell in summary['paired_method_comparisons']:
    ds=['may22','may26'].index(cell['dataset']);sc=['null','signal'].index(cell['scenario'])
    take=(a['dataset_index']==ds)&(a['scenario_index']==sc)
    w=a['wls_MDC'][take];p=a['poisson_MDC'][take];valid=np.isfinite(w)&np.isfinite(p)
    delta=[(float(x)-float(y))*1000 for x,y in zip(w[valid],p[valid])]
    close(statistics.mean(delta),cell['mean'])
    close(statistics.stdev(delta),cell['sd'])
result=dict(status='passed',n_observation_vectors=40000,n_MDC_summaries=8,
            separate_mean_sd_quantile_bias_rmse_checks=True,
            all_compact_decisions_recomputed=True,binomial_intervals_checked_via_beta_quantiles=True,
            paired_method_differences_checked=True,cells=checks)
(ROOT/'summary_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
