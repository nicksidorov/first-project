"""Independent audit of the May 22 large Lc/Ld experiment.

Does not import study, calibration, analyze, or xemat.detection. All raw counts,
seeds, full coordinates and decisions are checked. Selected fits are repeated;
Lc/MDC are checked with scaled QR Fisher inversion and brentq roots.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from scipy.stats import beta, binom, norm
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent
REPO=Path(os.environ.get('XEMAT_REPO','/workspace/first-project'))
sys.path.insert(0,str(REPO))
from xemat.model import JointModel
from xemat.solvers import fit
FAMILIES=('wls','poisson')
PILOTS={'wls':'nnls','poisson':'poisson_nonnegative'}
STAGE_CODES={'calibration':10,'validation':11,'signal':12,'extend':13,'hist_signal':15,'technical':99}

def read_json(path):return json.loads(Path(path).read_text())
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key_of(meta):return tuple(meta[k] for k in ('dataset','stage','target','factor','chunk','n'))

def cp_interval(k, n):
    """Clopper--Pearson endpoints from beta quantiles, independent of binomtest."""
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k))]


def verify_rate(reported, scores, cutoff):
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    n = int(valid.sum())
    k = int(np.count_nonzero(scores[valid] > cutoff))
    assert reported['n'] == len(scores)
    assert reported.get('n_valid', n) == n
    assert reported.get('missing', len(scores)-n) == len(scores)-n
    assert reported['k'] == k
    np.testing.assert_allclose(reported['rate'], k/n, rtol=0, atol=1e-15)
    np.testing.assert_allclose(reported['ci95'], cp_interval(k, n), rtol=1e-10, atol=1e-12)
    if 'bounds' in reported:
        N=len(scores)
        np.testing.assert_allclose(reported['bounds'],[k/N,(k+N-n)/N],rtol=0,atol=1e-15)
        expected=[cp_interval(k,N)[0],cp_interval(k+N-n,N)[1]]
        np.testing.assert_allclose(reported['conservative_ci'],expected,rtol=1e-10,atol=1e-12)


def linear_root(points, rule):
    """First upward crossing, written as a weighted activity average."""
    if points[0][rule]['rate'] > .95:
        return None
    if points[0][rule]['rate'] == .95:
        return points[0]['activity_mBq_m3']
    for left, right in zip(points[:-1], points[1:]):
        p, q = left[rule]['rate'], right[rule]['rate']
        if p <= .95 <= q and q > p:
            return (left['activity_mBq_m3']*(q-.95)
                    + right['activity_mBq_m3']*(.95-p))/(q-p)
    return None


def assert_endpoints(actual, expected, rtol=1e-11):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            assert a is not None
            np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-14)


def independent_grid_bounds(scores_by_factor, null_scores, cutoff_low, cutoff_high, nominal, total):
    confidence = 1-.025/18 if total else 1-.05/9
    tail = (1-confidence)/2
    lo, hi = 0., None
    for factor, scores in [(0., null_scores), *scores_by_factor]:
        finite = np.isfinite(scores)
        N, n = len(scores), int(finite.sum())
        k_liberal = int(np.count_nonzero(scores[finite] > cutoff_low))
        k_strict = int(np.count_nonzero(scores[finite] > cutoff_high))
        # Allow every completion of missing decisions in the conservative bound.
        high_k = k_liberal+N-n
        liberal_upper = 1. if high_k == N else beta.ppf(1-tail, high_k+1, N-high_k)
        strict_lower = 0. if k_strict == 0 else beta.ppf(tail, k_strict, N-k_strict+1)
        if liberal_upper < .95:
            lo = max(lo, factor)
        if strict_lower > .95:
            hi = factor if hi is None else min(hi, factor)
    return [lo*nominal, None if hi is None else hi*nominal]


def independent_bootstrap_curve(curve, signal_scores, calibration_bootstrap, master_seed):
    """Audit two complete paired curves using categorical enumeration.

    The production analysis obtains category sizes via marginal/both sums.
    Here direct bincount enumeration provides an independent calculation.
    """
    dataset, target = curve['dataset'], curve['target']
    dataset_index = ('may22', 'may26').index(dataset)
    points = curve['methods']['wls']['formal_points']
    factors = [point['factor'] for point in points]
    nominal = curve['nominal_MDC_mBq_m3']
    fitted = np.array([curve['methods'][f]['calibration']['c'] for f in FAMILIES])
    varying = calibration_bootstrap[dataset][:, :, target-1]
    repetitions = len(varying)
    result = {}
    for mode, code in [('standard', 3), ('conditional', 1), ('total', 2)]:
        fixed = np.ones(2) if mode == 'standard' else fitted
        cuts = np.broadcast_to(fixed, (repetitions, 2)) if mode != 'total' else varying
        rng = np.random.default_rng([master_seed, dataset_index, target, code])
        simulated = []
        for factor in factors:
            scores = np.stack([signal_scores[(dataset, target, factor, f)] for f in FAMILIES], axis=1)
            assert np.all(np.isfinite(scores)), 'Paired-bootstrap audit assumes no missing scores'
            if mode != 'total':
                decisions = scores > fixed
                category = 2*decisions[:, 0].astype(int)+decisions[:, 1].astype(int)
                counts = np.broadcast_to(np.bincount(category, minlength=4), (repetitions, 4))
            else:
                counts = []
                for cut in cuts:
                    decisions = scores > cut
                    category = 2*decisions[:, 0].astype(int)+decisions[:, 1].astype(int)
                    counts.append(np.bincount(category, minlength=4))
                counts = np.asarray(counts)
            draws = rng.multinomial(len(scores), counts/len(scores))
            simulated.append(np.column_stack([draws[:, 2]+draws[:, 3], draws[:, 1]+draws[:, 3]])/len(scores))
        powers = np.stack(simulated, axis=1)
        roots_by_method = []
        for j, family in enumerate(FAMILIES):
            roots = []
            for row in powers[:, :, j]:
                value = None
                if row[0] > .95:
                    value = -np.inf
                elif row[0] == .95:
                    value = factors[0]
                else:
                    for index in range(len(factors)-1):
                        p, q = row[index:index+2]
                        if p <= .95 <= q and q > p:
                            value = (factors[index]*(q-.95)+factors[index+1]*(.95-p))/(q-p)
                            break
                    if value is None:
                        value = np.inf if np.all(row < .95) else np.nan
                roots.append(value)
            roots = np.asarray(roots)
            with np.load(ROOT/'analysis_bootstrap.npz',allow_pickle=False) as saved:
                expected_roots=saved[f'{dataset}_i{target}_{family}_{mode}']
            np.testing.assert_allclose(roots,expected_roots,rtol=1e-12,atol=1e-14,equal_nan=True)
            roots_by_method.append(roots)
            metadata = curve['methods'][family]['bootstrap'][mode]
            assert metadata['n_repetitions'] == repetitions
            assert metadata['n_crossings'] == np.isfinite(roots).sum()
            assert metadata['n_below_grid'] == np.isneginf(roots).sum()
            assert metadata['n_above_grid'] == np.isposinf(roots).sum()
            assert metadata['n_ambiguous'] == np.isnan(roots).sum()
            if np.any(np.isnan(roots)):
                expected_ci = [None, None]
            else:
                percentile = np.quantile(roots, [.025, .975], method='inverted_cdf')
                expected_ci = [float(v)*nominal if np.isfinite(v) else None for v in percentile]
            assert_endpoints(curve['methods'][family][mode+'_ci95_mBq_m3'], expected_ci)
        wls, poisson = roots_by_method
        finite = np.isfinite(wls) & np.isfinite(poisson)
        difference = np.zeros(repetitions)
        difference[finite] = (wls[finite]-poisson[finite])*nominal
        lower = np.where(finite, difference, -np.inf)
        upper = np.where(finite, difference, np.inf)
        bounds = [np.quantile(lower, .025, method='inverted_cdf'),
                  np.quantile(upper, .975, method='inverted_cdf')]
        bounds = [float(v) if np.isfinite(v) else None for v in bounds]
        if mode != 'standard':
            pair = curve['paired_ld_difference']
            assert pair[mode+'_n_finite'] == finite.sum()
            assert_endpoints(pair[mode+'_ci95'], bounds)
        result[mode] = dict(repetitions=repetitions, paired_finite=int(finite.sum()))
    return dict(dataset=dataset, target=target, modes=result)


def qr_covariance(S, mu):
    """Full Fisher inverse via scaled QR, independently of production SVD."""
    assert np.all(mu > 0)
    weighted = S / np.sqrt(mu)[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    assert np.all(scale > 0)
    _, R = np.linalg.qr(weighted/scale, mode='reduced')
    inverse = solve_triangular(R, np.eye(S.shape[1]))
    return (inverse @ inverse.T) / scale[:, None] / scale[None, :]


def independent_limits(S, b, positive, target):
    nuisance=np.array(positive,dtype=float,copy=True);nuisance[target]=0
    mu0=b+S@nuisance
    assert np.all(mu0>0)
    z=float(norm.ppf(.95));sigma0=float(np.sqrt(qr_covariance(S,mu0)[target,target]))
    lc=z*sigma0
    def sigma(a):return float(np.sqrt(qr_covariance(S,mu0+a*S[:,target])[target,target]))
    def residual(a):return a-lc-z*sigma(a)
    upper=2*lc
    for _ in range(100):
        if residual(upper)>0:break
        upper*=2
    else:raise AssertionError('MDC root not bracketed')
    ld=brentq(residual,0.,upper,xtol=1e-15,rtol=2e-13)
    return dict(LC=lc,MDC=ld,sigma0=sigma0,sigma_at_MDC=sigma(ld))


def verify_inputs():
    manifest=read_json(ROOT/'manifest.json');lock=read_json(ROOT/'run_lock.json')
    for name,digest in lock.items():assert sha256(ROOT/name)==digest,name
    for name,digest in manifest['production_sha256'].items():assert sha256(REPO/name)==digest,name
    assert list(manifest['datasets'])==['may22'] and manifest['families']==list(FAMILIES)
    assert manifest['master_seed']==2026102701 and manifest['bootstrap']['master_seed']==2026102709
    assert manifest['alpha']==manifest['beta_ii']==.05 and manifest['systematics'] is False
    assert manifest['calibration_n_per_dataset']==manifest['validation_n_per_dataset']==100000
    assert manifest['signal_n_per_point']==manifest['histogram_signal_n_per_isotope_method']==20000
    assert manifest['stage_codes']==STAGE_CODES
    data=manifest['datasets']['may22'];model=JointModel(**data['model']);S=model.physical_matrix
    assert S.shape==(21,17) and model.physical_coordinate_label=='[a_V; gamma; zeta]'
    np.testing.assert_array_equal(data['truth0'],[0.]*10+[522.,854.,494.,66.,64.,89.,97.])
    np.testing.assert_array_equal(model.b,np.zeros(21))
    np.testing.assert_array_equal(data['mu0'],S@data['truth0']+model.b)
    for name in ('sample_efficiencies','gasbk_efficiencies'):
        np.testing.assert_array_equal(data['metadata'][name],[.241169,.367784,.233937,.233937])
    formula=[]
    for target in range(1,5):
        independent=independent_limits(S,model.b,data['truth0'],target)
        for name,field in [('LC','nominal_LC'),('MDC','nominal_MDC'),('sigma0','nominal_sigma0'),('sigma_at_MDC','nominal_sigma_at_MDC')]:
            np.testing.assert_allclose(independent[name],data[field][target-1],rtol=3e-8,atol=1e-14)
        formula.append(dict(target=target,**independent))
    return manifest,model,dict(independent_formula_checks=formula,all_production_hashes_unchanged=True)


def load_plans(manifest, calibration_only=False):
    plan_names=['calibration_plan.json'] if calibration_only else ['calibration_plan.json','validation_plan.json','signal_plan.json']
    if not calibration_only and (ROOT/'extension_plan.json').exists():
        marker=read_json(ROOT/'extension_plan.lock.json')
        assert marker['plan_sha256']==sha256(ROOT/'extension_plan.json')
        assert marker['manifest_sha256']==sha256(ROOT/'manifest.json')
        plan_names.append('extension_plan.json')
    planned=[j for name in plan_names for j in read_json(ROOT/name)]
    expected={key_of(j):j for j in planned}
    assert len(expected)==len(planned)
    counts=defaultdict(int);seed_set=set();signals=defaultdict(list)
    for j in planned:
        assert j['dataset']=='may22' and j['n']==1000
        assert abs(round(j['factor']*10000)/10000-j['factor'])<1e-12
        seed=(manifest['master_seed'],0,j['target'],round(j['factor']*10000),STAGE_CODES[j['stage']],j['chunk'])
        assert seed not in seed_set;seed_set.add(seed)
        counts[j['stage']]+=j['n']
        if j['stage'] in ('calibration','validation'):
            assert j['target']==0 and j['factor']==0 and 0<=j['chunk']<100
        else:
            assert j['stage'] in ('signal','extend') and j['target'] in (1,2,3,4)
            assert 0<=j['chunk']<20
            signals[(j['target'],j['factor'],j['stage'])].append(j['chunk'])
    assert counts['calibration']==100000
    if not calibration_only:
        assert counts['validation']==100000 and counts['signal']==320000
        for target in range(1,5):
            base=manifest['signal_plan'][str(target)]
            actual=sorted(k[1] for k in signals if k[0]==target and k[2]=='signal')
            assert actual==base
            extras=sorted(k[1] for k in signals if k[0]==target and k[2]=='extend')
            assert len(extras)<=4
            for factor in extras:
                assert factor<base[0] or factor>base[-1]
                distance=(base[0]-factor if factor<base[0] else factor-base[-1])/.04
                assert abs(distance-round(distance))<1e-10 and distance>=1-1e-10
        assert all(sorted(chunks)==list(range(20)) for chunks in signals.values())
    return expected,dict(counts)


def audit_batches(manifest,model,expected,calibration_only=False):
    stages={'calibration'} if calibration_only else {'calibration','validation','signal','extend'}
    paths=[]
    for path in sorted((ROOT/'points').glob('*.json')):
        if read_json(path)['stage'] in stages:paths.append(path)
    assert len(paths)==len(expected),'Incomplete planned data'
    seen=set();seeds=set();null=defaultdict(list);signal=defaultdict(list)
    cov=defaultdict(int);counts=defaultdict(int);errors=[];sources={}
    n_refits=n_limits=missing_decisions=mdc_valid=mdc_missing=0;max_lc_relative=0.;max_mdc_relative=0.
    S=model.physical_matrix
    for path in paths:
        meta=read_json(path);k=key_of(meta)
        assert k in expected and k not in seen;seen.add(k)
        assert all(meta[name]==value for name,value in expected[k].items())
        assert meta['manifest_sha256']==sha256(ROOT/'manifest.json')
        assert meta['npz_sha256']==sha256(path.with_suffix('.npz'))
        sources[path.stem]={'json':sha256(path),'npz':meta['npz_sha256']}
        assert meta['family']=='both' and meta['families']==list(FAMILIES)
        assert meta['mode']==('iterative' if meta['stage']=='validation' else 'fixed')
        assert meta['raw_activity_unit']=='Bq/m3' and meta['alpha']==meta['beta_ii']==.05
        seed=[manifest['master_seed'],0,meta['target'],round(10000*meta['factor']),STAGE_CODES[meta['stage']],meta['chunk']]
        assert meta['seed']==seed and tuple(seed) not in seeds;seeds.add(tuple(seed))
        indices=np.asarray([1,2,3,4] if meta['target']==0 else [meta['target']])
        np.testing.assert_array_equal(meta['targets'],indices)
        truth=np.asarray(manifest['datasets']['may22']['truth0']).copy()
        if meta['target']:truth[meta['target']]=meta['factor']*manifest['datasets']['may22']['nominal_MDC'][meta['target']-1]
        np.testing.assert_array_equal(meta['truth'],truth)
        mu=S@truth+model.b;np.testing.assert_array_equal(meta['mu'],mu)
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
            arrays={name:archive[name] for name in archive.files}
        n=meta['n'];assert arrays['x'].shape==(n,21)
        np.testing.assert_array_equal(arrays['x'],np.random.default_rng(seed).poisson(mu,size=(n,21)))
        np.testing.assert_array_equal(arrays['indices'],np.arange(n))
        np.testing.assert_array_equal(arrays['targets'],indices)
        counts[meta['stage']]+=n;errors.extend(dict(source=path.name,**e) for e in meta['errors'])
        for family in FAMILIES:
            for kind in ('free','positive'):
                p=arrays[family+'_'+kind];ok=arrays[family+'_'+kind+'_success']
                assert p.shape==(n,17) and ok.shape==(n,) and ok.dtype==np.dtype(bool)
                assert np.all(np.isfinite(p[ok])) and np.all(np.isnan(p[~ok]))
                if kind=='positive':assert np.all(p[ok]>=0)
                if family=='poisson' and np.any(ok):
                    expected_mu=p[ok]@S.T+model.b
                    roundoff=256*np.finfo(float).eps*(np.abs(p[ok])@np.abs(S).T+model.b+1)
                    assert np.all(expected_mu>=-roundoff)
                    assert np.all(expected_mu[arrays['x'][ok]>0]>0)
            avail=arrays[family+'_free_covariance_available']
            assert avail.shape==(n,) and avail.dtype==np.dtype(bool)
            cov[(meta['stage'],family)]+=int((~avail).sum())
            lc=arrays[family+'_LC'];free=arrays[family+'_free'][:,indices];decision=arrays[family+'_decision']
            assert lc.shape==decision.shape==(n,len(indices))
            assert np.all(np.isin(decision,[-1,0,1]))
            valid=np.isfinite(lc)&(lc>0)&np.isfinite(free)
            np.testing.assert_array_equal(valid,decision>=0)
            np.testing.assert_array_equal(decision[valid],(free>lc)[valid])
            ratio=np.divide(free,lc,out=np.full_like(lc,np.nan),where=valid)
            missing_decisions+=int((~valid).sum())
            for col,target in enumerate(indices):
                score=ratio[:,col]
                saved_rate=meta['rates'][family][col];available=np.isfinite(score);nn=int(available.sum());kk=int((score[available]>1).sum())
                assert saved_rate['n_valid']==nn and saved_rate['n_missing']==n-nn and saved_rate['k']==kk
                if nn:
                    np.testing.assert_allclose(saved_rate['rate'],kk/nn,rtol=0,atol=1e-15)
                    np.testing.assert_allclose(saved_rate['ci95'],cp_interval(kk,nn),rtol=1e-10,atol=1e-12)
                if meta['stage'] in ('calibration','validation'):
                    null[(meta['stage'],family,int(target))].append((meta['chunk'],score))
                else:signal[('may22',int(target),meta['factor'],family)].append(score)
            if meta['mode']=='iterative':
                mdc=arrays[family+'_MDC'];converged=arrays[family+'_converged']
                assert mdc.shape==converged.shape==lc.shape
                assert np.all(np.isfinite(mdc[converged])) and np.all(mdc[converged]>0)
                assert np.all(np.isnan(mdc[~converged]))
                mdc_valid+=int(converged.sum());mdc_missing+=int((~converged).sum())
        for row in (0,n-1):
            trial=JointModel(model.M,arrays['x'][row],model.b,model.n_sample_parameters,
                             model.T_c,model.coordinates,model.n_detector_parameters)
            for family in FAMILIES:
                both_success=True
                for kind,method in [('free',family),('positive',PILOTS[family])]:
                    result=fit(trial,method);stored_ok=bool(arrays[family+'_'+kind+'_success'][row])
                    result_ok=bool(result.success and result.physical_parameters is not None
                                   and np.all(np.isfinite(result.physical_parameters)))
                    assert result_ok==stored_ok
                    if stored_ok:
                        np.testing.assert_allclose(result.physical_parameters,arrays[family+'_'+kind][row],rtol=1e-8,atol=1e-10)
                    else:both_success=False
                    if kind=='free' and stored_ok:
                        assert (result.physical_covariance is not None)==bool(arrays[family+'_free_covariance_available'][row])
                    n_refits+=1
                if not both_success:continue
                for col,target in enumerate(indices):
                    if not np.isfinite(arrays[family+'_LC'][row,col]):continue
                    limits=independent_limits(S,model.b,arrays[family+'_positive'][row],int(target))
                    stored_lc=arrays[family+'_LC'][row,col]
                    np.testing.assert_allclose(stored_lc,limits['LC'],rtol=2e-10,atol=1e-14)
                    max_lc_relative=max(max_lc_relative,abs(stored_lc/limits['LC']-1));n_limits+=1
                    if meta['mode']=='iterative' and arrays[family+'_converged'][row,col]:
                        for name,field in [('MDC','MDC'),('sigma0','sigma0'),('sigma_at_MDC','sigma_at_MDC')]:
                            np.testing.assert_allclose(arrays[family+'_'+field][row,col],limits[name],rtol=3e-8,atol=1e-14)
                        max_mdc_relative=max(max_mdc_relative,abs(arrays[family+'_MDC'][row,col]/limits['MDC']-1))
        print('AUDIT',path.stem,flush=True)
    assert seen==set(expected)
    null={key:np.concatenate([a for _,a in sorted(rows,key=lambda v:v[0])]) for key,rows in null.items()}
    signal={key:np.concatenate(rows) for key,rows in signal.items()}
    report=dict(main_vectors_checked=sum(counts.values()),vectors_by_stage=dict(counts),batches_checked=len(paths),
        unique_seeds_checked=len(seeds),refitted_models=n_refits,independent_limits_checked=n_limits,
        max_relative_LC_difference=max_lc_relative,max_relative_MDC_difference=max_mdc_relative,
        missing_decisions=missing_decisions,validation_MDC_valid=mdc_valid,validation_MDC_missing=mdc_missing,
        errors=errors,source_batches=sources,
        free_covariance_unavailable=[dict(stage=key[0],family=key[1],n=value) for key,value in sorted(cov.items())])
    return null,signal,report,cov


def audit_calibration(manifest,null):
    calibration=read_json(ROOT/'calibration.json');marker=read_json(ROOT/'cutoff.json')
    assert marker['frozen'] and marker['validation_not_used'] and calibration['validation_not_used']
    assert marker['calibration_sha256']==sha256(ROOT/'calibration.json')
    assert marker['bootstrap_sha256']==sha256(ROOT/'calibration_bootstrap.npz')
    assert calibration['bootstrap']['sha256']==sha256(ROOT/'calibration_bootstrap.npz')
    assert calibration['family_order']==list(FAMILIES) and calibration['target_order']==[1,2,3,4]
    assert list(calibration['datasets'])==['may22']
    for name,digest in calibration['source_sha256'].items():
        assert sha256(ROOT/name)==digest,name
        assert 'validation' not in name and 'hist_signal' not in name
    rows=[];cutoffs={}
    for family in FAMILIES:
        for target in range(1,5):
            item=calibration['datasets']['may22']['methods'][family][target-1]
            values=null[('calibration',family,target)]
            assert values.shape==(100000,) and np.all(np.isfinite(values))
            ordered=np.sort(values);rank=int(np.ceil((len(values)+1)*.95));c=ordered[rank-1]
            assert rank==95001 and item['n']==len(values) and item['k_order']==rank
            assert item['target']==target and item['c']==c and c>0
            assert marker['coefficients']['may22'][family][target-1]==c
            assert item['ties']==int((values==c).sum())
            assert item['training_exceedances']==int((values>c).sum())
            np.testing.assert_allclose(item['achieved_training_rate'],np.mean(values>c),rtol=0,atol=1e-15)
            for confidence,label in [(.95,'ci95_c'),(.975,'ci975_c')]:
                tail=(1-confidence)/2
                low_rank=int(binom.ppf(tail,len(values),.95));high_rank=int(binom.ppf(1-tail,len(values),.95))+1
                np.testing.assert_array_equal(item[label],[ordered[low_rank-1],ordered[high_rank-1]])
            np.testing.assert_allclose(item['alpha_interval_from_calibration'],beta.ppf([.025,.975],len(values)+1-rank,rank),rtol=1e-12,atol=1e-15)
            np.testing.assert_allclose(item['marginal_alpha_without_ties'],(len(values)+1-rank)/(len(values)+1),rtol=0,atol=1e-15)
            cutoffs[(family,target)]=float(c)
            rows.append(dict(family=family,target=target,n=len(values),rank=rank,c=float(c),ties=item['ties'],training_exceedances=item['training_exceedances']))
    ratios=np.stack([np.stack([null[('calibration',f,i)] for i in range(1,5)],axis=1) for f in FAMILIES],axis=1)
    rng=np.random.default_rng([manifest['bootstrap']['master_seed'],0,0])
    with np.load(ROOT/'calibration_bootstrap.npz',allow_pickle=False) as archive:
        assert archive.files==['may22']
        bootstrap=archive['may22']
        assert bootstrap.shape==(2000,2,4) and np.all(np.isfinite(bootstrap))
        for row in range(3):
            sample=rng.integers(0,100000,size=100000)
            np.testing.assert_array_equal(np.sort(ratios[sample],axis=0)[95000],bootstrap[row])
    return calibration,cutoffs,dict(calibration_order_statistics=rows,paired_calibration_bootstrap_quantiles_checked=24)


def audit_summary(manifest,null,signal,batches,cov,calibration,cutoffs):
    summary=read_json(ROOT/'summary.json')
    assert summary['manifest_sha256']==sha256(ROOT/'manifest.json')
    assert summary['cutoff_sha256']==sha256(ROOT/'cutoff.json')
    assert summary['analysis_sha256']==sha256(ROOT/'analyze.py')
    assert summary['bootstrap_file']=='analysis_bootstrap.npz'
    assert summary['bootstrap_sha256']==sha256(ROOT/'analysis_bootstrap.npz')
    assert summary['technical_vectors_excluded']==manifest['technical_vectors_excluded']==128
    assert summary['n_reused_signal']==0 and summary['unit']=='mBq/m3'
    counts=batches['vectors_by_stage']
    expected_counts=dict(calibration=100000,validation=100000,signal=320000,signal_extension=counts.get('extend',0))
    assert summary['n_new_by_stage']==expected_counts
    assert counts['calibration']==100000 and counts['validation']==100000 and counts['signal']==320000
    assert len(summary['curves'])==4
    expected_errors=[]
    for path in sorted((ROOT/'points').glob('*.json')):
        meta=read_json(path)
        if meta['stage'] in ('signal','extend','validation'):
            expected_errors.extend((e,str(path)) for e in meta['errors'])
    assert len(summary['errors'])==len(expected_errors)
    for error,source in expected_errors:
        assert any(entry['error']==error and entry['source']==source for entry in summary['errors'])
    cov_rows={r['stage']:r for r in summary['covariance_diagnostics']}
    assert len(cov_rows)==len(summary['covariance_diagnostics']) and set(cov_rows)==set(counts)
    for stage,n in counts.items():
        assert cov_rows[stage]['n']==n and cov_rows[stage]['dataset']=='may22'
        for family in FAMILIES:assert cov_rows[stage][family+'_unavailable']==cov[(stage,family)]
    with np.load(ROOT/'analysis_bootstrap.npz',allow_pickle=False) as a:
        bootstrap={name:a[name] for name in a.files}
    expected_keys={f'may22_i{i}_{f}_{mode}' for i in range(1,5) for f in FAMILIES for mode in ('standard','conditional','total')}
    assert set(bootstrap)==expected_keys
    rate_checks=root_checks=0
    for curve in summary['curves']:
        assert curve['dataset']=='may22'
        target=curve['target'];assert target in range(1,5)
        data=manifest['datasets']['may22'];nominal=data['nominal_MDC'][target-1]*1000
        np.testing.assert_allclose(curve['nominal_MDC_mBq_m3'],nominal,rtol=0,atol=0)
        np.testing.assert_allclose(curve['nominal_LC_mBq_m3'],data['nominal_LC'][target-1]*1000,rtol=0,atol=0)
        for family in FAMILIES:
            method=curve['methods'][family];c=cutoffs[(family,target)]
            item=calibration['datasets']['may22']['methods'][family][target-1]
            assert method['calibration']==item
            validation=null[('validation',family,target)];assert validation.shape==(100000,)
            verify_rate(method['validation']['standard'],validation,1)
            verify_rate(method['validation']['calibrated'],validation,c);rate_checks+=2
            points=method['formal_points'];factors=[p['factor'] for p in points]
            assert factors==sorted(k[2] for k in signal if k[1]==target and k[3]==family)
            assert len(factors)<=8 and factors==sorted(set(factors))
            for point in points:
                scores=signal[('may22',target,point['factor'],family)]
                assert scores.shape==(20000,)
                np.testing.assert_allclose(point['activity_mBq_m3'],point['factor']*nominal,rtol=1e-13)
                verify_rate(point['standard'],scores,1);verify_rate(point['calibrated'],scores,c);rate_checks+=2
                assert (point['calibrated']['k']-point['standard']['k'])*(c-1)<=0
            for rule,label in [('standard','standard_ld_mBq_m3'),('calibrated','calibrated_ld_mBq_m3')]:
                root=linear_root(points,rule)
                if root is None:assert method[label] is None
                else:np.testing.assert_allclose(method[label],root,rtol=1e-12,atol=1e-14)
                root_checks+=1
            assert method['empirical_reversals']==int(np.sum(np.diff([p['calibrated']['rate'] for p in points])<0))
            scores_by_factor=[(p['factor'],signal[('may22',target,p['factor'],family)]) for p in points]
            for label,low,high,total in [('standard',1,1,False),('conditional',c,c,False),('total',*item['ci975_c'],True)]:
                expected=independent_grid_bounds(scores_by_factor,validation,low,high,nominal,total)
                assert_endpoints(method[label+'_grid_bound95_mBq_m3'],expected)
            for mode in ('standard','conditional','total'):
                roots=bootstrap[f'may22_i{target}_{family}_{mode}'];assert roots.shape==(2000,)
                record=method['bootstrap'][mode]
                assert record['n_repetitions']==2000 and record['n_crossings']==int(np.isfinite(roots).sum())
                assert record['n_below_grid']==int(np.isneginf(roots).sum()) and record['n_above_grid']==int(np.isposinf(roots).sum())
                assert record['n_ambiguous']==int(np.isnan(roots).sum())
                expected=[None,None] if np.any(np.isnan(roots)) else [float(v) if np.isfinite(v) else None for v in np.quantile(roots,[.025,.975],method='inverted_cdf')]
                assert_endpoints(record['ci95_factor'],expected)
                assert_endpoints(method[mode+'_ci95_mBq_m3'],[None if v is None else v*nominal for v in expected])
        pair=curve['paired_ld_difference']
        if pair:
            delta=curve['methods']['wls']['calibrated_ld_mBq_m3']-curve['methods']['poisson']['calibrated_ld_mBq_m3']
            np.testing.assert_allclose(pair['point'],delta,rtol=1e-12,atol=1e-14)
            for mode in ('conditional','total'):
                a=bootstrap[f'may22_i{target}_wls_{mode}'];b=bootstrap[f'may22_i{target}_poisson_{mode}']
                finite=np.isfinite(a)&np.isfinite(b);low=np.full(2000,-np.inf);high=np.full(2000,np.inf)
                low[finite]=(a[finite]-b[finite])*nominal;high[finite]=low[finite]
                bounds=[np.quantile(low,.025,method='inverted_cdf'),np.quantile(high,.975,method='inverted_cdf')]
                assert_endpoints(pair[mode+'_ci95'],[float(v) if np.isfinite(v) else None for v in bounds])
                assert pair[mode+'_n_finite']==int(finite.sum())
    with np.load(ROOT/'calibration_bootstrap.npz',allow_pickle=False) as a:cal_boot={name:a[name] for name in a.files}
    bootstrap_audit=[]
    for target in (2,4):
        curve=next(c for c in summary['curves'] if c['target']==target)
        bootstrap_audit.append(independent_bootstrap_curve(curve,signal,cal_boot,manifest['bootstrap']['master_seed']))
    return dict(reported_rate_checks=rate_checks,independent_ld_interpolations=root_checks,
        paired_power_bootstrap_rebuilds=bootstrap_audit,all_grid_bounds_independently_recomputed=True,
        all_saved_bootstrap_intervals_checked=True,remaining_extension_requests=summary['needs_extension'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only',action='store_true')
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest,model,input_report=verify_inputs()
    if args.inputs_only:
        report=input_report
    else:
        expected,counts=load_plans(manifest,args.calibration_only)
        null,signal,batch_report,cov=audit_batches(manifest,model,expected,args.calibration_only)
        assert batch_report['vectors_by_stage']==counts
        calibration,cutoffs,cal_report=audit_calibration(manifest,null)
        report={**input_report,**batch_report,**cal_report}
        if not args.calibration_only:
            report.update(audit_summary(manifest,null,signal,batch_report,cov,calibration,cutoffs))
    report.update(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256(ROOT/'manifest.json'),run_lock_sha256=sha256(ROOT/'run_lock.json'),
        audit_file_sha256=sha256(Path(__file__)),selected_refit_rows_per_batch=[0,999],
        audit_scope='inputs' if args.inputs_only else ('calibration' if args.calibration_only else 'main'),
        independent_method='All x/decisions/counts/summary; repeated fits; scaled QR Fisher and brentq MDC roots')
    if not args.inputs_only:
        report.update(calibration_sha256=sha256(ROOT/'calibration.json'),cutoff_sha256=sha256(ROOT/'cutoff.json'),
                      calibration_bootstrap_sha256=sha256(ROOT/'calibration_bootstrap.npz'))
    if not args.inputs_only and not args.calibration_only:
        report.update(summary_sha256=sha256(ROOT/'summary.json'),analysis_bootstrap_sha256=sha256(ROOT/'analysis_bootstrap.npz'))
    destination=args.output or ROOT/('verification_inputs.json' if args.inputs_only else ('verification_calibration.json' if args.calibration_only else 'verification.json'))
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_batches','calibration_order_statistics')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

