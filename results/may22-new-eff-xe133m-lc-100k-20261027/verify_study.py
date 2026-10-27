"""Independent audit of saved May22 Xe133m LC Monte Carlo artifacts.

This script does not simulate observations, optimize parameters, or calculate
LD/MDC. It checks saved observations, statuses, decisions, summaries and hashes.
Saved LC values are spot-checked with a scaled QR covariance independent of
XeMat's production SVD implementation. Run after analyze.py with:

    python verify_study.py

Use --partial while the original batches are still being completed. The
default audit requires all 100 batches and the merged NPZ and CSV.
Use --from-replicates in the portable archive without duplicated raw batches.
That mode checks all saved rows and summaries but cannot compare the merged
file to absent original batch NPZ bytes; the original full audit records this
comparison in audit_result.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parent
FAMILIES = ('wls', 'poisson')
EXPECTED_X = np.array([105,176,128,107,33,34,13,
                       52,86,47,51,12,12,11,
                       522,854,494,66,64,89,97])
EXPECTED_EFFICIENCIES = [.241169,.367784,.233937,.233937]
EXPECTED_RS = np.array([
    [1,0,0,0,0],[.65,1,0,0,0],[1.4,.01,1,.004,0],
    [.054,0,.32,1,.05],[.072,.01,.17,.043,1],
    [.03,.02,.34,.0456,0],[.089,0,.0576,0,.05]])
EXPECTED_RG = np.array([
    [1,0,0,0,0],[.65,1,0,0,0],[1.4,.01,1,.004,0],
    [.054,0,.33,1,.05],[.072,.01,.18,.049,1],
    [.03,.01,.18,.0456,0],[.089,0,.0576,0,.05]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, rtol=2e-11, atol=1e-12):
    check(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), message)


def same(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, message + ': shapes differ')
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        check(np.array_equal(a, b, equal_nan=True), message)
    else:
        check(np.array_equal(a, b), message)


def baseline():
    scenario = read(ROOT/'scenario.json')
    lock = read(ROOT/'runner_lock.json')
    for path, value in read(ROOT/'scenario.lock.json').items():
        check(sha(ROOT/path) == value, 'Scenario lock hash: ' + path)
    for filename in ('study.py','lc_only.py','scenario.json','scenario_arrays.npz'):
        field = filename.replace('.py','').replace('.json','').replace('.npz','')+'_sha256'
        check(sha(ROOT/filename) == lock[field], 'Runner lock hash: '+filename)
    for path, value in lock['core_sha256'].items():
        check(sha(ROOT/'kernel_snapshot'/path) == value, 'Kernel hash: '+path)
    for value in scenario['input_files'].values():
        check(sha(ROOT/value['path']) == value['sha256'], 'Input file hash')
    check(scenario['n_generated'] == 100_000 and scenario['batch_size'] == 1000,
          'Requested number of observations')
    check(scenario['detector'] == 221, 'Detector 221')
    check(scenario['target_index'] == 4 and scenario['target_isotope'] == 'Xe133m',
          'Only the fresh Xe133m detection is tested')
    check(scenario['fixed_roi'] and scenario['fixed_efficiencies'], 'Fixed calibration options')
    same(scenario['sample_efficiencies'], EXPECTED_EFFICIENCIES, 'SAMPLE efficiency')
    same(scenario['gasbk_efficiencies'], EXPECTED_EFFICIENCIES, 'GASBK efficiency')
    check(scenario['branching'] == dict(Rn222=.37,Xe135=.902,Xe133=.37,Xe131m=.54,Xe133m=.5577),
          'Branching probabilities')
    check(scenario['radon_mode'] == 'legacy' and scenario['radon_efficiency'] == .3,
          'Legacy radon calibration')
    check(scenario['metadata']['time_parameters'] == dict(t_sr=50591.18,t_sl=50400.,
          t_gr=25301.71,t_gl=25200.,t_a=43173.,t_p=16401.1,tau=32970.,t_bl=259200.),
          'Original measurement and transfer times')
    close(scenario['metadata']['effective_air_volume_m3'],14.382643678160921,'Effective air volume')
    check(scenario['source_fit_count'] == 1, 'Only one source truth fit')
    check(scenario['source_fit_success'], 'Reference nonnegative fit succeeded')
    check(scenario['no_mdc_or_ld_calculation'] and scenario['no_null_refit']
          and scenario['no_threshold_multiplier'], 'No additional detection calculations')
    with np.load(ROOT/'scenario_arrays.npz',allow_pickle=False) as saved:
        a = {key:saved[key] for key in saved.files}
    S,M,T,b,z,reference = (a[k] for k in ('S','M','T_V','b','z_truth','z_reference'))
    check(S.shape == (21,17) and M.shape == (21,17), 'Joint 21-observation,17-parameter model')
    check(np.linalg.matrix_rank(S/np.linalg.norm(S,axis=0)) == 17, 'Full design rank')
    same(a['x'],EXPECTED_X,'Original observed ROI counts')
    same(a['roi_order'],[1,2,3,5,6,7,8],'Fixed ROI order')
    check(np.all(z >= 0) and np.all(reference >= 0), 'Nonnegative physical truth')
    check(z[4] == 0, 'Tested fresh isotope absent')
    same(z[np.arange(17)!=4],reference[np.arange(17)!=4],'Other 16 physical coordinates preserved')
    same(a['mu0'],S@z+b,'Truth model mean')
    check(np.all(a['mu0'] > 0), 'Positive generating Poisson means')
    Q=np.eye(17);Q[:5,:5]=T
    close(S@Q,M,'Physical-to-spectral coordinate mapping')
    check(T[2,4] < 0,'Xe133m chain transformation is preserved')
    same(S[14:,10:],np.eye(7),'Seven jointly modeled DETBK means')
    same(S[14:,:10],np.zeros((7,10)),'DETBK has no fresh or memory signal')
    same(S[7:,4],np.zeros(14),'Deleting only fresh Xe133m preserves GASBK and DETBK')
    same(a['calibration_R_c0'],EXPECTED_RS,'Original detector221 Rs, all cells')
    same(a['calibration_R_g0'],EXPECTED_RG,'Original detector221 Rg, all cells')
    sensitivity=np.r_[.3,EXPECTED_EFFICIENCIES]*np.array([.37,.902,.37,.54,.5577])
    close(a['calibration_I_c'],sensitivity,'SAMPLE epsilon times branching, exactly once')
    close(a['calibration_I_g'],sensitivity,'GASBK epsilon times branching, exactly once')
    decomposition=sum(a['mu0_'+name] for name in
                      ('fresh_radon','fresh_xenon','memory','detector_background','fixed_offset'))
    close(decomposition,a['mu0'],'Stored physical component decomposition')
    fit=read(ROOT/'source_fit.json')
    check(fit['detection'] is None and fit['no_detection_calculation'], 'No source MDC/LC calculation')
    same(fit['fit']['physical_parameters'],reference,'Saved source fit and reference vector')
    close(fit['fit']['mu'],S@reference+b,'Saved source fitted mean')
    return scenario,a


def qr_lc(S,b,z,alpha):
    """Variance e_i^T(S^T diag(1/mu) S)^-1 e_i from scaled QR."""
    null=z.copy();null[4]=0
    mu=S@null+b
    check(np.all(mu>0),'Positive auxiliary null spectrum')
    A=S/np.sqrt(mu[:,None])
    scale=np.linalg.norm(A,axis=0)
    _,R=np.linalg.qr(A/scale,mode='reduced')
    unit=np.zeros(17);unit[4]=1
    v=np.linalg.solve(R.T,unit)
    sigma=np.linalg.norm(v)/scale[4]
    return 1000*norm.ppf(1-alpha)*sigma


def verify_batch(chunk,scenario,model,qr_per_batch,r=None,meta=None):
    path=ROOT/'points'/f'batch_{chunk:03d}.npz'
    from_raw_batch=r is None
    if from_raw_batch:
        meta=read(path.with_suffix('.json'))
    check(meta['chunk']==chunk and meta['n_generated']==1000,'Batch identifier and size')
    check(meta['seed']==[scenario['master_seed'],chunk],'Batch seed identity')
    if from_raw_batch:
        check(meta['npz_sha256']==sha(path),'Batch NPZ hash')
    check(meta['runner_lock_sha256']==sha(ROOT/'runner_lock.json'),'Batch source lock')
    if from_raw_batch:
        with np.load(path,allow_pickle=False) as saved:
            r={key:saved[key] for key in saved.files}
    same(r['replicate_id'],np.arange(chunk*1000,(chunk+1)*1000),'All original realization IDs')
    same(r['chunk'],np.full(1000,chunk),'Chunk vector')
    same(r['row_in_batch'],np.arange(1000),'Within-batch row IDs')
    same(r['seed'],[scenario['master_seed'],chunk],'Saved seed')
    check(r['x'].shape==(1000,21) and r['x'].dtype.kind in 'iu' and np.all(r['x']>=0),
          'All 21 raw Poisson counts retained')
    check(str(r['concentration_units'])=='mBq/m3','Concentration and LC units')
    same(r['isotope_order'],['Xe135','Xe133','Xe131m','Xe133m'],'Concentration order')
    expected_errors=[]
    qr_checks=0;max_error=0.
    for family in FAMILIES:
        free=r[family+'_free_status']==1
        aux=r[family+'_auxiliary_status']==1
        lc=r[family+'_LC_status']==1
        d=r[family+'_decision'];valid=d>=0
        check(np.all(np.isin(d,[-1,0,1])),'Decision status values')
        same(valid,free&lc,'Decision availability does not depend on free covariance')
        check(np.all(~lc|aux),'LC requires available auxiliary fit')
        same(d[valid],r[family+'_concentrations'][valid,3]>r[family+'_LC'][valid],
             'Strict free concentration > own LC')
        same(r[family+'_concentrations'],r[family+'_free_physical'][:,1:5]*1000,
             'Free signed concentrations preserved and converted once')
        check(np.all(np.isfinite(r[family+'_LC'][lc])) and np.all(r[family+'_LC'][lc]>=0),
              'Available LC values are finite and nonnegative')
        close(r[family+'_LC'][lc],norm.ppf(1-scenario['alpha'])*r[family+'_sigma0'][lc],
              'LC normal quantile and native-unit conversion')
        for phase in ('free','auxiliary'):
            prefix=family+'_'+phase
            status=r[prefix+'_status'];good=status==1
            check(np.all(np.isin(status,[-1,0,1,2])),'Every fit attempted exactly once')
            check(r[prefix+'_physical'].shape==(1000,17),'All 17 physical estimates retained')
            check(np.all(np.isfinite(r[prefix+'_physical'][good])),'Successful fit parameters finite')
            if phase=='auxiliary':
                check(np.all(r[prefix+'_physical'][good]>=0),'Physical auxiliary nonnegativity')
            catalogue=meta['diagnostic_catalogue']
            codes=r[prefix+'_diagnostic_code']
            check(np.all((codes>=0)&(codes<len(catalogue))),'Every fit diagnostic code resolves')
            for row in np.flatnonzero(good):
                check(catalogue[codes[row]].get('success') is True,'Successful diagnostic record')
            expected_errors.extend((int(r['replicate_id'][row]),family,phase)
                                   for row in np.flatnonzero(~good))
        check(np.all(np.isin(r[family+'_LC_status'],[-2,-1,0,1])),'LC status codes')
        same(r[family+'_LC_status']==-2,~aux,'LC attempted exactly when auxiliary fit available')
        expected_errors.extend((int(r['replicate_id'][row]),family,'LC')
                               for row in np.flatnonzero(np.isin(r[family+'_LC_status'],[-1,0])))
        actual=meta['summaries'][family]
        check(actual['n_available']==int(valid.sum()),'Batch available count')
        check(actual['n_false_detections']==int((d==1).sum()),'Batch false-detection count')
        check(actual['n_failed_decisions']==int((~valid).sum()),'Batch unavailable count')
        cov=r[family+'_free_covariance_available']
        check(actual['n_successful_free_covariance_unavailable']==int((free&~cov).sum()),
              'Batch missing covariance count')
        check(actual['n_covariance_unavailable_but_decision_available']==int((valid&~cov).sum()),
              'Missing covariance with available decision retained')
        indices=np.flatnonzero(lc)
        if indices.size:
            selected=indices[np.unique(np.linspace(0,len(indices)-1,min(qr_per_batch,len(indices)),dtype=int))]
            for row in selected:
                expected=qr_lc(model['S'],model['b'],r[family+'_auxiliary_physical'][row],scenario['alpha'])
                actual=r[family+'_LC'][row]
                close(actual,expected,'Independent full-model QR calculation of saved LC')
                max_error=max(max_error,float(abs(actual-expected)));qr_checks+=1
    actual_errors=[(e['replicate_id'],e['family'],e['phase']) for e in meta['errors']]
    check(Counter(actual_errors)==Counter(expected_errors),'All failures retained with reasons')
    return r,meta,qr_checks,max_error


def check_merged(all_rows,metas,scenario):
    with np.load(ROOT/'replicates.npz',allow_pickle=False) as merged:
        catalogue=[json.loads(str(x)) for x in merged['diagnostic_catalogue_json']]
        same(merged['replicate_id'],np.arange(100000),'Merged IDs retain all original rows')
        check(str(merged['scenario_sha256'])==sha(ROOT/'scenario.json'),'Merged scenario hash')
        check(str(merged['runner_lock_sha256'])==sha(ROOT/'runner_lock.json'),'Merged source lock')
        same(merged['seed_table'],[[scenario['master_seed'],i] for i in range(100)],'Merged seeds')
        for key,expected in all_rows.items():
            if key.endswith('_diagnostic_code'):
                actual=merged[key]
                check(actual.shape==(100000,),'Global diagnostic code shape')
                for chunk,meta in enumerate(metas):
                    sl=slice(chunk*1000,(chunk+1)*1000)
                    for code in np.unique(expected[sl]):
                        ix=np.flatnonzero(expected[sl]==code)+chunk*1000
                        global_codes=np.unique(actual[ix])
                        check(len(global_codes)==1,'Consistent diagnostic remapping')
                        check(catalogue[int(global_codes[0])]==meta['diagnostic_catalogue'][int(code)],
                              'Diagnostic messages preserved during concatenation')
            else:
                same(merged[key],expected,'Merged row data preserved: '+key)


def portable_batches(scenario):
    """Restore local diagnostic code views from the merged, unfiltered rows."""
    with np.load(ROOT/'replicates.npz',allow_pickle=False) as saved:
        merged={key:saved[key] for key in saved.files}
    same(merged['replicate_id'],np.arange(100000),'Portable merged row IDs')
    check(str(merged['scenario_sha256'])==sha(ROOT/'scenario.json'),'Portable scenario hash')
    check(str(merged['runner_lock_sha256'])==sha(ROOT/'runner_lock.json'),'Portable source lock')
    same(merged['seed_table'],[[scenario['master_seed'],i] for i in range(100)],'Portable seed table')
    catalogue=[json.loads(str(x)) for x in merged['diagnostic_catalogue_json']]
    metas=[json.loads(str(x)) for x in merged['batch_metadata_json']]
    check(len(metas)==100,'Portable metadata for all 100 batches')
    constant_keys=('isotope_order','target_isotope','target_index','concentration_units',
                   'physical_coordinate_label','physical_units')
    row_keys=[key for key,value in merged.items() if value.ndim and value.shape[0]==100000]
    for chunk,meta in enumerate(metas):
        sidecar=ROOT/'batch_metadata'/f'batch_{chunk:03d}.json'
        if sidecar.exists():
            check(read(sidecar)==meta,'Portable sidecar metadata agrees with merged metadata')
        sl=slice(chunk*1000,(chunk+1)*1000)
        rows={key:merged[key][sl] for key in row_keys}
        rows.update({key:merged[key] for key in constant_keys})
        rows['seed']=merged['seed_table'][chunk]
        local_catalogue=meta['diagnostic_catalogue']
        local_lookup={json.dumps(entry,sort_keys=True,ensure_ascii=False):i
                      for i,entry in enumerate(local_catalogue)}
        for key in row_keys:
            if not key.endswith('_diagnostic_code'):
                continue
            source=rows[key];restored=np.empty_like(source)
            for code in np.unique(source):
                check(0<=code<len(catalogue),'Portable global diagnostic code resolves')
                entry=json.dumps(catalogue[int(code)],sort_keys=True,ensure_ascii=False)
                check(entry in local_lookup,'Portable message exists in original batch catalogue')
                restored[source==code]=local_lookup[entry]
            rows[key]=restored
        yield chunk,rows,meta


def check_summary(all_rows):
    with (ROOT/'summary.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    check(len(rows)==2,'Only two methods and the requested Xe133m test')
    results={}
    for family in FAMILIES:
        row=next(r for r in rows if r['method_key']==family)
        check(row['isotope']=='Xe133m','Summary isotope')
        d=all_rows[family+'_decision'];valid=d>=0
        free=all_rows[family+'_free_status']==1;lc=all_rows[family+'_LC_status']==1
        cov=all_rows[family+'_free_covariance_available']
        n=len(d);nv=int(valid.sum());k=int((d==1).sum());nf=n-nv
        estimates=all_rows[family+'_concentrations'][:,3]
        thresholds=all_rows[family+'_LC']
        lo=0. if k==0 else float(beta.ppf(.025,k,nv-k+1))
        hi=1. if k==nv else float(beta.ppf(.975,k+1,nv-k))
        expected=dict(n_generated=n,n_valid=nv,n_false=k,frequency=k/nv,
            ci95_low=lo,ci95_high=hi,n_fail=nf,frequency_all_min=k/n,
            frequency_all_max=(k+nf)/n,n_available_estimate=int(free.sum()),
            mean_estimate_mBq_m3=float(np.mean(estimates[free])),
            sd_estimate_mBq_m3=float(np.std(estimates[free],ddof=1)),
            n_available_lc=int(lc.sum()),lc_mean_mBq_m3=float(np.mean(thresholds[lc])),
            lc_sd_mBq_m3=float(np.std(thresholds[lc],ddof=1)),
            lc_median_mBq_m3=float(np.median(thresholds[lc])),
            lc_q025_mBq_m3=float(np.quantile(thresholds[lc],.025)),
            lc_q975_mBq_m3=float(np.quantile(thresholds[lc],.975)),n_paired=nv,
            correlation_estimate_LC=float(np.corrcoef(estimates[valid],thresholds[valid])[0,1]),
            n_successful_free_covariance_unavailable=int((free&~cov).sum()),
            n_covariance_unavailable_decision=int((valid&~cov).sum()))
        for key,value in expected.items():
            close(float(row[key]),value,'Summary CSV field: '+family+'/'+key)
        results[family]=expected
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--partial',action='store_true',help='Check completed batches without requiring merged outputs')
    modes.add_argument('--from-replicates',action='store_true',help='Audit merged archive without original batch NPZ files')
    parser.add_argument('--qr-per-batch',type=int,default=21,help='Deterministically spaced valid rows per method and batch')
    args=parser.parse_args()
    check(args.qr_per_batch>=1,'At least one independent covariance check per method and batch')
    scenario,model=baseline()
    chunks=(list(range(100)) if args.from_replicates else
            sorted(int(p.stem.split('_')[1]) for p in (ROOT/'points').glob('batch_*.json')))
    check(bool(chunks),'At least one completed batch')
    check(len(set(chunks))==len(chunks) and all(0<=i<100 for i in chunks),'Unique planned batch IDs')
    if not args.partial:
        check(chunks==list(range(100)),'All 100 planned batches are required')
    row_pieces={};metas=[];qr_checks=0;max_error=0.
    iterator=(portable_batches(scenario) if args.from_replicates else
              ((chunk,None,None) for chunk in chunks))
    for chunk,loaded,metadata in iterator:
        arrays,meta,nqr,error=verify_batch(chunk,scenario,model,args.qr_per_batch,loaded,metadata)
        metas.append(meta);qr_checks+=nqr;max_error=max(max_error,error)
        for key,value in arrays.items():
            if value.ndim and value.shape[0]==1000:
                row_pieces.setdefault(key,[]).append(value)
    all_rows={key:np.concatenate(values) for key,values in row_pieces.items()}
    output=dict(status='PASS',mode='portable' if args.from_replicates else 'partial' if args.partial else 'complete',
        n_batches=len(chunks),n_rows=len(all_rows['replicate_id']),
        independent_QR_LC_checks=qr_checks,max_absolute_QR_LC_difference_mBq_m3=max_error,
        no_new_observations_generated=True,no_parameter_fits_performed=True,
        no_LD_or_MDC_calculated=True,scenario_sha256=sha(ROOT/'scenario.json'),
        runner_lock_sha256=sha(ROOT/'runner_lock.json'),
        verifier_sha256=sha(__file__),
        checks=['Frozen source/scenario/input hashes','Original counts, calibration, coordinates and truth',
                'All planned row IDs and raw observations','Saved optimizer status and failure reasons',
                'Own LC and strict detection decisions','Missing free covariance does not exclude a decision',
                'Independent scaled-QR spot checks of original full-model LC formula'])
    if not args.partial:
        if not args.from_replicates:
            check_merged(all_rows,metas,scenario)
            output['checks']+=['Merged NPZ exact preservation and diagnostic remapping']
        else:
            output['checks']+=['Portable merged rows, metadata and diagnostic remapping']
            output['original_batch_npz_byte_comparison']=False
        output['summary']=check_summary(all_rows)
        output['checks']+=['Counts, exact Clopper-Pearson intervals and failure bounds',
            'Moments, quantiles and paired correlation from saved full data']
        output['replicates_sha256']=sha(ROOT/'replicates.npz')
        output['summary_sha256']=sha(ROOT/'summary.csv')
    name='audit_portable.json' if args.from_replicates else 'audit_partial.json' if args.partial else 'audit_result.json'
    (ROOT/name).write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
