"""Independent audit of saved sensitivity fits; no fitting or simulation."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
METHODS = ('wls','nnls','poisson','poisson_nonnegative')
ROIS = (1,2,3,5,6,7,8)
ISOTOPES = ('Rn222','Xe135','Xe133','Xe131m','Xe133m')
BLOCKS = ('SAMPLE','GASBK','DETBK')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, atol=2e-9, rtol=2e-9):
    assert_allclose(actual, expected, atol=atol, rtol=rtol)


def independent_deviance(x, mu):
    # Extended precision reduces cancellation without using XeMat's routine.
    x=np.asarray(x,dtype=np.longdouble);mu=np.asarray(mu,dtype=np.longdouble)
    assert np.all(mu>=0) and np.all(mu[x>0]>0)
    terms=2*mu
    pos=x>0
    terms[pos]=2*(x[pos]*np.log(x[pos]/mu[pos])+mu[pos]-x[pos])
    return np.asarray(terms,dtype=float)


def covariance_qr(S, mu):
    A=S/np.sqrt(mu)[:,None]
    scales=np.linalg.norm(A,axis=0)
    _,R=np.linalg.qr(A/scales)
    inverse=np.linalg.solve(R,np.eye(S.shape[1]))
    return (inverse@inverse.T)/scales[:,None]/scales[None,:]


def main():
    scenario=read(ROOT/'scenario.json'); source=scenario['source_scenario']
    assert scenario['status']=='complete' and scenario['no_monte_carlo']
    assert scenario['values']==dict(original_034=.34,test_035=.35,test_036=.36)
    assert scenario['changed_base_matrix']=='Rs_221' and scenario['zero_based_index']==[5,2]
    assert scenario['changed_M_entries']==[[5,2],[5,4]]
    assert not scenario['code_or_constants_file_changes']
    assert scenario['memory_response_unchanged']
    assert sha(ROOT/'run_sensitivity.py')==scenario['runner_sha256']
    assert source==read(ROOT/'baseline/scenario.json')
    for name,digest in source['kernel_sha256'].items():
        assert sha(ROOT/'kernel_snapshot'/name)==digest,name
    production=Path('/workspace/first-project')
    production_checked=(production/'constants.py').is_file()
    if production_checked:
        for name,digest in source['kernel_sha256'].items():
            assert sha(production/name)==digest,name
    for item in source['input_files'].values():
        assert sha(ROOT/'inputs'/item['file'])==item['sha256']
    baseline={m:read(ROOT/'baseline'/f'result_{m}.json') for m in METHODS}
    for m in METHODS:
        assert sha(ROOT/'baseline'/f'result_{m}.json')==scenario['baseline_results_sha256'][m]
        assert read(ROOT/'original_034'/f'result_{m}.json')==baseline[m]
    construction=read(ROOT/'construction_checks.json')
    assert construction['status']=='passed' and len(construction['checks'])==12
    assert all(construction[k] for k in ('no_new_observations','baseline_fits_reused','memory_response_unchanged','constants_files_unchanged'))
    summaries={(r['variant'],r['method_key']):r for r in rows(ROOT/'summary.csv')}
    roi_rows={(r['variant'],r['method_key'],r['block'],int(r['roi'])):r for r in rows(ROOT/'roi_diagnostics.csv')}
    activities={(r['variant'],r['method_key'],r['isotope']):r for r in rows(ROOT/'concentrations.csv')}
    assert len(summaries)==12 and len(roi_rows)==252 and len(activities)==48
    base=baseline['poisson'];base_M=np.asarray(base['model']['M'])
    calibration={k:np.asarray(v) for k,v in base['calibration_matrices'].items()}
    assert calibration['R_c0'][5,2]==.34
    close(scenario['daughter_coefficient'],calibration['C_s_c'][2,4])
    audits=[];threshold_checks=[]
    files=['scenario.json','construction_checks.json','run_sensitivity.py','summary.csv','roi_diagnostics.csv','concentrations.csv','baseline/scenario.json']
    for variant,value in scenario['values'].items():
        results={m:read(ROOT/variant/f'result_{m}.json') for m in METHODS}
        for method,result in results.items():
            filename=f'{variant}/result_{method}.json';files.append(filename)
            fit,model,activity,quality=result['fit'],result['model'],result['activities'],result['fit_quality']
            assert fit['success'] and fit['rank']==17
            assert result['use_fixed_roi'] and result['use_fixed_efficiencies']
            assert result['metadata']['input_detector_id']==221
            expected_R=calibration['R_c0'].copy();expected_R[5,2]=value
            assert_array_equal(result['calibration_matrices']['R_c0'],expected_R)
            close(result['calibration_matrices']['R_c'],expected_R@calibration['C_s_c'],atol=1e-15,rtol=0)
            for name,old in calibration.items():
                if name not in ('R_c0','R_c'):
                    assert_array_equal(result['calibration_matrices'][name],old,name)
            M=np.asarray(model['M']);x=np.asarray(model['x']);b=np.asarray(model['b']);T=np.asarray(model['T_c'])
            expected=base_M.copy();expected[5,:5]+=(value-.34)*calibration['C_s_c'][2,:]
            close(M,expected,atol=1e-15,rtol=0)
            if value==.34:
                assert_array_equal(M,base_M)
            else:
                assert_array_equal(np.argwhere(M!=base_M),[[5,2],[5,4]])
            for name in ('x','b','T_c'):
                assert_array_equal(model[name],base['model'][name])
            for name,item in base['metadata'].items():
                if name!='calibration_source':
                    assert result['metadata'][name]==item,name
            theta=np.asarray(fit['theta']);physical=np.asarray(fit['physical_parameters']);mu=np.asarray(fit['mu'])
            Q=np.eye(17);Q[:5,:5]=T
            S=np.linalg.solve(Q.T,M.T).T
            close(M@theta+b,mu);close(Q@theta,physical)
            close(activity['a_c'],physical[:5])
            if method in ('nnls','poisson_nonnegative'):
                assert np.min(physical)>=-1e-10
                assert fit['covariance'] is None
            else:
                K=covariance_qr(S,np.where(x==0,1,x) if method=='wls' else mu)
                close(fit['physical_covariance'],K)
                close(activity['K_A_c_stat'],K[:5,:5])
            d=independent_deviance(x,mu)
            expected_stat=float(np.sum((x-mu)**2/np.where(x==0,1,x))) if method in ('wls','nnls') else float(d.sum())
            close(expected_stat,quality['statistic'])
            assert quality['degrees_of_freedom']==4
            close(expected_stat/4,quality['reduced_statistic'])
            summary=summaries[variant,method]
            assert summary['success']=='True'
            assert int(summary['dof'])==4
            assert (summary['nominal_dof']=='True')==(method in ('nnls','poisson_nonnegative'))
            expected_summary=dict(Rs_ROI7_Xe133=value,statistic=expected_stat,reduced_statistic=expected_stat/4,
                D_all_rows=d.sum(),D_SAMPLE=d[:7].sum(),D_GASBK=d[7:14].sum(),D_DETBK=d[14:].sum(),
                sample_ROI7_x=x[5],sample_ROI7_mu=mu[5],sample_ROI7_residual=x[5]-mu[5],sample_ROI7_D=d[5])
            for field,expected_value in expected_summary.items():close(float(summary[field]),expected_value)
            for j,(block,roi) in enumerate((block,roi) for block in BLOCKS for roi in ROIS):
                record=roi_rows[variant,method,block,roi]
                expected_roi=dict(x=x[j],mu=mu[j],residual=x[j]-mu[j],relative_residual_percent=100*(x[j]-mu[j])/mu[j],
                    deviance_contribution=d[j],signed_deviance_residual=np.sign(x[j]-mu[j])*np.sqrt(max(d[j],0)))
                for field,expected_value in expected_roi.items():close(float(record[field]),expected_value,atol=2e-8)
            det=result['detection'];assert det is not None and all(det['converged'])
            assert det['component_indices']==[1,2,3,4]
            free='poisson' if method.startswith('poisson') else 'wls'
            positive='poisson_nonnegative' if method.startswith('poisson') else 'nnls'
            free_values=np.asarray(results[free]['fit']['physical_parameters'])
            positive_values=np.asarray(results[positive]['fit']['physical_parameters'])
            close(det['free_activity'],free_values[1:5]);close(det['positive_activity'],positive_values[1:5])
            for i,isotope in enumerate(ISOTOPES[1:],1):
                j=i-1;record=activities[variant,method,isotope]
                close(float(record['concentration_mBq_m3']),1000*physical[i])
                if activity['K_A_c_stat'] is None:
                    assert record['sigma_stat_mBq_m3']==record['sigma_total_mBq_m3']==''
                else:
                    close(float(record['sigma_stat_mBq_m3']),1000*np.sqrt(activity['K_A_c_stat'][i][i]))
                    close(float(record['sigma_total_mBq_m3']),1000*activity['std_A_c'][i])
                for field,source_field in [('LC_mBq_m3','L_C'),('MDC_mBq_m3','MDC')]:
                    close(float(record[field]),1000*det[source_field][j])
                assert (record['detected']=='True')==det['detected'][j]==(free_values[i]>det['L_C'][j])
                if method==free:
                    null=S@positive_values+b-positive_values[i]*S[:,i]
                    assert np.min(null)>0
                    close(det['null_spectra'][j],null)
                    K0=covariance_qr(S,null)
                    close(det['null_covariances'][j],K0)
                    sigma0=np.sqrt(K0[i,i]);lc=norm.ppf(1-det['alpha'])*sigma0
                    close(det['sigma0'][j],sigma0);close(det['L_C'][j],lc)
                    md=det['MDC'][j];KL=covariance_qr(S,null+md*S[:,i])
                    close(det['sigma_at_MDC'][j],np.sqrt(KL[i,i]))
                    close(md,lc+norm.ppf(1-det['beta_ii'])*np.sqrt(KL[i,i]),atol=1e-11,rtol=2e-8)
                    threshold_checks.append(dict(variant=variant,method=method,isotope=isotope,own_null_model_checked=True,
                        LC_mBq_m3=1000*lc,MDC_mBq_m3=1000*md))
            audits.append(dict(variant=variant,method=method,success=True,Rs_ROI7_Xe133=value,
                statistic=expected_stat,reduced_statistic=expected_stat/4,ROI7_deviance=float(d[5]),
                ROI7_residual=float(x[5]-mu[5]),only_expected_matrix_entries_changed=True))
    for m in METHODS:files.append(f'baseline/result_{m}.json')
    verification=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(__file__),no_fits=True,no_simulation=True,all_twelve_results_checked=True,
        all_observations_unchanged=True,all_memory_matrices_unchanged=True,source_kernel_hashes_unchanged=True,
        local_production_kernel_checked_unchanged=production_checked,
        baseline_results_reused_exactly=True,n_summary_rows=12,n_roi_rows=252,n_concentration_rows=48,
        n_independent_QR_null_and_MDC_checks=len(threshold_checks),methods=audits,threshold_checks=threshold_checks,
        files_sha256={name:sha(ROOT/name) for name in files})
    (ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('methods','threshold_checks','files_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
