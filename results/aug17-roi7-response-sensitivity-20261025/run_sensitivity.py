"""Refit the August measurement with Rs[ROI7, Xe133]=0.35 or 0.36.

The saved 0.34 results are reused. A temporary registry belongs only to the
calibration module in this process: no constants.py is edited. Rg is copied
without changing any value. The usual constructor propagates the Rs change
through C_s_c, including the Xe133 daughter contribution in the Xe133m column.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'kernel_snapshot'))
import numpy as np
from process_measurement import process_prepared_measurement, METHOD_NAMES
from results_saver import format_results
import xemat.calibration as calibration
from xemat.diagnostics import poisson_deviance
from xemat.io import jsonable, save_json

VALUES = [('original_034', 0.34), ('test_035', 0.35), ('test_036', 0.36)]
METHODS = list(METHOD_NAMES)
LABELS = dict(wls='LS', nnls='NNLS', poisson='Poisson ML', poisson_nonnegative='Nonnegative Poisson ML')
ROIS = [1, 2, 3, 5, 6, 7, 8]
ISOTOPES = ['Rn222','Xe135','Xe133','Xe131m','Xe133m']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    source=read(ROOT/'baseline/scenario.json')
    for name,digest in source['kernel_sha256'].items():
        assert sha(ROOT/'kernel_snapshot'/name)==digest,name
    for item in source['input_files'].values():
        assert sha(ROOT/'inputs'/item['file'])==item['sha256']
    saved_registry=calibration.DETECTOR_RESPONSE_MATRICES
    original=read(ROOT/'baseline/result_poisson.json')
    base_cal={k:np.asarray(v) for k,v in original['calibration_matrices'].items()}
    base_M=np.asarray(original['model']['M'])
    assert base_cal['R_c0'][5,2]==0.34
    summary, roi_rows, activity_rows, checked=[],[],[],[]
    try:
        for variant,value in VALUES:
            folder=ROOT/variant;folder.mkdir(exist_ok=True)
            registry={key:(np.array(pair[0],copy=True),np.array(pair[1],copy=True))
                      for key,pair in saved_registry.items()}
            registry[221][0][5,2]=value
            calibration.DETECTOR_RESPONSE_MATRICES=registry
            for method in METHODS:
                if variant=='original_034':
                    result=read(ROOT/'baseline'/f'result_{method}.json')
                else:
                    result=process_prepared_measurement(
                        **{key:str(ROOT/'inputs'/item['file']) for key,item in source['input_files'].items()},
                        method=method,use_fixed_roi=True,use_fixed_efficiencies=True,radon_mode='legacy')
                    result['metadata']['temporary_response_override']={
                        'matrix':'Rs_221','zero_based_index':[5,2],'roi':7,'isotope':'Xe133',
                        'old_value':0.34,'new_value':value,'delta':value-0.34,
                        'memory_response_changed':False,'constants_file_modified':False}
                    result['metadata']['calibration_source']+='; temporary in-memory Rs_221[5,2]='+str(value)
                    (folder/f'result_{method}.txt').write_text(format_results(result),encoding='utf-8')
                    result=jsonable(result)
                save_json(result, folder/f'result_{method}.json')
                model=result['model'];fit=result['fit'];activity=result['activities'];detect=result['detection']
                assert fit['success']
                x=np.asarray(model['x']);mu=np.asarray(fit['mu']);M=np.asarray(model['M'])
                expected=base_M.copy();expected[5,:5]+=(value-0.34)*base_cal['C_s_c'][2,:]
                np.testing.assert_allclose(M,expected,rtol=0,atol=1e-15)
                for key in ('x','b','T_c'):
                    np.testing.assert_array_equal(model[key],original['model'][key])
                for key in ('R_g0','R_g1','R_g2','F','B','H','H_g1','H_g2','Gamma','G_p','G_s','C_s_c','C_s_g','C_g_g','I_c','I_g','V0','delta_I','delta_V0','kappa_s','kappa_g'):
                    np.testing.assert_array_equal(result['calibration_matrices'][key],base_cal[key])
                expected_R=base_cal['R_c0'].copy();expected_R[5,2]=value
                np.testing.assert_array_equal(result['calibration_matrices']['R_c0'],expected_R)
                d=np.array([poisson_deviance(np.array([xx]),np.array([mm])) for xx,mm in zip(x,mu)])
                q=result['fit_quality']
                if method.startswith('poisson'):
                    np.testing.assert_allclose(d.sum(),q['statistic'],rtol=1e-13)
                for i,(block,roi) in enumerate((block,roi) for block in ('SAMPLE','GASBK','DETBK') for roi in ROIS):
                    roi_rows.append(dict(variant=variant,Rs_ROI7_Xe133=value,method=LABELS[method],method_key=method,
                        block=block,roi=roi,x=float(x[i]),mu=float(mu[i]),residual=float(x[i]-mu[i]),
                        relative_residual_percent=float(100*(x[i]-mu[i])/mu[i]),deviance_contribution=float(d[i]),
                        signed_deviance_residual=float(np.sign(x[i]-mu[i])*np.sqrt(d[i]))))
                summary.append(dict(variant=variant,Rs_ROI7_Xe133=value,method=LABELS[method],method_key=method,
                    success=fit['success'],criterion=q['statistic_name'],statistic=q['statistic'],
                    nominal_dof=q['nominal_dof'],dof=q['degrees_of_freedom'],reduced_statistic=q['reduced_statistic'],
                    D_all_rows=float(d.sum()),D_SAMPLE=float(d[:7].sum()),D_GASBK=float(d[7:14].sum()),D_DETBK=float(d[14:].sum()),
                    sample_ROI7_x=float(x[5]),sample_ROI7_mu=float(mu[5]),sample_ROI7_residual=float(x[5]-mu[5]),
                    sample_ROI7_D=float(d[5]),solver_message=fit['message'],iterations=fit['iterations']))
                for i,name in enumerate(ISOTOPES[1:],1):
                    j=list(detect['component_indices']).index(i) if detect else None
                    activity_rows.append(dict(variant=variant,Rs_ROI7_Xe133=value,method=LABELS[method],method_key=method,
                        isotope=name,concentration_mBq_m3=1000*activity['a_c'][i],
                        sigma_stat_mBq_m3=None if activity['K_A_c_stat'] is None else 1000*np.sqrt(activity['K_A_c_stat'][i][i]),
                        sigma_total_mBq_m3=None if activity['std_A_c'] is None else 1000*activity['std_A_c'][i],
                        LC_mBq_m3=None if j is None else 1000*detect['L_C'][j],
                        MDC_mBq_m3=None if j is None else 1000*detect['MDC'][j],
                        detected=None if j is None else detect['detected'][j]))
                checked.append(dict(variant=variant,method=method,only_expected_fresh_M_row_changed=True,
                    all_memory_matrices_unchanged=True,x_b_TV_times_efficiencies_unchanged=True))
                print(variant,method,'criterion',q['reduced_statistic'],'ROI7res',x[5]-mu[5],
                      'ROI7D',d[5],'activities',1000*np.array(activity['a_c'][1:]),flush=True)
    finally:
        calibration.DETECTOR_RESPONSE_MATRICES=saved_registry
    write_csv(ROOT/'summary.csv',summary)
    write_csv(ROOT/'roi_diagnostics.csv',roi_rows)
    write_csv(ROOT/'concentrations.csv',activity_rows)
    save_json(dict(status='passed',checks=checked,no_new_observations=True,baseline_fits_reused=True,
                   memory_response_unchanged=True,constants_files_unchanged=True),ROOT/'construction_checks.json')
    save_json(dict(status='complete',source_scenario=source,values=dict(VALUES),
        changed_base_matrix='Rs_221',zero_based_index=[5,2],roi=7,isotope='Xe133',
        changed_M_entries=[[5,2],[5,4]],daughter_coefficient=base_cal['C_s_c'][2,4],
        memory_response_unchanged=True,code_or_constants_file_changes=False,
        no_monte_carlo=True,case_purpose='Sensitivity experiment, not a new adopted calibration',
        baseline_results_sha256={m:sha(ROOT/'baseline'/f'result_{m}.json') for m in METHODS},
        runner_sha256=sha(__file__)),ROOT/'scenario.json')


if __name__=='__main__':
    main()
