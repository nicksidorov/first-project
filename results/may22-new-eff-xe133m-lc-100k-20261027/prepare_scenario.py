"""Freeze the updated May-22 Xe133m null mixture; no LC/LD/MDC calculations.

One joint nonnegative Poisson fit to the real data defines the plug-in truth.
Only fresh atmospheric Xe133m (physical z[4]) is then set to zero. Other fresh
components, all memory and the seven jointly estimated DETBK means are retained.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'kernel_snapshot'))
import numpy as np
import scipy
from constants import FIXED_ROI_TABLE,ROI_INFO,HALF_LIFE_DAYS,Rs_221,Rg_221
from process_measurement import _constants_input,_read_spectra
from xemat.io import jsonable
from xemat.solvers import fit
from xemat.diagnostics import compute_fit_quality

COMMIT='69ff5a62ccaaddeb6372f7f003109ce5ff6cf333'
ISOTOPES=['Rn222','Xe135','Xe133','Xe131m','Xe133m']
ROIS=[1,2,3,5,6,7,8]
INPUTS=dict(SAMPLE='SAMPLE20260522073631203.PHD',GASBK='GASBK20260521152531814.PHD',DETBK='DETBK20250610213044796.PHD')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,data):
    path.write_text(json.dumps(jsonable(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    if (ROOT/'scenario.json').exists():
        raise FileExistsError('Scenario already frozen; do not overwrite it')
    paths=[str(ROOT/'inputs'/name) for name in INPUTS.values()]
    model,extras,metadata=_constants_input(paths,True,'legacy',None,True)
    parsed=_read_spectra(paths)
    assert metadata['input_detector_id']==221
    assert model.M.shape==(21,17) and np.all(model.b==0)
    for key in ['sample_efficiencies','gasbk_efficiencies']:
        np.testing.assert_array_equal(metadata[key],[.241169,.367784,.233937,.233937])
    np.testing.assert_array_equal(extras['R_c0'],Rs_221)
    np.testing.assert_array_equal(extras['R_g0'],Rg_221)
    np.testing.assert_array_equal(model.x[14:],[522,854,494,66,64,89,97])
    S=model.physical_matrix
    assert np.all(S>=0) and np.linalg.matrix_rank(S/np.linalg.norm(S,axis=0))==17
    reference=fit(model,method='poisson_nonnegative')
    if not reference.success:
        raise RuntimeError(reference.message)
    z_reference=reference.physical_parameters.copy()
    assert np.all(np.isfinite(z_reference)) and np.all(z_reference>=0)
    z_truth=z_reference.copy();z_truth[4]=0.
    unchanged=[i for i in range(17) if i!=4]
    np.testing.assert_array_equal(z_reference[unchanged],z_truth[unchanged])
    mu0=model.b+S@z_truth
    assert np.all(mu0>0)
    components=dict(fresh_radon=S[:,0]*z_truth[0],fresh_xenon=S[:,1:5]@z_truth[1:5],
        memory=S[:,5:10]@z_truth[5:10],detector_background=S[:,10:]@z_truth[10:],fixed_offset=model.b)
    np.testing.assert_allclose(mu0,sum(components.values()),rtol=1e-14,atol=1e-12)
    names=['a_V_aux:Rn222']+['a_V:'+s for s in ISOTOPES[1:]]+['gamma:'+s for s in ISOTOPES]+[f'zeta:ROI{r}' for r in ROIS]
    units=['Bq/m3 (auxiliary, not atmospheric radon)']+['Bq/m3']*4+['reference GASBK expected counts']*5+['DETBK expected counts']*7
    obs=[f'{block}:ROI{r}' for block in INPUTS for r in ROIS]
    arrays=dict(M=model.M,x=model.x.astype(np.int64),b=model.b,T_V=model.T_V,S=S,
        z_reference=z_reference,z_truth=z_truth,mu0=mu0,theta_reference=reference.theta,
        mu_reference=reference.mu,physical_parameter_names=np.array(names),physical_parameter_units=np.array(units),
        observation_names=np.array(obs),roi_order=np.array(ROIS),isotope_order=np.array(ISOTOPES[1:]),
        concentration_units=np.array('mBq/m3'))
    arrays.update({'mu0_'+k:v for k,v in components.items()})
    for k,v in extras.items():
        if isinstance(v,np.ndarray):arrays['calibration_'+k]=v
    for key,data in parsed.items():
        for field in ['histogram','g_energy','b_energy','g_spectrum','b_spectrum']:
            arrays[key+'_'+field]=np.asarray(data[field])
    np.savez_compressed(ROOT/'scenario_arrays.npz',**arrays)
    save(ROOT/'source_fit.json',dict(model=model,fit=reference,fit_quality=compute_fit_quality(model,reference),
        metadata=metadata,calibration_matrices=extras,detection=None,no_detection_calculation=True))
    kernel_hashes={str(p.relative_to(ROOT/'kernel_snapshot')):sha(p) for p in sorted((ROOT/'kernel_snapshot').rglob('*.py'))}
    inputs={role:dict(file=name,path='inputs/'+name,sha256=sha(ROOT/'inputs'/name)) for role,name in INPUTS.items()}
    scenario=dict(scenario_id='may22_2026_detector221_fresh_Xe133m_null_updated_efficiencies',created_date='2026-10-27',
        n_generated=100000,n_batches=100,batch_size=1000,master_seed=2026102701,target_index=4,target_isotope='Xe133m',alpha=.05,
        seed_rule='numpy.random.default_rng([master_seed,chunk]).poisson(mu0,size=(1000,21)); chunk=0..99',
        bit_generator='PCG64',methods=['wls','poisson'],auxiliary_methods={'wls':'nnls','poisson':'poisson_nonnegative'},
        input_files=inputs,metadata=metadata,detector=221,fixed_roi=True,fixed_efficiencies=True,
        fixed_roi_bounds=FIXED_ROI_TABLE,roi_order=ROIS,
        sample_efficiencies=metadata['sample_efficiencies'],gasbk_efficiencies=metadata['gasbk_efficiencies'],
        branching={name:info['br'] for name,info in ROI_INFO.items()},half_life_days=HALF_LIFE_DAYS,radon_mode='legacy',radon_efficiency=.3,
        physical_coordinates='[a_V; gamma; zeta]',parameter_names=names,parameter_units=units,
        concentration_order=ISOTOPES[1:],concentration_units='mBq/m3',observation_names=obs,
        model=jsonable(model),z_reference=z_reference,z_truth=z_truth,mu0=mu0,
        truth_construction='One current nonnegative Poisson fit of all real SAMPLE/GASBK/DETBK. Set ONLY fresh physical Xe133m z[4]=0; retain all 16 other coordinates, including other fresh Xe, Rn, memory and fitted DETBK. No null refit.',
        source_fit_count=1,source_fit_success=reference.success,source_fit_message=reference.message,
        source_fit_iterations=reference.iterations,source_fit_warnings=reference.warnings,
        unchanged_truth_indices=unchanged,fresh_zeroed_indices=[4],
        generator='Independent Poisson for all21 counts; each observation vector is used by both estimator families',
        weights='W=diag(1/where(x==0,1,x)) from each realization for both WLS and NNLS; original zeros retained. Poisson uses raw counts.',
        solver_settings=dict(entrypoint='xemat.solvers.fit',max_iter=2000,tol=1e-10,overrides=False),
        lc_rule='One corresponding auxiliary nonnegative fit per family; zero its fresh z[4]; mu_null=b+S@z_aux_null; K=model_covariance(S,mu_null); LC=norm.ppf(.95)*sqrt(K[4,4]); full17 columns retained; detect iff free z[4]>LC.',
        no_null_refit=True,no_threshold_multiplier=True,threshold_multiplier=1.,no_mdc_or_ld_calculation=True,no_power_curves=True,
        systematics=False,plugin_truth_uncertainty_resampled=False,
        missing_policy='No row exclusion or replacement; failed decisions unknown, not nondetections. Missing free covariance alone does not invalidate estimate+LC.',
        production_commit=COMMIT,kernel_sha256=kernel_hashes,preparation_script_sha256=sha(__file__),
        scenario_arrays_sha256=sha(ROOT/'scenario_arrays.npz'),source_fit_sha256=sha(ROOT/'source_fit.json'),
        software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    save(ROOT/'scenario.json',scenario)
    save(ROOT/'scenario.lock.json',{k:sha(ROOT/k) for k in ['scenario.json','scenario_arrays.npz','source_fit.json','prepare_scenario.py']})
    print(json.dumps(jsonable(dict(source_fit=reference.message,reference_fresh_mBq_m3=1000*z_reference[:5],
        null_fresh_mBq_m3=1000*z_truth[:5],mu0=mu0,observed=model.x,metadata=metadata)),indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
