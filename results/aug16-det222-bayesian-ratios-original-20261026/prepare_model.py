"""Build the detector-222 model and save its two original Poisson estimates.

Uses the unchanged XeMat parser, calibration builder and solvers. User-supplied
fixed efficiencies enter a process-local copy of the calibration registries;
constants.py, Rs_222 and Rg_222 remain unchanged. No LC/LD/MDC or simulated counts.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE / 'kernel_snapshot'))
import numpy as np
import scipy
import constants
import xemat.calibration as calibration
from process_measurement import _constants_input
from xemat.activities import compute_activities
from xemat.diagnostics import compute_fit_quality
from xemat.io import jsonable
from xemat.solvers import fit

COMMIT = '69ff5a62ccaaddeb6372f7f003109ce5ff6cf333'
INPUTS = dict(sample_file='SAMPLE20250816192330864.PHD',
              background_file='GASBK20250816032815.phd',
              detector_file='DETBK2025061021311249.PHD')
ISOTOPES = ['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']
MATRIX_KEYS = ('R_c0', 'R_g0', 'R_c', 'R_g1', 'R_g2', 'C_s_c', 'C_s_g', 'C_g_g',
               'B', 'F', 'H_g1', 'H_g2', 'H', 'Gamma', 'G_p', 'G_s', 'I_c', 'I_g',
               'V0', 'delta_I', 'delta_V0', 'kappa_s', 'kappa_g')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    config = json.loads((ROOT/'calibration_config.json').read_text())
    assert config['detector_id'] == 222
    assert config['isotope_order'] == ISOTOPES[1:]
    assert config['branching'] == {name: info['br'] for name,info in constants.ROI_INFO.items()}
    for key in ('sample_efficiencies','gasbk_efficiencies'):
        eps = np.asarray(config[key])
        assert eps.shape == (4,) and np.all(np.isfinite(eps)) and np.all((eps > 0) & (eps <= 1))
    original_s, original_g = calibration.FIXED_EFFICIENCIES_SAMPLE, calibration.FIXED_EFFICIENCIES_GASBK
    try:
        calibration.FIXED_EFFICIENCIES_SAMPLE = {**original_s, 222: np.array(config['sample_efficiencies'])}
        calibration.FIXED_EFFICIENCIES_GASBK = {**original_g, 222: np.array(config['gasbk_efficiencies'])}
        model, extras, metadata = _constants_input(
            tuple(str(SOURCE/'inputs'/name) for name in INPUTS.values()), True, 'legacy', None, True)
    finally:
        calibration.FIXED_EFFICIENCIES_SAMPLE, calibration.FIXED_EFFICIENCIES_GASBK = original_s, original_g
    np.testing.assert_array_equal(extras['R_c0'], constants.Rs_222)
    np.testing.assert_array_equal(extras['R_g0'], constants.Rg_222)
    assert model.M.shape == (21,17) and model.physical_matrix.min() >= 0
    assert metadata['input_detector_id'] == 222
    metadata['efficiency_source'] = 'calibration_config.json: user fixed epsilon for detector 222, 2026-10-26'
    metadata['calibration_source'] = 'Unmodified constants.py Rs_222/Rg_222 and ROI_INFO; fixed epsilon from calibration_config.json; original IMS times and volume'
    metadata['calibration_uncertainties_included_in_analysis'] = False
    arrays = dict(M=model.M, x=model.x, b=model.b, T_V=model.T_c,
                  physical_matrix=model.physical_matrix, isotope_order=np.array(ISOTOPES),
                  concentration_units=np.array('mBq/m3'), covariance_units=np.array('(mBq/m3)^2'))
    calibration_matrices = {k: extras[k] for k in MATRIX_KEYS if k in extras}
    arrays.update({'calibration_'+k: np.asarray(v) for k,v in calibration_matrices.items()})
    diagnostics = {}
    # Only two point estimates needed for the Bayesian comparison. Direct use
    # of the existing solver avoids invoking the ordinary LC/MDC workflow.
    for method in ['poisson', 'poisson_nonnegative']:
        solution = fit(model, method=method)  # unchanged max_iter=2000, tol=1e-10
        if not solution.success:
            raise RuntimeError(f'{method}: {solution.message}')
        covariance_beta = None if solution.covariance is None else solution.covariance[:5,:5]
        activity = compute_activities(solution.theta[:5], covariance_beta,
            H=extras['H'], Gamma=extras['Gamma'], G_p=extras['G_p'], G_s=extras['G_s'],
            V0=extras['V0'], T_c=model.T_c, delta_I=np.zeros(5), delta_V0=0.)
        quality = compute_fit_quality(model, solution)
        result = dict(model=model, fit=solution, activities=activity, fit_quality=quality,
            detection=None, detection_not_computed='Not part of this Bayesian analysis',
            regression_method_used=method, use_fixed_roi=True, use_fixed_efficiencies=True,
            metadata=metadata, calibration_matrices=calibration_matrices,
            parameter_names=extras['parameter_names'], isotope_names=ISOTOPES,
            warnings=list(extras.get('warnings',[]))+list(solution.warnings),
            calibration_config=config, input_files=INPUTS,
            statistical_uncertainty_only=True, solver_settings=dict(max_iter=2000,tol=1e-10))
        save(SOURCE/f'result_{method}.json',result)
        arrays[method+'_theta'] = solution.theta
        arrays[method+'_physical_parameters'] = solution.physical_parameters
        arrays[method+'_mu'] = solution.mu
        arrays[method+'_concentrations_mBq_m3'] = 1000*activity.a_c
        if solution.covariance is not None:
            arrays[method+'_full_covariance_theta'] = solution.covariance
            arrays[method+'_full_covariance_physical'] = solution.physical_covariance
        if activity.K_A_c_stat is not None:
            arrays[method+'_concentration_covariance_statistical'] = 1e6*activity.K_A_c_stat
        if method == 'poisson' and solution.physical_covariance is None:
            raise RuntimeError('Free ML statistical covariance is unavailable')
        diagnostics[method] = dict(success=solution.success, message=solution.message,
            iterations=solution.iterations, warnings=solution.warnings,
            concentration_mBq_m3=1000*activity.a_c[1:], fit_quality=quality)
        print(method, 'success',solution.success, 'concentrations mBq/m3',1000*activity.a_c[1:],flush=True)
    np.savez_compressed(SOURCE/'measurement_arrays.npz', **arrays)
    scenario = dict(production_commit=COMMIT,
        input_files={key: dict(file=name, sha256=sha(SOURCE/'inputs'/name)) for key,name in INPUTS.items()},
        kernel_sha256={str(p.relative_to(SOURCE/'kernel_snapshot')): sha(p) for p in sorted((SOURCE/'kernel_snapshot').rglob('*.py'))},
        fixed_roi=True, fixed_efficiencies=True, radon_mode='legacy', detector=222,
        response_model='Original constants.py Rs_222/Rg_222; no response modifications',
        metadata=metadata, fixed_roi_bounds=constants.FIXED_ROI_TABLE, branching=config['branching'],
        concentration_units='mBq/m3', isotope_order=ISOTOPES, calibration_config=config,
        no_observation_simulation=True, no_LC_LD_MDC_calculation=True,
        covariance_rescaled_by_reduced_chi2=False, statistical_uncertainty_only=True,
        baseline_methods=['poisson','poisson_nonnegative'], baseline_fits_new_for_this_measurement=True,
        software=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__))
    save(SOURCE/'scenario.json', scenario)
    save(ROOT/'baseline_diagnostics.json',diagnostics)
    manifest = dict(source_kind='Three user-supplied detector-222 PHD files and unchanged XeMat kernel',
        production_commit=COMMIT, reference_bayesian_implementation_commit='abbb795d3ca1cd87e5429032f2457f281e7c4fc6',
        calibration_config_sha256=sha(ROOT/'calibration_config.json'), preparation_script_sha256=sha(__file__),
        source_files={str(p.relative_to(SOURCE)): sha(p) for p in sorted(SOURCE.rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts})
    save(ROOT/'source_manifest.json', manifest)
    print('Model, two original estimates, metadata and source hashes saved.',flush=True)


if __name__ == '__main__':
    main()
