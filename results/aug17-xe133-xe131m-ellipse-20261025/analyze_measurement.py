"""Ordinary XeMat processing of the August spectra; no Monte Carlo draws.

All four methods use the unmodified GUI workflow. The saved covariance blocks
allow plotting the two free-estimator confidence ellipses without refitting.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'kernel_snapshot'))

import numpy as np
import scipy
from scipy.stats import chi2
from constants import ROI_INFO, FIXED_ROI_TABLE
from process_measurement import process_prepared_measurement, METHOD_NAMES
from results_saver import format_results
from xemat.io import jsonable, save_json

CODE_COMMIT = '69ff5a62ccaaddeb6372f7f003109ce5ff6cf333'
INPUTS = dict(sample_file='SAMPLE20250817072414190.PHD',
              background_file='GASBK20250816152938777.PHD',
              detector_file='DETBK20250610213044796.PHD')
LABELS = dict(wls='LS', nnls='NNLS', poisson='Poisson ML',
              poisson_nonnegative='Nonnegative Poisson ML')
PAIR = np.array([2, 3])
CONFIDENCE = 0.95


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, ellipses, quality = [], [], []
    arrays = dict(pair_isotope_order=np.array(['Xe133', 'Xe131m']),
                  isotope_order=np.array(['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']),
                  concentration_units=np.array('mBq/m3'), covariance_units=np.array('(mBq/m3)^2'),
                  confidence_level=np.array(CONFIDENCE), chi2_quantile=np.array(chi2.ppf(CONFIDENCE, 2)))
    previous_model = None
    for method in METHOD_NAMES:
        result = process_prepared_measurement(
            **{key: str(ROOT/'inputs'/name) for key, name in INPUTS.items()},
            method=method, use_fixed_roi=True, use_fixed_efficiencies=True, radon_mode='legacy')
        solution, activity, detection = result['fit'], result['activities'], result['detection']
        model, goodness = result['model'], result['fit_quality']
        assert solution.success and activity is not None
        assert result['metadata']['input_detector_id'] == 221
        assert list(result['isotope_names']) == list(arrays['isotope_order'])
        if previous_model is not None:
            for name in ('M', 'x', 'b', 'T_c'):
                np.testing.assert_array_equal(getattr(model, name), getattr(previous_model, name))
        previous_model = model
        save_json(result, ROOT/f'result_{method}.json')
        (ROOT/f'result_{method}.txt').write_text(format_results(result), encoding='utf-8')
        arrays[method+'_theta'] = solution.theta
        arrays[method+'_physical_parameters'] = solution.physical_parameters
        arrays[method+'_mu'] = solution.mu
        arrays[method+'_concentrations_mBq_m3'] = 1000*activity.a_c
        if solution.covariance is not None:
            arrays[method+'_full_covariance_theta'] = solution.covariance
            arrays[method+'_full_covariance_physical'] = solution.physical_covariance
        if activity.K_A_c_stat is not None:
            for label, covariance in [('statistical', activity.K_A_c_stat), ('systematic', activity.K_A_c_sys), ('total', activity.K_A_c)]:
                arrays[method+'_concentration_covariance_'+label] = 1e6*covariance
        d_indices = [] if detection is None else list(detection.component_indices)
        for i, isotope in enumerate(result['isotope_names']):
            if i == 0:
                continue  # The legacy radon coordinate is not an atmospheric Xe concentration.
            j = d_indices.index(i) if i in d_indices else None
            rows.append(dict(method=LABELS[method], method_key=method, isotope=str(isotope),
                concentration_mBq_m3=float(1000*activity.a_c[i]),
                sigma_stat_mBq_m3=None if activity.K_A_c_stat is None else float(1000*np.sqrt(activity.K_A_c_stat[i,i])),
                sigma_total_mBq_m3=None if activity.std_A_c is None else float(1000*activity.std_A_c[i]),
                LC_mBq_m3=None if j is None else float(1000*detection.L_C[j]),
                MDC_mBq_m3=None if j is None else float(1000*detection.MDC[j]),
                detected_from_free_estimate=None if j is None else bool(detection.detected[j]),
                mdc_converged=None if j is None else bool(detection.converged[j])))
        quality.append(dict(method=LABELS[method], method_key=method, **jsonable(goodness),
            success=bool(solution.success), solver_message=solution.message,
            solver_iterations=solution.iterations, workflow_warnings=jsonable(result['warnings'])))
        if method in ('wls', 'poisson'):
            if activity.K_A_c_stat is None:
                raise RuntimeError(f'{method}: statistical covariance unavailable; no ellipse can be drawn')
            center = 1000*activity.a_c[PAIR]
            arrays[method+'_pair_center'] = center
            for label, covariance in [('statistical', activity.K_A_c_stat), ('total', activity.K_A_c)]:
                block = 1e6*covariance[np.ix_(PAIR,PAIR)]
                np.linalg.cholesky(block)
                values, vectors = np.linalg.eigh(block)
                order = np.argsort(values)[::-1]
                radii = np.sqrt(arrays['chi2_quantile']*values[order])
                direction = vectors[:,order[0]]
                record = dict(method=LABELS[method],method_key=method,uncertainty=label,
                    center_mBq_m3=center.tolist(),covariance_mBq2_m6=block.tolist(),
                    correlation=float(block[0,1]/np.sqrt(block[0,0]*block[1,1])),
                    semi_axes_mBq_m3=radii.tolist(),
                    major_axis_angle_degrees=float(np.degrees(np.arctan2(direction[1],direction[0]))),
                    confidence_level=CONFIDENCE,chi2_df=2,chi2_quantile=float(arrays['chi2_quantile']))
                ellipses.append(record)
                arrays[method+'_pair_covariance_'+label] = block
        print(method, 'success', solution.success, 'concentrations mBq/m3', 1000*activity.a_c[1:],
              'reduced statistic', goodness.reduced_statistic, flush=True)
    arrays.update(M=model.M, x=model.x, b=model.b, T_V=model.T_c,
                  physical_matrix=model.physical_matrix)
    for key, value in result['calibration_matrices'].items():
        arrays['calibration_'+key] = np.asarray(value)
    np.savez_compressed(ROOT/'measurement_arrays.npz', **arrays)
    csv_write(ROOT/'concentrations.csv', rows)
    csv_write(ROOT/'fit_quality.csv', [{k:v for k,v in row.items() if k not in ('warnings','workflow_warnings')} for row in quality])
    save_json(ellipses, ROOT/'ellipses.json')
    save_json(quality, ROOT/'fit_diagnostics.json')
    scenario = dict(production_commit=CODE_COMMIT,
        input_files={key:dict(file=name,sha256=sha(ROOT/'inputs'/name)) for key,name in INPUTS.items()},
        kernel_sha256={str(p.relative_to(ROOT/'kernel_snapshot')):sha(p)
                       for p in sorted((ROOT/'kernel_snapshot').rglob('*.py'))},
        fixed_roi=True, fixed_efficiencies=True, radon_mode='legacy', detector=221,
        response_model='Original detector 221 Rs/Rg; no equal-response replacement',
        metadata=jsonable(result['metadata']), fixed_roi_bounds=FIXED_ROI_TABLE,
        branching={name:info['br'] for name,info in ROI_INFO.items()},
        concentration_units='mBq/m3', isotope_order=arrays['isotope_order'].tolist(),
        pair_isotope_order=arrays['pair_isotope_order'].tolist(),
        ellipse_description='Approximate joint 95% normal/Wald region for the two free concentrations; full nuisance covariance propagated; statistical and configured total covariance saved separately',
        no_monte_carlo=True, covariance_rescaled_by_reduced_chi2=False,
        software=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__))
    save_json(scenario, ROOT/'scenario.json')
    print('Ellipse summaries:', json.dumps(ellipses,ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
