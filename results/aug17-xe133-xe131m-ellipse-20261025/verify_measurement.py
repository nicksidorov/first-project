"""Read-only numerical audit of saved ordinary fits and confidence ellipses.

No fitting or simulation. Full 17-parameter covariance is independently
reconstructed by column-scaled QR, before selecting the two Xe coordinates.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import xlogy
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
METHODS = ('wls', 'nnls', 'poisson', 'poisson_nonnegative')
PAIR = np.array([2, 3])


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def main():
    scenario = read('scenario.json')
    assert scenario['detector'] == 221
    assert scenario['fixed_roi'] and scenario['fixed_efficiencies']
    assert scenario['no_monte_carlo'] and not scenario['covariance_rescaled_by_reduced_chi2']
    assert scenario['pair_isotope_order'] == ['Xe133', 'Xe131m']
    assert scenario['concentration_units'] == 'mBq/m3'
    assert scenario['isotope_order'] == ['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']
    for item in scenario['input_files'].values():
        assert sha(ROOT/'inputs'/item['file']) == item['sha256']
    for filename, digest in scenario['kernel_sha256'].items():
        assert sha(ROOT/'kernel_snapshot'/filename) == digest
    arrays = dict(np.load(ROOT/'measurement_arrays.npz', allow_pickle=False))
    M, S, T, x, b = (arrays[k] for k in ('M','physical_matrix','T_V','x','b'))
    assert M.shape == S.shape == (21, 17)
    assert_array_equal(b, np.zeros(21))
    assert_array_equal(M[14:, -7:], np.eye(7))
    assert_array_equal(M[14:, :10], np.zeros((7,10)))
    assert_array_equal(x, x.astype(np.int64))
    Q = np.eye(17); Q[:5,:5] = T
    close(S @ Q, M)
    methods = {}
    results = {}
    for method in METHODS:
        result = read(f'result_{method}.json'); results[method] = result
        fit, activities = result['fit'], result['activities']
        assert fit['success'] and fit['rank'] == 17
        assert result['use_fixed_roi'] and result['use_fixed_efficiencies']
        assert result['metadata']['input_detector_id'] == 221
        assert all(item['id'] == 221 for item in result['metadata']['detector_info'].values())
        for key in ('M', 'x', 'b'):
            assert_array_equal(result['model'][key], arrays[key])
        close(result['model']['T_c'], T)
        theta = np.asarray(fit['theta']); physical = np.asarray(fit['physical_parameters'])
        mu = np.asarray(fit['mu']); concentration = np.asarray(activities['a_c'])
        close(M @ theta + b, mu)
        close(Q @ theta, physical)
        close(physical[:5], concentration)
        close(1000*concentration, arrays[method+'_concentrations_mBq_m3'])
        assert_array_equal(theta, arrays[method+'_theta'])
        if method in ('wls','nnls'):
            statistic = np.sum((x-mu)**2/np.where(x == 0, 1, x))
        else:
            assert np.all(mu >= 0) and np.all(mu[x>0] > 0)
            statistic = 2*np.sum(xlogy(x, x/np.where(mu>0, mu, 1)) - (x-mu))
        close(statistic, result['fit_quality']['statistic'])
        close(statistic/4, result['fit_quality']['reduced_statistic'])
        assert result['fit_quality']['degrees_of_freedom'] == 4
        if method not in ('wls','poisson'):
            assert fit['covariance'] is None and activities['K_A_c_stat'] is None
            assert np.min(physical) >= -1e-10
            methods[method] = dict(success=True, constrained_covariance_not_used=True)
            continue
        variance = np.where(x == 0, 1, x) if method == 'wls' else mu
        weighted = S/np.sqrt(variance)[:,None]
        scales = np.linalg.norm(weighted, axis=0)
        _, R = np.linalg.qr(weighted/scales)
        inverse_R = np.linalg.solve(R, np.eye(17))
        full_K = (inverse_R @ inverse_R.T)/scales[:,None]/scales[None,:]
        close(full_K, fit['physical_covariance'])
        close(Q @ np.asarray(fit['covariance']) @ Q.T, full_K)
        Kstat = np.asarray(activities['K_A_c_stat'])
        close(Kstat, full_K[:5,:5])
        calibration = result['calibration_matrices']
        H = np.asarray(calibration['H'])
        a = H @ theta[:5]
        sensitivity_cov = np.diag((a*np.asarray(calibration['delta_I']))**2)
        transfer = np.linalg.solve(H.T, T.T).T
        Ksys = calibration['delta_V0']**2*np.outer(concentration, concentration)
        Ksys += transfer @ sensitivity_cov @ transfer.T
        close(Ksys, activities['K_A_c_sys'])
        close(Kstat+Ksys, activities['K_A_c'])
        for label, covariance in (('statistical', Kstat), ('systematic', Ksys), ('total', Kstat+Ksys)):
            close(1e6*covariance, arrays[method+'_concentration_covariance_'+label])
        methods[method] = dict(success=True, covariance_method='independent scaled QR of all 17 physical columns',
            max_absolute_full_covariance_difference=float(np.max(np.abs(full_K-np.asarray(fit['physical_covariance'])))),
            statistical_and_systematic_propagation_checked=True)
    ellipses = read('ellipses.json')
    assert len(ellipses) == 4
    expected_quantile = chi2.ppf(.95, 2)
    checks = []
    for ellipse in ellipses:
        method, kind = ellipse['method_key'], ellipse['uncertainty']
        assert method in ('wls','poisson') and kind in ('statistical','total')
        assert ellipse['chi2_df'] == 2 and ellipse['confidence_level'] == .95
        close(ellipse['chi2_quantile'], expected_quantile)
        mean = arrays[method+'_concentrations_mBq_m3'][PAIR]
        K = arrays[method+'_concentration_covariance_'+kind][np.ix_(PAIR,PAIR)]
        close(mean, ellipse['center_mBq_m3'])
        close(K, ellipse['covariance_mBq2_m6'])
        close(K[0,1]/np.sqrt(K[0,0]*K[1,1]), ellipse['correlation'])
        radii = np.asarray(ellipse['semi_axes_mBq_m3'])
        assert np.all(radii > 0)
        angle = np.deg2rad(ellipse['major_axis_angle_degrees'])
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        close(rotation @ np.diag(radii**2/expected_quantile) @ rotation.T, K)
        t = np.linspace(0, 2*np.pi, 721)
        offset = rotation @ (radii[:,None]*np.array([np.cos(t),np.sin(t)]))
        mahalanobis = np.sum(offset*np.linalg.solve(K, offset),axis=0)
        close(mahalanobis, np.full(721, expected_quantile))
        checks.append(dict(method=method,uncertainty=kind,correlation=ellipse['correlation'],
            n_contour_points=721,max_mahalanobis_error=float(np.max(np.abs(mahalanobis-expected_quantile)))))
    with (ROOT/'concentrations.csv').open(encoding='utf-8-sig',newline='') as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 16
    for row in table:
        result = results[row['method_key']]
        index = scenario['isotope_order'].index(row['isotope'])
        close(float(row['concentration_mBq_m3']), 1000*result['activities']['a_c'][index])
        if result['activities']['K_A_c_stat'] is None:
            assert row['sigma_stat_mBq_m3'] == row['sigma_total_mBq_m3'] == ''
        else:
            close(float(row['sigma_stat_mBq_m3']), 1000*np.sqrt(result['activities']['K_A_c_stat'][index][index]))
            close(float(row['sigma_total_mBq_m3']), 1000*np.sqrt(result['activities']['K_A_c'][index][index]))
        j = result['detection']['component_indices'].index(index)
        close(float(row['LC_mBq_m3']), 1000*result['detection']['L_C'][j])
        close(float(row['MDC_mBq_m3']), 1000*result['detection']['MDC'][j])
        assert (row['detected_from_free_estimate']=='True') == result['detection']['detected'][j]
        assert (row['mdc_converged']=='True') == result['detection']['converged'][j]
    filenames = ['scenario.json','measurement_arrays.npz','concentrations.csv','ellipses.json','fit_quality.csv','fit_diagnostics.json','analyze_measurement.py']
    filenames += [f'result_{method}.{suffix}' for method in METHODS for suffix in ('json','txt')]
    record = dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(__file__),no_refitting=True,no_simulation=True,
        n_methods=4,n_joint_observations=21,n_joint_parameters=17,n_csv_rows=16,
        full_nuisance_covariance_checked=True,chi2_quantile=expected_quantile,
        isotope_pair=['Xe133','Xe131m'],concentration_units='mBq/m3',methods=methods,
        ellipses=checks,files_sha256={name:sha(ROOT/name) for name in filenames})
    (ROOT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
