"""Audit the saved Bayesian analysis without optimization or simulation.

Run next to the saved files: python verify_bayesian.py
This script does not import the sampler or the original fitting workflow.
It reconstructs the matrix identity, full Fisher covariance, ratios, and
diagnostics directly from archived source arrays and retained chain draws.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import arviz as az
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, xlogy
from scipy.stats import chi2, norm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_csv(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def matrix_csv(name, expected, names):
    rows = read_csv(name)
    assert [row['parameter'] for row in rows] == list(names)
    numeric = np.asarray([[float(row[n]) for n in names] for row in rows])
    close(numeric, expected, rtol=1e-9)


def close(actual, expected, **kwargs):
    np.testing.assert_allclose(actual, expected, rtol=kwargs.get('rtol', 2e-11),
                               atol=kwargs.get('atol', 2e-12))


def main():
    manifest = read_json('source_manifest.json')
    for name, digest in manifest['source_files'].items():
        assert sha(ROOT / 'source' / name) == digest, name
    source = dict(np.load(ROOT / 'source' / 'measurement_arrays.npz', allow_pickle=False))
    S, M, x, b = [source[k] for k in ('physical_matrix', 'M', 'x', 'b')]
    assert S.shape == M.shape == (21, 17)
    assert np.all(S >= 0) and np.all(S.sum(axis=0) > 0)
    assert np.all(b == 0)
    expected_x = [121,154,78936,27029,13886,28048,4441,
                  52,76,79,21,9,20,14,522,854,494,66,64,89,97]
    np.testing.assert_array_equal(x, expected_x)
    assert source['calibration_R_c0'][5, 2] == 0.34
    assert source['calibration_R_g0'][5, 2] == 0.18
    Q = np.eye(17)
    Q[:5, :5] = source['T_V']
    close(S @ Q, M)
    assert Q[2, 4] < 0  # physical a_V>=0 differs from beta>=0.
    assert np.linalg.matrix_rank(S / np.linalg.norm(S, axis=0)) == 17
    scenario = read_json('source/scenario.json')
    assert scenario['radon_mode'] == 'legacy'
    assert scenario['fixed_roi'] and scenario['fixed_efficiencies']
    assert scenario['detector'] == 221
    for block in ('sample_efficiencies', 'gasbk_efficiencies'):
        close(scenario['metadata'][block], [.241169,.367784,.233937,.233937])
    for method in ('poisson', 'poisson_nonnegative'):
        close(Q @ source[method+'_theta'], source[method+'_physical_parameters'])
        close(S @ source[method+'_physical_parameters'], source[method+'_mu'])
    original_ml = source['poisson_physical_parameters']
    original_map = source['poisson_nonnegative_physical_parameters']
    assert np.all(original_map >= 0)
    # Independent QR of the full physical expected Fisher matrix: no nuisance
    # parameter is held fixed. This must reproduce the archived statistical K.
    A = S / np.sqrt(source['poisson_mu'])[:, None]
    _, R = np.linalg.qr(A)
    R_inv = solve_triangular(R, np.eye(17))
    K = R_inv @ R_inv.T
    close(K, source['poisson_full_covariance_physical'], rtol=1e-9)
    close(K[:5,:5] * 1e6, source['poisson_concentration_covariance_statistical'], rtol=1e-9)
    ml_c = original_ml[1:5]*1000
    K_c = K[1:5,1:5]*1e6
    gradients = np.zeros((2,4))
    for j, numerator in enumerate((2,3)):
        gradients[j,numerator] = 1/ml_c[1]
        gradients[j,1] = -ml_c[numerator]/ml_c[1]**2
    ml_ratios = ml_c[[2,3]]/ml_c[1]
    K_ratios = gradients @ K_c @ gradients.T
    settings = read_json('settings.json')
    assert settings['fixed_calibrations'] and settings['statistical_covariance_only']
    assert not settings['covariance_rescaling']
    assert not settings['new_observation_simulation']
    assert not settings['new_LC_LD_MDC_calculations']
    assert settings['sampler_script_sha256'] == sha(ROOT/'sample_posterior.py')
    scale = np.asarray(settings['scale'])
    close(scale, np.sqrt(np.diag(source['poisson_full_covariance_physical'])))
    initial = dict(np.load(ROOT/'initial_states.npz', allow_pickle=False))
    np.testing.assert_array_equal(initial['z'], settings['initial_z'])
    assert initial['z'].shape == (4,17) and np.all(initial['z'] > 0)
    assert np.unique(initial['z'], axis=0).shape[0] == 4
    close(initial['u'], initial['z']/scale)
    for item in settings['jacobian_checks']:
        close(item['compiled_logp'], item['poisson_loglik_plus_log_jacobian'], atol=2e-9)

    samples = dict(np.load(ROOT/'posterior.npz', allow_pickle=False))
    z = samples['z']
    shape = (4, settings['draws_per_chain'], 17)
    assert z.shape == shape and np.isfinite(z).all() and np.all(z > 0)
    assert settings['posterior_npz_sha256'] == sha(ROOT/'posterior.npz')
    np.testing.assert_array_equal(samples['parameter_names'], settings['parameter_names'])
    np.testing.assert_array_equal(samples['parameter_units'], settings['parameter_units'])
    np.testing.assert_array_equal(samples['chain'], np.arange(4))
    np.testing.assert_array_equal(samples['draw'], np.arange(shape[1]))
    c = samples['concentrations_mBq_m3']
    ratios = samples['ratios']
    close(c, 1000*z[:,:,1:5])
    close(ratios, c[:,:,[2,3]]/c[:,:,1,None])
    assert ratios.shape == (4,shape[1],2)
    # lp is the transformed NUTS density. A flat prior in u with q=log(u)
    # requires exactly +sum(q). Checking every retained lp catches both a
    # missing Jacobian and accidental double inclusion without running PyMC.
    flat_z = z.reshape(-1,17)
    mu = flat_z @ S.T
    assert np.all(mu > 0)
    ll = np.sum(xlogy(x,mu)-mu-gammaln(x+1), axis=1)
    expected_lp = ll + np.log(flat_z/scale).sum(axis=1)
    lp = samples['sample_stats_lp'].reshape(-1)
    close(lp, expected_lp, rtol=0, atol=3e-8)
    idata = az.from_netcdf(ROOT/'posterior.nc')
    for name, saved in [('z',z), ('concentrations_mBq_m3',c), ('ratios',ratios)]:
        np.testing.assert_array_equal(idata.posterior[name].values,saved)
    assert idata.warmup_posterior.sizes['draw'] == settings['tune_per_chain']
    assert idata.posterior.sizes['chain'] == 4
    diag = az.summary(idata, var_names=['z','ratios'], kind='diagnostics', round_to=None)
    rhat = az.rhat(idata, var_names=['z','ratios'], method='rank')
    bulk = az.ess(idata,var_names=['z','ratios'],method='bulk')
    tail = az.ess(idata,var_names=['z','ratios'],method='tail')
    rhat_values = np.concatenate([rhat['z'].values,rhat['ratios'].values])
    bulk_values = np.concatenate([bulk['z'].values,bulk['ratios'].values])
    tail_values = np.concatenate([tail['z'].values,tail['ratios'].values])
    assert np.isfinite(rhat_values).all() and np.max(rhat_values) < 1.01
    published_indices = [1,2,3,4,17,18]
    assert np.min(bulk_values[published_indices]) >= 1000
    assert np.min(tail_values[published_indices]) >= 1000
    assert int(samples['sample_stats_diverging'].sum()) == 0
    assert not np.any(samples['sample_stats_reached_max_treedepth'])
    assert min(az.bfmi(idata)) > .3

    # Validate every exported numerical table against all retained chain draws.
    original = dict(np.load(ROOT/'original_mle.npz',allow_pickle=False))
    np.testing.assert_array_equal(original['physical_mean'],original_ml)
    np.testing.assert_array_equal(original['map_physical'],original_map)
    close(original['physical_covariance_statistical'],K,rtol=1e-9)
    close(original['concentration_covariance_statistical'],K_c,rtol=1e-9)
    close(original['ratio_mean'],ml_ratios)
    close(original['ratio_covariance_delta'],K_ratios,rtol=1e-9)
    matrix_csv('original_mle_covariance_physical.csv',K,samples['parameter_names'])
    matrix_csv('original_mle_covariance_concentrations.csv',K_c,samples['isotope_names'])
    matrix_csv('original_mle_covariance_ratios.csv',K_ratios,samples['ratio_names'])
    factor = np.ones(17); factor[:5] = 1000
    displayed = np.concatenate([z*factor,ratios],axis=2)
    names = samples['parameter_names'].tolist()+samples['ratio_names'].tolist()
    exported_diagnostics = read_csv('diagnostics.csv')
    all_summary = read_csv('posterior_summary_all.csv')
    assert len(exported_diagnostics) == len(all_summary) == 19
    for j, name in enumerate(names):
        values = displayed[:,:,j]
        row = exported_diagnostics[j]
        assert row['quantity'] == name
        expected_d = dict(r_hat=rhat_values[j],ess_bulk=bulk_values[j],ess_tail=tail_values[j],
            mcse_mean=np.asarray(az.mcse(values,method='mean')).item(),
            mcse_sd=np.asarray(az.mcse(values,method='sd')).item(),
            mcse_median=np.asarray(az.mcse(values,method='median')).item(),
            mcse_q025=np.asarray(az.mcse(values,method='quantile',prob=.025)).item(),
            mcse_q975=np.asarray(az.mcse(values,method='quantile',prob=.975)).item())
        for key,value in expected_d.items(): close(float(row[key]),value)
        row = all_summary[j]
        assert row['quantity'] == name
        expected_s = dict(mean=values.mean(),median=np.median(values),sd=values.std(ddof=1),
            q025=np.quantile(values,.025),q975=np.quantile(values,.975),n_draws=values.size)
        for key,value in expected_s.items(): close(float(row[key]),value)
    for filename,values,centers,cov,reference_map,indices in [
        ('concentrations.csv',c,ml_c,K_c,1000*original_map[1:5],range(1,5)),
        ('ratios.csv',ratios,ml_ratios,K_ratios,original_map[[3,4]]/original_map[2],range(17,19))]:
        rows = read_csv(filename)
        assert len(rows) == values.shape[2]
        for j, k in enumerate(indices):
            row = rows[j]
            for key in ('mean','median','sd','q025','q975','n_draws'):
                close(float(row[key]),float(all_summary[k][key]))
            for key in ('r_hat','ess_bulk','ess_tail','mcse_mean','mcse_sd','mcse_median','mcse_q025','mcse_q975'):
                close(float(row[key]),float(exported_diagnostics[k][key]))
            sd = np.sqrt(cov[j,j])
            expected_c = dict(free_ml=centers[j],free_ml_statistical_sd=sd,
                free_ml_normal_lower95=centers[j]-norm.ppf(.975)*sd,
                free_ml_normal_upper95=centers[j]+norm.ppf(.975)*sd,
                saved_map=reference_map[j],sd_ratio_posterior_to_ml=values[:,:,j].std(ddof=1)/sd)
            for key,value in expected_c.items(): close(float(row[key]),value,rtol=1e-9)
    cf,rf = c.reshape(-1,4),ratios.reshape(-1,2)
    posterior_cov = dict(np.load(ROOT/'posterior_covariances.npz',allow_pickle=False))
    for key,expected in dict(physical_covariance=np.cov(flat_z,rowvar=False,ddof=1),
        concentration_covariance=np.cov(cf,rowvar=False,ddof=1),
        concentration_correlation=np.corrcoef(cf,rowvar=False),
        ratio_covariance=np.cov(rf,rowvar=False,ddof=1),ratio_correlation=np.corrcoef(rf,rowvar=False)).items():
        close(posterior_cov[key],expected)
    matrix_csv('posterior_covariance_concentrations.csv',np.cov(cf,rowvar=False,ddof=1),samples['isotope_names'])
    matrix_csv('posterior_correlation_concentrations.csv',np.corrcoef(cf,rowvar=False),samples['isotope_names'])
    mle_rows = read_csv('original_mle_parameters.csv')
    for j,row in enumerate(mle_rows):
        assert row['parameter'] == samples['parameter_names'][j]
        close(float(row['free_ml']),original_ml[j])
        close(float(row['statistical_sd']),np.sqrt(K[j,j]),rtol=1e-9)
        close(float(row['saved_nonnegative_map']),original_map[j])
    chain_rows = read_csv('chain_diagnostics.csv')
    assert len(chain_rows) == 4
    for j,row in enumerate(chain_rows):
        assert int(row['chain']) == j and int(row['saved_draws']) == shape[1]
        close(float(row['bfmi']),az.bfmi(idata)[j])
        for key,expected in dict(divergences=int(idata.sample_stats.diverging[j].sum()),
            warmup_divergences=int(idata.warmup_sample_stats.diverging[j].sum()),
            max_tree_depth=int(idata.sample_stats.tree_depth[j].max()),
            max_treedepth_hits=int(idata.sample_stats.reached_max_treedepth[j].sum()),
            mean_acceptance_rate=float(idata.sample_stats.acceptance_rate[j].mean()),
            mean_n_steps=float(idata.sample_stats.n_steps[j].mean()),
            step_size=float(idata.sample_stats.step_size_bar[j,-1])).items(): close(float(row[key]),expected)

    plot = read_json('plot_metadata.json')
    cf, rf = c.reshape(-1,4), ratios.reshape(-1,2)
    assert plot['joint']['posterior_draw_count'] == shape[0]*shape[1]
    assert plot['ratios']['posterior_draw_count'] == shape[0]*shape[1]
    assert not plot['ratios']['normal_tail_truncated_at_zero']
    assert plot['style']['png_dpi'] == 300 and plot['style']['vector_pdf']
    assert plot['style']['mathtext_fontset'] == 'cm'
    scatter_indices = np.asarray(plot['joint']['scatter_flat_indices'])
    assert np.unique(scatter_indices).size == scatter_indices.size
    assert np.all((scatter_indices >= 0) & (scatter_indices < len(cf)))
    for j, (pair, numerator) in enumerate(zip(plot['joint']['pairs'], (2,3))):
        indices = [1,numerator]
        block = K_c[np.ix_(indices,indices)]
        close(pair['ml_center_mBq_m3'], ml_c[indices])
        close(pair['map_center_mBq_m3'], original_map[1:5][indices]*1000)
        close(pair['ml_stat_covariance_mBq2_m6'], block, rtol=1e-9)
        points = np.asarray(pair['ellipse_points_mBq_m3']) - ml_c[indices]
        quadratic = np.einsum('ni,in->n',points,np.linalg.solve(block,points.T))
        close(quadratic, chi2.ppf(.95,2), rtol=1e-9)
        close(pair['posterior_correlation'],np.corrcoef(cf[:,indices].T)[0,1])
        rd = plot['ratios']['ratios'][j]
        close(rd['posterior_mean'], rf[:,j].mean())
        close(rd['posterior_sd'], rf[:,j].std(ddof=1))
        close(rd['posterior_median'], np.median(rf[:,j]))
        close(rd['credible_interval_equal_tail_95'],np.quantile(rf[:,j],[.025,.975]))
        close(rd['delta_method']['mean'],ml_ratios[j])
        close(rd['delta_method']['sd'],np.sqrt(K_ratios[j,j]),rtol=1e-9)
        edges = np.asarray(rd['hist_bin_edges'])
        counts,_ = np.histogram(rf[:,j], bins=edges)
        assert counts.sum() == len(rf)
        close(rd['hist_density'],counts/(len(rf)*np.diff(edges)))
        close(rd['hist_integral'],1)
        close(rd['normal_probability_below_zero'],norm.cdf(-ml_ratios[j]/np.sqrt(K_ratios[j,j])),rtol=1e-8)
    result = dict(status='PASS', source_sha256_files_checked=len(manifest['source_files']),
        original_Rs_ROI7_Xe133=float(source['calibration_R_c0'][5,2]),
        original_Rg_ROI7_Xe133=float(source['calibration_R_g0'][5,2]),
        original_observations=len(x), parameters=17, shape_chain_draw_parameter=list(shape),
        all_chain_draws_retained=True, no_refits_or_observation_simulation=True,
        full_nuisance_Fisher_covariance_verified=True,
        csv_tables_checked=11, posterior_summary_quantities_checked=19,
        max_log_density_jacobian_error=float(np.max(abs(lp-expected_lp))),
        max_rank_normalized_rhat=float(rhat_values.max()),
        min_bulk_ess_all19=float(bulk_values.min()), min_tail_ess_all19=float(tail_values.min()),
        posterior_divergences=int(samples['sample_stats_diverging'].sum()),
        BFMI_per_chain=az.bfmi(idata).tolist(),
        delta_ratios=ml_ratios.tolist(), delta_ratio_sd=np.sqrt(np.diag(K_ratios)).tolist(),
        ellipse_contours_checked=2, posterior_and_normal_ratio_curves_checked=2)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
