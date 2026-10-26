"""Summaries and diagnostics from saved chains only; no fits or simulation."""
from pathlib import Path
import os
import csv
import json

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import numpy as np
import arviz as az
from scipy.stats import norm


def csv_write(name, rows):
    with (ROOT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def matrix_csv(name, a, names):
    csv_write(name, [dict(parameter=key, **{k: float(v) for k, v in zip(names, row)}) for key, row in zip(names, a)])


def summary(values):
    v = np.asarray(values).ravel()
    q = np.quantile(v, [.025, .5, .975])
    return dict(mean=float(v.mean()), median=float(q[1]), sd=float(v.std(ddof=1)),
                q025=float(q[0]), q975=float(q[2]), n_draws=int(v.size))


def scalar(value):
    return np.asarray(value).item()


def main():
    a = dict(np.load(ROOT / 'source' / 'measurement_arrays.npz', allow_pickle=False))
    p = dict(np.load(ROOT / 'posterior.npz', allow_pickle=False))
    settings = json.loads((ROOT / 'settings.json').read_text())
    z, c, ratios = (p[k] for k in ['z', 'concentrations_mBq_m3', 'ratios'])
    assert z.shape == (4, settings['draws_per_chain'], 17) and np.all(np.isfinite(z)) and np.all(z > 0)
    np.testing.assert_array_equal(c, 1000 * z[:, :, 1:5])
    np.testing.assert_allclose(ratios, np.stack([c[:, :, 2] / c[:, :, 1], c[:, :, 3] / c[:, :, 1]], axis=-1), rtol=1e-14)
    factor = np.ones(17); factor[:5] = 1000
    # Summaries/MCSE use display units; stored z retains exact native units.
    display = z * factor
    values = np.concatenate([display, ratios], axis=2)
    names = p['parameter_names'].tolist() + p['ratio_names'].tolist()
    units = ['mBq/m3 (legacy auxiliary, not atmospheric Rn)'] + ['mBq/m3'] * 4 + p['parameter_units'][5:].tolist() + ['dimensionless'] * 2
    diag, all_summary = [], []
    for i, name in enumerate(names):
        v = values[:, :, i]
        row = dict(quantity=name, units=units[i], r_hat=scalar(az.rhat(v, method='rank')),
            ess_bulk=scalar(az.ess(v, method='bulk')), ess_tail=scalar(az.ess(v, method='tail')),
            mcse_mean=scalar(az.mcse(v, method='mean')), mcse_sd=scalar(az.mcse(v, method='sd')),
            mcse_median=scalar(az.mcse(v, method='median')),
            mcse_q025=scalar(az.mcse(v, method='quantile', prob=.025)),
            mcse_q975=scalar(az.mcse(v, method='quantile', prob=.975)))
        diag.append(row)
        all_summary.append(dict(quantity=name, units=units[i], **summary(v)))
    csv_write('diagnostics.csv', diag)
    csv_write('posterior_summary_all.csv', all_summary)
    native_mean = a['poisson_physical_parameters']
    native_cov = a['poisson_full_covariance_physical']
    map_z = a['poisson_nonnegative_physical_parameters']
    mle_c = 1000 * native_mean[1:5]
    K_c = 1e6 * native_cov[1:5, 1:5]
    # Full 17-parameter covariance, after nuisance propagation. Delta gradients
    # have zero nuisance entries but this is not the inverse of a reduced fit.
    G = np.zeros((2, 17))
    for j, numerator in enumerate([3, 4]):
        G[j, numerator] = 1 / native_mean[2]
        G[j, 2] = -native_mean[numerator] / native_mean[2] ** 2
    mle_ratio = native_mean[[3, 4]] / native_mean[2]
    K_ratio = G @ native_cov @ G.T
    np.savez_compressed(ROOT / 'original_mle.npz',
        theta=a['poisson_theta'], physical_mean=native_mean, physical_covariance_statistical=native_cov,
        theta_covariance_statistical=a['poisson_full_covariance_theta'],
        concentration_mean_mBq_m3=mle_c, concentration_covariance_statistical=K_c,
        map_physical=map_z, map_concentration_mBq_m3=1000*map_z[1:5],
        ratio_mean=mle_ratio, ratio_covariance_delta=K_ratio, ratio_gradient_physical=G,
        parameter_names=p['parameter_names'], parameter_units=p['parameter_units'],
        isotope_names=p['isotope_names'], ratio_names=p['ratio_names'],
        concentration_units=np.array('mBq/m3'), ratio_units=np.array('dimensionless'))
    csv_write('original_mle_parameters.csv', [dict(parameter=name, native_units=unit, free_ml=float(m),
        statistical_sd=float(sd), saved_nonnegative_map=float(m0))
        for name, unit, m, sd, m0 in zip(p['parameter_names'], p['parameter_units'], native_mean,
                                      np.sqrt(np.diag(native_cov)), map_z)])
    matrix_csv('original_mle_covariance_physical.csv', native_cov, p['parameter_names'])
    matrix_csv('original_mle_covariance_concentrations.csv', K_c, p['isotope_names'])
    matrix_csv('original_mle_covariance_ratios.csv', K_ratio, p['ratio_names'])
    compare_tables = []
    for filename, arr, labels, mean, cov, modes, diagnostics_indices, unit in [
        ('concentrations.csv', c, p['isotope_names'], mle_c, K_c, 1000*map_z[1:5], range(1,5), 'mBq/m3'),
        ('ratios.csv', ratios, p['ratio_names'], mle_ratio, K_ratio, map_z[[3,4]]/map_z[2], range(17,19), 'dimensionless')]:
        rows = []
        for j, (name, k) in enumerate(zip(labels, diagnostics_indices)):
            d = {key: val for key,val in diag[k].items() if key not in ['quantity','units']}
            sd = float(np.sqrt(cov[j,j]))
            row = dict(quantity=str(name), units=unit, **summary(arr[:,:,j]), free_ml=float(mean[j]),
                free_ml_statistical_sd=sd, free_ml_normal_lower95=float(mean[j]-norm.ppf(.975)*sd),
                free_ml_normal_upper95=float(mean[j]+norm.ppf(.975)*sd),
                saved_map=float(modes[j]), sd_ratio_posterior_to_ml=float(arr[:,:,j].std(ddof=1)/sd), **d)
            rows.append(row)
        csv_write(filename, rows)
        compare_tables.append(rows)
    flat_c, flat_z, flat_ratio = c.reshape(-1,4), z.reshape(-1,17), ratios.reshape(-1,2)
    np.savez_compressed(ROOT/'posterior_covariances.npz',
        physical_covariance=np.cov(flat_z,rowvar=False,ddof=1),
        concentration_covariance=np.cov(flat_c,rowvar=False,ddof=1), concentration_correlation=np.corrcoef(flat_c,rowvar=False),
        ratio_covariance=np.cov(flat_ratio,rowvar=False,ddof=1), ratio_correlation=np.corrcoef(flat_ratio,rowvar=False))
    matrix_csv('posterior_covariance_concentrations.csv', np.cov(flat_c,rowvar=False,ddof=1), p['isotope_names'])
    matrix_csv('posterior_correlation_concentrations.csv', np.corrcoef(flat_c,rowvar=False), p['isotope_names'])
    ss = {k.removeprefix('sample_stats_'):v for k,v in p.items() if k.startswith('sample_stats_')}
    idata = az.from_netcdf(ROOT/'posterior.nc')
    bfmi = az.bfmi(idata)
    chains = []
    for j in range(4):
        chains.append(dict(chain=j, saved_draws=z.shape[1], tune=settings['tune_per_chain'],
            divergences=int(ss['diverging'][j].sum()), warmup_divergences=int(idata.warmup_sample_stats.diverging[j].sum()),
            bfmi=float(bfmi[j]), max_tree_depth=int(ss['tree_depth'][j].max()),
            max_treedepth_hits=int(ss['reached_max_treedepth'][j].sum()),
            mean_acceptance_rate=float(ss['acceptance_rate'][j].mean()),
            mean_n_steps=float(ss['n_steps'][j].mean()), step_size=float(ss['step_size_bar'][j,-1])))
    csv_write('chain_diagnostics.csv', chains)
    published = [diag[k] for k in [1,2,3,4,17,18]]
    passed = (max(d['r_hat'] for d in diag)<1.01 and min(d['ess_bulk'] for d in published)>=1000
        and min(d['ess_tail'] for d in published)>=1000 and int(ss['diverging'].sum())==0
        and int(ss['reached_max_treedepth'].sum())==0 and float(np.min(bfmi))>.3)
    result = dict(status='passed' if passed else 'needs_attention', all_17_parameters_and_2_ratios_checked=True,
        posterior_draws=z.shape[0]*z.shape[1], chains=z.shape[0], saved_draws_per_chain=z.shape[1],
        max_rhat=float(max(d['r_hat'] for d in diag)), min_bulk_ess_all=float(min(d['ess_bulk'] for d in diag)),
        min_tail_ess_all=float(min(d['ess_tail'] for d in diag)),
        min_bulk_ess_published=float(min(d['ess_bulk'] for d in published)),
        min_tail_ess_published=float(min(d['ess_tail'] for d in published)),
        divergences=int(ss['diverging'].sum()), max_treedepth_hits=int(ss['reached_max_treedepth'].sum()),
        bfmi=bfmi.tolist(), warmup_discarded_from_summaries=True, draws_excluded=0,
        prior_positive_probability_not_used_for_detection=True)
    (ROOT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    for rows in compare_tables:
        for row in rows:
            print(row['quantity'], 'posterior mean/median/sd',row['mean'],row['median'],row['sd'],
                  '95% credible interval',row['q025'],row['q975'], 'freeML/sd',row['free_ml'],row['free_ml_statistical_sd'],flush=True)


if __name__ == '__main__':
    main()
