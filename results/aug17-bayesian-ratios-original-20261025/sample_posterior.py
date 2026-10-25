"""Sample the unchanged August joint model; no refit or new observations.

Run from the archive directory: python sample_posterior.py
The flat prior is on physical z=[a_V; gamma; zeta], not on beta or log(z).
The first coordinate is the legacy auxiliary radon coordinate.
"""
from pathlib import Path
import os
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('PYTENSOR_FLAGS', 'base_compiledir=' + str(ROOT / '.cache' / 'pytensor'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import gammaln, xlogy

ISOTOPES = ['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']
ROIS = [1, 2, 3, 5, 6, 7, 8]
NAMES = ['a_V_aux:Rn222'] + ['a_V:' + s for s in ISOTOPES[1:]]
NAMES += ['gamma:' + s for s in ISOTOPES] + [f'zeta:ROI{r}' for r in ROIS]
UNITS = ['Bq/m3 (legacy auxiliary, not atmospheric Rn)'] + ['Bq/m3'] * 4
UNITS += ['expected reference GASBK counts'] * 5 + ['expected DETBK counts'] * 7
RATIO_NAMES = ['Xe131m/Xe133', 'Xe133m/Xe133']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_model():
    a = dict(np.load(ROOT / 'source' / 'measurement_arrays.npz', allow_pickle=False))
    S, x, b = (a[k] for k in ['physical_matrix', 'x', 'b'])
    assert S.shape == (21, 17) and x.shape == (21,) and np.all(b == 0)
    assert np.all(S >= 0) and np.all(S.sum(axis=0) > 0)
    assert np.linalg.matrix_rank(S / np.linalg.norm(S, axis=0)) == 17
    Q = np.eye(17)
    Q[:5, :5] = a['T_V']
    np.testing.assert_allclose(S @ Q, a['M'], rtol=1e-12, atol=1e-12)
    # Fixed positive scales only improve numerical conditioning. The linear
    # transformation z_i = scale_i*u_i has a constant determinant, so a flat
    # prior in u is exactly the requested flat prior in physical z.
    scale = np.sqrt(np.diag(a['poisson_full_covariance_physical']))
    assert np.all(scale > 0)
    coords = dict(parameter=NAMES, observation=[f'{block}:ROI{r}' for block in ['SAMPLE', 'GASBK', 'DETBK'] for r in ROIS],
                  isotope=ISOTOPES[1:], ratio=RATIO_NAMES)
    with pm.Model(coords=coords) as model:
        # HalfFlat is an improper CONSTANT density on u>0, not a log-uniform
        # prior. PyMC's default log transform adds sum(log(u)) automatically.
        # No additional Potential/Jacobian is added here.
        u = pm.HalfFlat('u', dims='parameter')
        z = pm.Deterministic('z', scale * u, dims='parameter')
        mu = pt.dot(S, z) + b
        pm.Poisson('observed_counts', mu=mu, observed=x.astype(np.int64), dims='observation')
        c = pm.Deterministic('concentrations_mBq_m3', 1000 * z[1:5], dims='isotope')
        pm.Deterministic('ratios', pt.stack([c[2] / c[1], c[3] / c[1]]), dims='ratio')
    return model, a, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draws', type=int, default=5000)
    parser.add_argument('--tune', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026102501)
    parser.add_argument('--initial-seed', type=int, default=2026102502)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    model, arrays, scale = load_model()
    map_z = arrays['poisson_nonnegative_physical_parameters']
    rng = np.random.default_rng(args.initial_seed)
    # No optimization: perturb the saved MAP by fractions of the original
    # statistical SD, moving boundary coordinates strictly into the orthant.
    starts = []
    for chain in range(4):
        z0 = map_z + scale * rng.normal(0, 0.8, 17)
        boundary = map_z < 0.2 * scale
        z0[boundary] = scale[boundary] * rng.uniform(0.2, 1.5, boundary.sum())
        z0 = np.maximum(z0, 0.1 * scale)
        starts.append(z0)
    starts = np.asarray(starts)
    assert np.all(starts > 0) and len(np.unique(starts, axis=0)) == 4
    initvals = [{'u': z0 / scale} for z0 in starts]
    # Directly check the transformed density, including EXACTLY one Jacobian.
    logp_fn = model.compile_logp(jacobian=True)
    checks = []
    for z0 in starts:
        q = np.log(z0 / scale)
        mu = arrays['physical_matrix'] @ z0 + arrays['b']
        ll = np.sum(xlogy(arrays['x'], mu) - mu - gammaln(arrays['x'] + 1))
        actual = float(logp_fn({'u_log__': q}))
        expected = float(ll + q.sum())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-9)
        checks.append(dict(compiled_logp=actual, poisson_loglik_plus_log_jacobian=expected))
    packages = ['pymc', 'arviz', 'pytensor', 'numpy', 'scipy', 'matplotlib', 'pandas', 'xarray', 'h5netcdf', 'h5py']
    settings = dict(model='Original joint Poisson SAMPLE/GASBK/DETBK, 21 observations, 17 physical parameters',
        production_commit='69ff5a62ccaaddeb6372f7f003109ce5ff6cf333',
        source_manifest=json.loads((ROOT / 'source_manifest.json').read_text()),
        prior='p(z) proportional to 1 on z >= 0; no upper cutoff',
        posterior_proper_reason='S is nonnegative and every column has positive sum; likelihood has exponential tails times a polynomial',
        physical_coordinates='[a_V; gamma; zeta]', parameter_names=NAMES, parameter_units=UNITS,
        concentration_order=ISOTOPES[1:], concentration_units='mBq/m3', ratio_order=RATIO_NAMES,
        scale=scale.tolist(), initial_z=starts.tolist(), initial_seed=args.initial_seed,
        sampler_seed=args.seed, seed_handling='Integer seed passed to PyMC sample; its independent chain RNGs are spawned internally',
        sampler='PyMC NUTS', chains=4, tune_per_chain=args.tune, draws_per_chain=args.draws,
        init='adapt_diag (no MAP optimization and no initialization jitter)', target_accept=0.95,
        max_treedepth=12, cores=args.cores, blas_cores=1,
        transformed_prior='u=z/scale ~ HalfFlat; default log transform q=log(u); automatic +sum(q), no manual Jacobian',
        jacobian_checks=checks, map_source='Saved original nonnegative Poisson solution; no new MAP optimization',
        fixed_calibrations=True, statistical_covariance_only=True, covariance_rescaling=False,
        new_observation_simulation=False, new_LC_LD_MDC_calculations=False,
        software={p: importlib.metadata.version(p) for p in packages}, python=platform.python_version(),
        platform=platform.platform(), sampler_script_sha256=sha(__file__))
    dump('settings.json', settings)
    np.savez_compressed(ROOT / 'initial_states.npz', z=starts, u=starts / scale, scale=scale,
                        parameter_names=np.array(NAMES), parameter_units=np.array(UNITS))
    begin = time.monotonic()
    with model:
        trace = pm.sample(draws=args.draws, tune=args.tune, chains=4, cores=args.cores,
            random_seed=args.seed, init='adapt_diag', initvals=initvals, progressbar=False,
            target_accept=0.95, nuts={'max_treedepth': 12}, blas_cores=1,
            discard_tuned_samples=False, return_inferencedata=True,
            idata_kwargs={'save_warmup': True, 'log_likelihood': False})
    trace.to_netcdf(ROOT / 'posterior.nc')
    output = dict(z=trace.posterior['z'].transpose('chain', 'draw', 'parameter').values,
        concentrations_mBq_m3=trace.posterior['concentrations_mBq_m3'].transpose('chain', 'draw', 'isotope').values,
        ratios=trace.posterior['ratios'].transpose('chain', 'draw', 'ratio').values,
        chain=trace.posterior.chain.values, draw=trace.posterior.draw.values,
        parameter_names=np.array(NAMES), parameter_units=np.array(UNITS), isotope_names=np.array(ISOTOPES[1:]),
        concentration_units=np.array('mBq/m3'), ratio_names=np.array(RATIO_NAMES), ratio_units=np.array('dimensionless'))
    for name in trace.sample_stats.data_vars:
        output['sample_stats_' + name] = trace.sample_stats[name].values
    np.savez_compressed(ROOT / 'posterior.npz', **output)
    settings['wall_seconds'] = time.monotonic() - begin
    settings['posterior_shape'] = list(output['z'].shape)
    settings['posterior_divergences'] = int(trace.sample_stats.diverging.sum())
    settings['warmup_divergences'] = int(trace.warmup_sample_stats.diverging.sum())
    settings['posterior_npz_sha256'] = sha(ROOT / 'posterior.npz')
    dump('settings.json', settings)
    print(json.dumps({k: settings[k] for k in ['posterior_shape', 'wall_seconds', 'posterior_divergences', 'warmup_divergences']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
