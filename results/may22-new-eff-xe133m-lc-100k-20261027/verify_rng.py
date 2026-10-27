"""Check that saved observations exactly reproduce the recorded random seeds.

This is a deterministic replay for integrity checking, not a new Monte Carlo
experiment. No estimates, LC, LD or MDC are calculated or overwritten.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    scenario = json.loads((ROOT / 'scenario.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'scenario_arrays.npz', allow_pickle=False) as source:
        mu0 = source['mu0']
    with np.load(ROOT / 'replicates.npz', allow_pickle=False) as saved:
        x = saved['x']
        seeds = saved['seed_table']
        ids = saved['replicate_id']
    n = scenario['n_generated']
    size = scenario['batch_size']
    assert x.shape == (n, 21)
    np.testing.assert_array_equal(ids, np.arange(n))
    for chunk in range(scenario['n_batches']):
        seed = [scenario['master_seed'], chunk]
        np.testing.assert_array_equal(seeds[chunk], seed)
        replay = np.random.default_rng(seed).poisson(mu0, size=(size, 21))
        np.testing.assert_array_equal(x[chunk*size:(chunk+1)*size], replay)
    result = dict(status='PASS', n_rows_verified=n, n_counts_verified=int(x.size),
                  all_observations_identical_to_seed_replay=True,
                  replay_only=True, no_new_experiment=True,
                  no_fits_or_detection_calculations=True, numpy_version=np.__version__,
                  replicates_sha256=hashlib.sha256((ROOT/'replicates.npz').read_bytes()).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'rng_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
