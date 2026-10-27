"""The existing XeMat LC calculation, stopping before any LD/MDC work.

The formulas and covariance implementation are exactly those of the LC part
of xemat.detection.compute_detection_limits. A separate small adapter is used
because that public API also calculates an unwanted MDC even in fixed mode.
No optimizer or null-hypothesis fit is called here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from xemat.detection import model_covariance


@dataclass(frozen=True)
class CriticalLevel:
    L_C: float
    sigma0: float
    null_spectrum: np.ndarray


def critical_level(S, b, physical_positive, *, target_index=4, alpha=0.05):
    """Return LC in native physical units (Bq/m3 for fresh Xe).

    z=[a_V; gamma; zeta] is the already-fitted nonnegative physical vector.
    Only the tested fresh coordinate is zeroed. All nuisance estimates and
    all 17 design columns remain in the Fisher covariance. Negative or zero
    null means are rejected, never clipped to epsilon.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or not S.size or not np.all(np.isfinite(S)):
        raise ValueError('S: expected a nonempty finite matrix')
    n_rows, n_parameters = S.shape
    background = np.asarray(b, dtype=float)
    positive = np.asarray(physical_positive, dtype=float)
    if background.shape != (n_rows,) or not np.all(np.isfinite(background)):
        raise ValueError('b: expected finite vector of length n_rows')
    if np.any(background < 0):
        raise ValueError('Fixed additive b must be nonnegative')
    if positive.shape != (n_parameters,) or not np.all(np.isfinite(positive)):
        raise ValueError('physical_positive: expected finite complete physical vector')
    if np.any(positive < 0):
        raise ValueError('The auxiliary physical estimate must be nonnegative')
    if (isinstance(target_index, (bool, np.bool_)) or
            not isinstance(target_index, (int, np.integer)) or
            not 0 <= target_index < n_parameters):
        raise ValueError('target_index: invalid physical component index')
    if not np.isfinite(alpha) or not 0 < alpha < .5:
        raise ValueError('alpha must lie in (0, 0.5)')
    nuisance = positive.copy()
    nuisance[target_index] = 0.
    mu0 = background + S @ nuisance
    if np.any(mu0 <= 0):
        raise ValueError('The null model mean must be > 0 in all ROI; no epsilon replacement')
    # This is the production SVD covariance, with its unchanged default rcond.
    K0 = model_covariance(S, mu0)
    sigma0 = float(np.sqrt(K0[target_index, target_index]))
    LC = float(norm.ppf(1-alpha) * sigma0)
    return CriticalLevel(L_C=LC, sigma0=sigma0, null_spectrum=mu0)
