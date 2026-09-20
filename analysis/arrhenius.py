"""Weighted Arrhenius fitting utilities.

The tracer and vacancy diffusivities are fitted to

.. math::

    \\ln D = \\ln D_0 - \\frac{Q}{k_B T},

i.e. a straight line in :math:`1/T` with slope :math:`-Q/k_B` and intercept
:math:`\\ln D_0`.

Why weighted least squares
--------------------------
The MSD-derived diffusivity is heteroscedastic: the statistical uncertainty of
:math:`D` grows towards the temperature extremes, where the MSD is either too
small to converge (low T) or the run is short relative to the jump rate
(high T).  Ordinary least squares assumes a constant variance and therefore
lets the noisiest points dominate the fit.  Weighted least squares with
:math:`w_i = 1/\\sigma_i^2` gives each point its proper statistical weight and
yields an unbiased estimate of :math:`Q` and :math:`D_0`.

When no per-point uncertainty is supplied, the weights default to unity and
the result reduces to ordinary least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import KB_EV

__all__ = ["ArrheniusFit", "fit_arrhenius_irls", "fit_arrhenius_wls"]


@dataclass(frozen=True)
class ArrheniusFit:
    """Result of a linear Arrhenius fit in inverse-temperature space.

    Attributes
    ----------
    ln_D0 : float
        Fitted intercept, equal to :math:`\\ln D_0`.
    slope : float
        Fitted slope, equal to :math:`-Q/k_B`.
    Q_eV : float
        Activation energy [eV].
    r2 : float
        Weighted coefficient of determination.
    n_points : int
        Number of points used in the fit.
    """

    ln_D0: float
    slope: float
    Q_eV: float
    r2: float
    n_points: int

    @property
    def D0(self) -> float:
        """Pre-exponential factor :math:`D_0` in the same units as ``D``."""
        return float(np.exp(self.ln_D0))


def fit_arrhenius_wls(
    inv_T: np.ndarray,
    ln_D: np.ndarray,
    sigma: np.ndarray | None = None,
) -> ArrheniusFit | None:
    """Fit ``ln_D`` against ``inv_T`` by weighted least squares.

    Parameters
    ----------
    inv_T : numpy.ndarray
        Inverse temperatures :math:`1/T` [K^-1].
    ln_D : numpy.ndarray
        Natural logarithm of the diffusivity.
    sigma : numpy.ndarray, optional
        Per-point standard deviation of ``ln_D``.  When provided, weights are
        :math:`w_i = 1/\\sigma_i^2`; otherwise unit weights are used.

    Returns
    -------
    ArrheniusFit or None
        The fit, or ``None`` if fewer than three finite points are available
        or the design matrix is singular.
    """
    x = np.asarray(inv_T, dtype=float)
    y = np.asarray(ln_D, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma_arr = np.asarray(sigma, dtype=float)
        mask &= np.isfinite(sigma_arr) & (sigma_arr > 0)
    else:
        sigma_arr = None

    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None

    if sigma_arr is None:
        weights = np.ones_like(x)
    else:
        weights = 1.0 / sigma_arr[mask] ** 2

    sum_w = weights.sum()
    sum_wx = (weights * x).sum()
    sum_wy = (weights * y).sum()
    sum_wxx = (weights * x * x).sum()
    sum_wxy = (weights * x * y).sum()

    denominator = sum_w * sum_wxx - sum_wx ** 2
    if abs(denominator) < 1e-30:
        return None

    slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denominator
    intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denominator

    y_hat = intercept + slope * x
    y_bar = sum_wy / sum_w
    ss_res = float((weights * (y - y_hat) ** 2).sum())
    ss_tot = float((weights * (y - y_bar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return ArrheniusFit(
        ln_D0=float(intercept),
        slope=float(slope),
        Q_eV=float(-slope * KB_EV),
        r2=float(r2),
        n_points=int(x.size),
    )


def fit_arrhenius_irls(
    inv_T: np.ndarray,
    ln_D: np.ndarray,
    n_iter: int = 3,
    floor_fraction: float = 0.1,
) -> ArrheniusFit | None:
    """Fit ``ln_D`` against ``inv_T`` by iteratively reweighted least squares.

    The per-point uncertainty is not available in the aggregated ``Dv.txt``
    and ``Cv.txt`` files, so it is estimated from the residuals of the current
    fit.  Starting from an ordinary least-squares fit, each iteration sets

    .. math::

        \\sigma_i = \\max\\left(|r_i|, \\; f \\cdot \\mathrm{MAD}(r)\\right),

    where :math:`r_i` are the residuals, :math:`\\mathrm{MAD}` is the median
    absolute deviation, and :math:`f` is ``floor_fraction``.  The floor keeps
    the weights finite when a residual is near zero.  This is the standard
    iteratively reweighted least-squares scheme and it down-weights the noisy
    temperature extremes that dominate an unweighted fit.

    Parameters
    ----------
    inv_T : numpy.ndarray
        Inverse temperatures :math:`1/T` [K^-1].
    ln_D : numpy.ndarray
        Natural logarithm of the diffusivity.
    n_iter : int, optional
        Number of reweighting iterations.
    floor_fraction : float, optional
        Fraction of the residual MAD used as the weight floor.

    Returns
    -------
    ArrheniusFit or None
        The final fit, or ``None`` if fewer than three finite points are
        available.
    """
    x = np.asarray(inv_T, dtype=float)
    y = np.asarray(ln_D, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None

    fit = fit_arrhenius_wls(x, y)
    if fit is None:
        return None

    for _ in range(max(0, n_iter)):
        residuals = y - (fit.ln_D0 + fit.slope * x)
        scale = float(np.median(np.abs(residuals)))
        floor = max(scale * floor_fraction, 1e-12)
        sigma = np.maximum(np.abs(residuals), floor)
        updated = fit_arrhenius_wls(x, y, sigma=sigma)
        if updated is None:
            break
        fit = updated

    return fit