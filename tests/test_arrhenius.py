"""Tests for the weighted Arrhenius fitting utilities."""

from __future__ import annotations

import numpy as np

from analysis.arrhenius import fit_arrhenius_irls, fit_arrhenius_wls


def test_wls_recovers_exact_line() -> None:
    """An exact line must be recovered to machine precision."""
    x = np.linspace(3e-4, 6e-4, 20)
    intercept, slope = -5.0, -12000.0
    y = intercept + slope * x

    fit = fit_arrhenius_wls(x, y)
    assert fit is not None
    assert abs(fit.ln_D0 - intercept) < 1e-6
    assert abs(fit.slope - slope) < 1e-3
    assert fit.r2 > 0.9999
    assert fit.n_points == 20


def test_wls_downweights_outlier() -> None:
    """A large sigma must suppress the influence of an outlier."""
    x = np.linspace(3e-4, 6e-4, 10)
    intercept, slope = 0.0, -10000.0
    y = intercept + slope * x
    y[0] += 5.0   # inject an outlier

    sigma = np.ones_like(x)
    sigma[0] = 100.0   # declare the outlier as highly uncertain

    fit = fit_arrhenius_wls(x, y, sigma=sigma)
    assert fit is not None
    assert abs(fit.slope - slope) < 50.0


def test_wls_returns_none_for_too_few_points() -> None:
    """Fewer than three points must yield ``None``."""
    x = np.array([3e-4, 4e-4])
    y = np.array([1.0, 2.0])
    assert fit_arrhenius_wls(x, y) is None


def test_irls_returns_fit_on_noisy_data() -> None:
    """IRLS must return a finite fit on noisy data."""
    rng = np.random.default_rng(42)
    x = np.linspace(3e-4, 6e-4, 15)
    intercept, slope = -4.0, -11000.0
    y = intercept + slope * x + rng.normal(0.0, 0.05, size=x.size)

    fit = fit_arrhenius_irls(x, y)
    assert fit is not None
    assert np.isfinite(fit.Q_eV)
    assert abs(fit.slope - slope) < 500.0