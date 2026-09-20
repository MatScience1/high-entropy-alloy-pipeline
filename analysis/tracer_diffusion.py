"""Stage 8 - tracer self-diffusion coefficient D*(T) per element.

Interface (called by ``run_pipeline.py``)
-----------------------------------------
``run(comp, txt_dir, plot_dir, comp_label)``

Physics
-------
Tracer diffusivity from the per-element MSD via the Einstein relation:

.. math::

    D^*_i(T) = \\lim_{t \\to \\infty} \\frac{\\mathrm{MSD}_i(t)}{6t}
    \\quad [\\text{\\AA}^2\\,\\text{ps}^{-1}],

converted to SI by multiplying by ``CONV_A2PS_TO_M2S = 1e-8``.

Tracer versus vacancy diffusion
-------------------------------
Two distinct quantities are computed by the pipeline:

* :math:`D_{v,i}` (Stage 6) is the *vacancy* diffusion coefficient, obtained
  from the MSD of the vacancy site.  It measures how fast the vacancy itself
  migrates.
* :math:`D^*_i` (this module) is the *tracer* self-diffusion coefficient of
  species :math:`i`, the quantity measured in a radiotracer experiment.  It is
  related to the vacancy mechanism by

  .. math::

      D^*_i = f_i \\, C_v \\, \\frac{D_{v,i}}{x_i},

  where :math:`C_v` is the equilibrium vacancy concentration (Stage 7),
  :math:`x_i` the mole fraction, and :math:`f_i` the tracer correlation
  factor.

Correlation factor
------------------
The correlation factor :math:`f` accounts for the geometric probability that
successive vacancy jumps of a given atom are uncorrelated.  For the BCC
lattice :math:`f = 0.727` (Compaan & Haven 1956).  Because :math:`D^*_i` is
measured directly from the atomic MSD, :math:`f` is already embedded in the
data and must not be applied a second time.  The uncorrelated jump
diffusivity would be :math:`D_J = D^*_i / f`, but :math:`D^*_i` is the
experimentally comparable quantity and is reported directly.

Arrhenius fit
-------------
.. math::

    \\ln D^*_i = \\ln D_{0,i} - \\frac{Q_i}{k_B T},

fitted by iteratively reweighted least squares (see
:mod:`analysis.arrhenius`) to account for the heteroscedastic MSD noise at the
temperature extremes.

Output
------
``results/<comp_id>/txt/D2_components.txt``
``results/<comp_id>/plots/D2_vs_invT.png``
``results/<comp_id>/plots/Dtotal_vs_invT.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from analysis.arrhenius import ArrheniusFit, fit_arrhenius_irls  # noqa: E402
from analysis.logparse import LogParseError  # noqa: E402
from logging_config import configure_logging, get_logger  # noqa: E402
from reporting import log_dataframe  # noqa: E402

logger = get_logger(__name__)

ELEMENTS_ALL: list[str] = ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
COLORS: dict[str, str] = {
    "W": "steelblue", "Mo": "darkorange", "Nb": "forestgreen",
    "Zr": "goldenrod", "Ti": "orchid", "Ta": "tomato",
}
T_MIN: int = 900
T_MAX: int = 3100


# ══════════════════════════════════════════════════════════════════════════════
#  Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_cv(txt_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T, Cv)`` from ``Cv.txt``, filtered to ``[T_MIN, T_MAX]``.

    Parameters
    ----------
    txt_dir : pathlib.Path
        Directory containing ``Cv.txt``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Temperatures [K] and vacancy concentrations.

    Raises
    ------
    LogParseError
        If the file is missing or cannot be parsed.
    """
    path = txt_dir / "Cv.txt"
    if not path.exists():
        raise LogParseError(f"file not found: {path}")
    try:
        data = np.genfromtxt(path, skip_header=2)
    except (OSError, ValueError) as exc:
        raise LogParseError(f"cannot parse {path}: {exc}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mask = (data[:, 0] >= T_MIN) & (data[:, 0] <= T_MAX)
    return data[mask, 0], data[mask, 2]


def _load_dv(txt_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``(T, {element: Dv_i})`` from ``Dv.txt``.

    The column layout is
    ``T  1/T  Dv_all  Dv_W  Dv_Mo  Dv_Nb  Dv_Zr  Dv_Ti  Dv_Ta``.

    Parameters
    ----------
    txt_dir : pathlib.Path
        Directory containing ``Dv.txt``.

    Returns
    -------
    tuple of (numpy.ndarray, dict of str to numpy.ndarray)
        Temperatures [K] and per-element vacancy diffusivities.

    Raises
    ------
    LogParseError
        If the file is missing or cannot be parsed.
    """
    path = txt_dir / "Dv.txt"
    if not path.exists():
        raise LogParseError(f"file not found: {path}")
    try:
        with open(path) as fh:
            header = fh.readline().lstrip("#").strip()
        col_names = header.split()
        data = np.genfromtxt(path, skip_header=1)
    except (OSError, ValueError) as exc:
        raise LogParseError(f"cannot parse {path}: {exc}") from exc
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mask = (data[:, 0] >= T_MIN) & (data[:, 0] <= T_MAX)
    data = data[mask]
    temperatures = data[:, 0]
    dv: dict[str, np.ndarray] = {}
    for j, col in enumerate(col_names):
        for element in ELEMENTS_ALL:
            if col.lower() == f"dv_{element.lower()}(m2/s)":
                dv[element] = data[:, j]
    return temperatures, dv


# ══════════════════════════════════════════════════════════════════════════════
#  Main function
# ══════════════════════════════════════════════════════════════════════════════

def run(
    comp: dict[str, float],
    txt_dir: Path,
    out_dir: Path,
    comp_label: str = "",
) -> None:
    """Compute D*_i and D*_total, then save text files and plots.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol; elements with zero
        concentration are skipped.
    txt_dir : pathlib.Path
        Directory containing ``Cv.txt`` and ``Dv.txt``.
    out_dir : pathlib.Path
        Output directory for the text file and plots.
    comp_label : str, optional
        Composition label used in plot titles.

    Raises
    ------
    LogParseError
        If the input files are missing or the temperature grids disagree.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    temperatures, cv = _load_cv(txt_dir)
    t_dv, dv_map = _load_dv(txt_dir)

    if not np.allclose(temperatures, t_dv, atol=0.5):
        raise LogParseError("temperature grids in Cv.txt and Dv.txt do not match")

    inv_t = 1.0 / temperatures

    active = [
        e for e in ELEMENTS_ALL
        if comp.get(e, 0.0) > 1e-6 and e in dv_map
    ]

    # Per-element tracer diffusion: D*_i = Cv * Dv_i / x_i.
    d_star: dict[str, np.ndarray] = {}
    for element in active:
        d_star[element] = (1.0 / comp[element]) * cv * dv_map[element]

    # Total: D*_total = Cv * sum_i Dv_i = sum_i x_i D*_i.
    d_total = cv * sum(dv_map[e] for e in active)

    # ── save text ─────────────────────────────────────────────────────────────
    header = (
        "T(K) 1/T(1/K) "
        + " ".join(f"D_{e}(m2/s)" for e in active)
        + " D_total(m2/s)"
    )
    out_array = np.column_stack(
        [temperatures, inv_t] + [d_star[e] for e in active] + [d_total]
    )
    txt_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_dir / "D2_components.txt", out_array,
               header=header, fmt="%15.6e")
    logger.info("Saved D2_components.txt -> %s", txt_dir)

    # ── Arrhenius summary table ───────────────────────────────────────────────
    rows: list[dict[str, object]] = []
    fits: dict[str, ArrheniusFit | None] = {}
    for element in active:
        fit = fit_arrhenius_irls(inv_t, np.log(np.where(d_star[element] > 0,
                                                        d_star[element], np.nan)))
        fits[element] = fit
        if fit is not None:
            rows.append({
                "element": element,
                "D0 (m2/s)": fit.D0,
                "Q (eV)": fit.Q_eV,
                "R2": fit.r2,
                "N": fit.n_points,
            })
    fit_total = fit_arrhenius_irls(
        inv_t, np.log(np.where(d_total > 0, d_total, np.nan))
    )
    if fit_total is not None:
        rows.append({
            "element": "total",
            "D0 (m2/s)": fit_total.D0,
            "Q (eV)": fit_total.Q_eV,
            "R2": fit_total.r2,
            "N": fit_total.n_points,
        })
    if rows:
        log_dataframe(
            logger, pd.DataFrame(rows),
            title="Tracer diffusion Arrhenius parameters (WLS):",
        )

    # ── plot 1: per-element D* ────────────────────────────────────────────────
    formula_el = (
        r"$D^*_i = C_v \cdot D_v^{(i)} / c_i$"
        "\n" r"$c_i$ = composition fraction"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for element in active:
        d = d_star[element]
        valid = (d > 0) & np.isfinite(d)
        ax.semilogy(inv_t[valid], d[valid], "o", color=COLORS[element],
                    ms=5, label=element)
        fit = fits[element]
        if fit is not None:
            ax.semilogy(inv_t[valid],
                        np.exp(fit.ln_D0 + fit.slope * inv_t[valid]),
                        "--", color=COLORS[element], alpha=0.75,
                        label=f"  Q={fit.Q_eV:.2f} eV  R2={fit.r2:.4f}")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel(r"$D^*_i$  (m$^2$/s)", fontsize=12)
    ax.set_title(f"Tracer diffusion - per element  {comp_label}", fontsize=11)
    ax.text(0.97, 0.04, formula_el, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "D2_vs_invT.png", dpi=150)
    plt.close(fig)

    # ── plot 2: D*_total + per-element ───────────────────────────────────────
    formula_tot = (
        r"$D^*_\mathrm{total} = C_v \cdot \sum_i D_v^{(i)}$"
        "\n" r"$= \sum_i c_i \cdot D^*_i$"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for element in active:
        d = d_star[element]
        valid = (d > 0) & np.isfinite(d)
        ax.semilogy(inv_t[valid], d[valid], "o", color=COLORS[element],
                    ms=4, alpha=0.5, label=element)
    valid_tot = (d_total > 0) & np.isfinite(d_total)
    ax.semilogy(inv_t[valid_tot], d_total[valid_tot], "s",
                color="black", ms=6, zorder=5,
                label=r"$D^*_\mathrm{total}$")
    if fit_total is not None:
        ax.semilogy(inv_t[valid_tot],
                    np.exp(fit_total.ln_D0 + fit_total.slope * inv_t[valid_tot]),
                    "-", color="black", lw=2,
                    label=f"  Q={fit_total.Q_eV:.2f} eV")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel(r"$D^*$  (m$^2$/s)", fontsize=12)
    ax.set_title(r"Tracer diffusion - $D^*_\mathrm{total}$  " + comp_label,
                 fontsize=11)
    ax.text(0.97, 0.04, formula_tot, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "Dtotal_vs_invT.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved -> %s", out_dir)


def main() -> None:
    """Command-line entry point for tracer diffusion."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Compute tracer diffusion.")
    parser.add_argument("--txt_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--comp", nargs="+", required=True,
                        help="Element fractions, e.g. W:0.25 Mo:0.25 Nb:0.25 Ta:0.25")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    comp: dict[str, float] = {}
    for token in args.comp:
        element, value = token.split(":")
        comp[element] = float(value)
    run(comp, Path(args.txt_dir), Path(args.out_dir), args.label)


if __name__ == "__main__":
    main()