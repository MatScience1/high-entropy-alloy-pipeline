"""Stage 7 - equilibrium vacancy concentration Cv(T).

Interface (called by ``run_pipeline.py``)
-----------------------------------------
``run(Ef, C0, T_grid, txt_dir, comp_label)``

Physics
-------
Arrhenius model (Shewmon 1963):

.. math::

    C_v(T) = C_0 \\exp\\!\\left(-\\frac{E_f}{k_B T}\\right),

where

* :math:`E_f` is the mean vacancy formation energy [eV] from
  ``constants_all.csv``,
* :math:`C_0 = \\exp(S_f / k_B)` is the formation-entropy prefactor, with
  :math:`S_f` expressed in units of :math:`k_B`,
* :math:`k_B = 8.617333 \\times 10^{-5}` eV K^-1.

Output
------
``results/<comp_id>/txt/Cv.txt``
``results/<comp_id>/plots/cv_vs_invT_<label>.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import KB_EV  # noqa: E402
from logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

# Reference data from Starikov 2024 Fig. 7 (WMoNb), used for the comparison
# plot only.
_PAPER_INV_T = np.array([3.152866e-4, 3.305732e-4, 3.369427e-4, 3.592357e-4])
_PAPER_CV = np.array([3.840746e-3, 1.930698e-3, 1.765035e-3, 6.385498e-4])
_EF_WMONB = 2.9     # WMoNb formation energy from the Fig. 7 caption [eV]


def compute_cv_curve(T_arr: np.ndarray, Ef: float, C0: float) -> np.ndarray:
    """Return the vacancy concentration for each temperature in ``T_arr``.

    Parameters
    ----------
    T_arr : numpy.ndarray
        Temperatures [K].
    Ef : float
        Vacancy formation energy [eV].
    C0 : float
        Formation-entropy prefactor.

    Returns
    -------
    numpy.ndarray
        ``Cv = C0 * exp(-Ef / (kB T))``.
    """
    return C0 * np.exp(-Ef / (KB_EV * T_arr))


def run(
    Ef: float,
    C0: float,
    sim_temps: list[float],
    out_dir: Path,
    comp_label: str = "",
) -> None:
    """Compute Cv, write the text file, and produce the comparison plot.

    Parameters
    ----------
    Ef : float
        Vacancy formation energy [eV].
    C0 : float
        Formation-entropy prefactor.
    sim_temps : list of float
        Simulation temperatures [K].
    out_dir : pathlib.Path
        Output directory for the text file and plot.
    comp_label : str, optional
        Composition label used in the plot title and file name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    t_sim = np.array(sim_temps, dtype=float)
    cv_sim = compute_cv_curve(t_sim, Ef, C0)
    inv_sim = 1.0 / t_sim

    txt_path = out_dir / "Cv.txt"
    with open(txt_path, "w") as fout:
        fout.write(f"# C0 = {C0:.6e}   Ef = {Ef:.6f} eV\n")
        fout.write(f"# {'T(K)':>10}  {'1/T(1/K)':>14}  {'Cv':>14}\n")
        for t_val, inv_t, cv in zip(t_sim, inv_sim, cv_sim):
            fout.write(f"  {t_val:>10.1f}  {inv_t:>14.8e}  {cv:>14.8e}\n")
    logger.info("Saved %s", txt_path)

    t_line = np.linspace(max(700, 0.5 * t_sim.min()), t_sim.max() * 1.1, 600)
    inv_line = 1.0 / t_line
    cv_line = compute_cv_curve(t_line, Ef, C0)

    # WMoNb comparison line with C0 fitted to the paper reference.
    c0_wmn = float(np.exp(np.mean(
        np.log(_PAPER_CV / np.exp(-_EF_WMONB / (KB_EV / _PAPER_INV_T)))
    )))
    cv_wmn = c0_wmn * np.exp(-_EF_WMONB / (KB_EV * t_line))

    formula = (
        r"$C_v = C_0 \cdot \exp\!\left(-\dfrac{E_f}{k_B T}\right)$"
        "\n" + r"$C_0 = \exp(S_f/k_B)$"
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(inv_line, cv_line, color="tomato", lw=1.8, ls="--",
                label=f"{comp_label}  $E_f$={Ef:.4f} eV")
    ax.semilogy(inv_line, cv_wmn, color="goldenrod", lw=1.8, ls="-.",
                label=f"WMoNb ref  $E_f$={_EF_WMONB} eV")
    ax.semilogy(inv_sim, cv_sim, "o", color="tomato", ms=4)
    ax.semilogy(_PAPER_INV_T, _PAPER_CV, "D", color="goldenrod", ms=8,
                markeredgecolor="saddlebrown", zorder=5,
                label="Fig. 7 WMoNb MD (Starikov 2024)")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel("$C_v$", fontsize=12)
    ax.set_title(f"Equilibrium vacancy concentration  {comp_label}", fontsize=12)
    ax.text(0.97, 0.04, formula, transform=ax.transAxes, fontsize=9.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    axins = inset_axes(ax, width="40%", height="40%", loc="lower left",
                       bbox_to_anchor=(0.05, 0.08, 1, 1),
                       bbox_transform=ax.transAxes)
    x1, x2 = 2.85e-4, 3.75e-4
    mask_line = (inv_line >= x1) & (inv_line <= x2)
    mask_sim = (inv_sim >= x1) & (inv_sim <= x2)
    axins.semilogy(inv_line[mask_line], cv_line[mask_line],
                   color="tomato", lw=1.8, ls="--")
    axins.semilogy(inv_line[mask_line], cv_wmn[mask_line],
                   color="goldenrod", lw=1.8, ls="-.")
    if mask_sim.any():
        axins.semilogy(inv_sim[mask_sim], cv_sim[mask_sim],
                       "o", color="tomato", ms=4)
    axins.semilogy(_PAPER_INV_T, _PAPER_CV, "D", color="goldenrod", ms=7,
                   markeredgecolor="saddlebrown")
    all_y = np.concatenate([cv_line[mask_line], cv_wmn[mask_line], _PAPER_CV])
    axins.set_xlim(x1, x2)
    axins.set_ylim(all_y.min() * 0.3, all_y.max() * 3)
    axins.set_title("zoom: Starikov 2024 region", fontsize=7)
    axins.tick_params(labelsize=7)
    axins.grid(True, which="both", ls=":", alpha=0.35)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.55", lw=0.8)

    fig.tight_layout()
    safe_label = comp_label.replace(" ", "_").replace(",", "") or "alloy"
    png_path = out_dir / f"cv_vs_invT_{safe_label}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", png_path)


def main() -> None:
    """Command-line entry point for the Cv calculation."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Plot Cv vs 1/T for one composition."
    )
    parser.add_argument("--Ef", type=float, required=True, help="Ef [eV].")
    parser.add_argument("--C0", type=float, required=True, help="Prefactor C0.")
    parser.add_argument("--temps", nargs="+", type=float, required=True,
                        help="Simulation temperatures [K].")
    parser.add_argument("--out_dir", default="../results/plots")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    run(args.Ef, args.C0, args.temps, Path(args.out_dir), args.label)


if __name__ == "__main__":
    main()