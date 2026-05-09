"""
analysis/Cv.py
==============
Stage 7 — Equilibrium vacancy concentration Cv(T).

Interface (called by run_pipeline.py)
--------------------------------------
    run(Ef: float, C0: float, T_grid: list[float],
        txt_dir: Path, comp_label: str) -> None

Physics
-------
Arrhenius model (Shewmon 1963):

    Cv(T) = C0 · exp(−Ef / kB T)

where:
    Ef  = mean vacancy formation energy [eV]    (from constants_all.csv)
    C0  = exp(Sf / kB)                          (Sf in kB units)
    kB  = 8.617333 × 10⁻⁵ eV K⁻¹

Output
------
    results/<comp_id>/txt/Cv_vs_T.txt    Cv at each T in T_grid
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
sys.path.insert(0, str(_HERE))
from config import KB_EV

# Reference data from Starikov 2024 Fig. 7 (WMoNb, for comparison plot only)
_PAPER_INV_T = np.array([3.152866e-4, 3.305732e-4, 3.369427e-4, 3.592357e-4])
_PAPER_CV    = np.array([3.840746e-3, 1.930698e-3, 1.765035e-3, 6.385498e-4])
_EF_WMONN    = 2.9     # WMoNb formation energy from Fig. 7 caption [eV]


def compute_cv_curve(T_arr: np.ndarray, Ef: float,
                     C0: float) -> np.ndarray:
    """Return C_v for each temperature in T_arr."""
    return C0 * np.exp(-Ef / (KB_EV * T_arr))


def run(Ef: float, C0: float, sim_temps: list[float],
        out_dir: Path, comp_label: str = "") -> None:
    """
    Compute C_v, save txt, plot.

    Parameters
    ----------
    Ef          Vacancy formation energy [eV]
    C0          Prefactor = exp(Sf)  (dimensionless)
    sim_temps   List of simulation temperatures [K]
    out_dir     Directory to write outputs
    comp_label  String to use in plot title / file names
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    T_sim   = np.array(sim_temps, dtype=float)
    Cv_sim  = compute_cv_curve(T_sim, Ef, C0)
    inv_sim = 1.0 / T_sim

    # Save txt
    txt_path = out_dir / "Cv.txt"
    with open(txt_path, "w") as fout:
        fout.write(f"# C0 = {C0:.6e}   Ef = {Ef:.6f} eV\n")
        fout.write(f"# {'T(K)':>10}  {'1/T(1/K)':>14}  {'Cv':>14}\n")
        for T, iT, cv in zip(T_sim, inv_sim, Cv_sim):
            fout.write(f"  {T:>10.1f}  {iT:>14.8e}  {cv:>14.8e}\n")
    print(f"[Cv] Saved {txt_path}")

    # Smooth curve
    T_line  = np.linspace(max(700, 0.5 * T_sim.min()), T_sim.max() * 1.1, 600)
    iT_line = 1.0 / T_line
    Cv_line = compute_cv_curve(T_line, Ef, C0)

    # WMoNb comparison line (C0 fitted to paper reference)
    C0_wmn = float(np.exp(np.mean(
        np.log(_PAPER_CV / np.exp(-_EF_WMONN / (KB_EV / _PAPER_INV_T)))
    )))
    Cv_wmn  = C0_wmn * np.exp(-_EF_WMONN / (KB_EV * T_line))

    FORMULA = (
        r"$C_v = C_0 \cdot \exp\!\left(-\dfrac{E_f}{k_B T}\right)$"
        "\n" + r"$C_0 = \exp(S_f/k_B)$"
    )
    title_str = f"Equilibrium vacancy concentration  {comp_label}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(iT_line, Cv_line, color="tomato", lw=1.8, ls="--",
                label=f"{comp_label}  $E_f$={Ef:.4f} eV")
    ax.semilogy(iT_line, Cv_wmn,  color="goldenrod", lw=1.8, ls="-.",
                label=f"WMoNb ref  $E_f$={_EF_WMONN} eV")
    ax.semilogy(inv_sim, Cv_sim, "o", color="tomato", ms=4)
    ax.semilogy(_PAPER_INV_T, _PAPER_CV, "D", color="goldenrod", ms=8,
                markeredgecolor="saddlebrown", zorder=5,
                label="Fig. 7 WMoNb MD (Starikov 2024)")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel("$C_v$",      fontsize=12)
    ax.set_title(title_str, fontsize=12)
    ax.text(0.97, 0.04, FORMULA, transform=ax.transAxes, fontsize=9.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Inset zoom around paper reference region
    axins = inset_axes(ax, width="40%", height="40%", loc="lower left",
                       bbox_to_anchor=(0.05, 0.08, 1, 1),
                       bbox_transform=ax.transAxes)
    x1, x2 = 2.85e-4, 3.75e-4
    ml = (iT_line >= x1) & (iT_line <= x2)
    ms = (inv_sim  >= x1) & (inv_sim  <= x2)
    axins.semilogy(iT_line[ml], Cv_line[ml], color="tomato",    lw=1.8, ls="--")
    axins.semilogy(iT_line[ml], Cv_wmn[ml],  color="goldenrod", lw=1.8, ls="-.")
    if ms.any():
        axins.semilogy(inv_sim[ms], Cv_sim[ms], "o", color="tomato", ms=4)
    axins.semilogy(_PAPER_INV_T, _PAPER_CV, "D", color="goldenrod", ms=7,
                   markeredgecolor="saddlebrown")
    all_y = np.concatenate([Cv_line[ml], Cv_wmn[ml], _PAPER_CV])
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
    print(f"[Cv] Saved {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot Cv vs 1/T for one composition")
    ap.add_argument("--Ef",   type=float, required=True, help="Ef [eV]")
    ap.add_argument("--C0",   type=float, required=True, help="prefactor C0")
    ap.add_argument("--temps", nargs="+", type=float, required=True,
                    help="List of simulation temperatures [K]")
    ap.add_argument("--out_dir", default="../results/plots")
    ap.add_argument("--label",   default="")
    args = ap.parse_args()
    run(args.Ef, args.C0, args.temps, Path(args.out_dir), args.label)

