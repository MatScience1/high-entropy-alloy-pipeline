"""
analysis/D2.py
==============
Stage 8 — Tracer self-diffusion coefficient D*(T) per element.

Interface (called by run_pipeline.py)
--------------------------------------
    run(comp: dict[str, float], txt_dir: Path,
        plot_dir: Path, comp_label: str) -> None

Physics
-------
Tracer diffusivity from the per-element MSD via the Einstein relation:

    D*_i(T) = lim_{t→∞}  MSD_i(t) / (6t)   [Å² ps⁻¹]

Converted to SI: multiply by CONV_A2PS_TO_M2S = 1 × 10⁻⁸  (= 10⁻²⁰/10⁻¹²).

Arrhenius fit over the T_grid:
    ln D*_i = ln D0_i − Q_i / (kB T)

Note: D*_i is measured directly from atomic MSD and already incorporates
the correlation factor f (≈ 0.727 for BCC).  It is therefore NOT equal to
the uncorrelated vacancy-mechanism diffusivity f·Cv·Dv unless f is divided
out.  Report D*_i directly for comparison with tracer-diffusion experiments.

Output
------
    results/<comp_id>/txt/D2_<elem>_vs_T.txt
    results/<comp_id>/plots/D2_arrhenius_<comp_label>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ELEMENTS_ALL = ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
COLORS = {
    "W":  "steelblue",  "Mo": "darkorange", "Nb": "forestgreen",
    "Zr": "goldenrod",  "Ti": "orchid",     "Ta": "tomato",
}
T_MIN, T_MAX = 900, 3100


# ══════════════════════════════════════════════════════════════════════════════
#  Data loaders
# ══════════════════════════════════════════════════════════════════════════════

def _load_cv(txt_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, Cv) from Cv.txt, filtered to [T_MIN, T_MAX]."""
    data = np.genfromtxt(txt_dir / "Cv.txt", skip_header=2)
    mask = (data[:, 0] >= T_MIN) & (data[:, 0] <= T_MAX)
    return data[mask, 0], data[mask, 2]


def _load_dv(txt_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Return (T, {element: Dv_i}) from Dv.txt.
    Column layout: T  1/T  Dv_all  Dv_W  Dv_Mo  Dv_Nb  Dv_Zr  Dv_Ti  Dv_Ta
    """
    path = txt_dir / "Dv.txt"
    with open(path) as fh:
        header = fh.readline().lstrip("#").strip()
    col_names = header.split()
    data = np.genfromtxt(path, skip_header=1)
    mask = (data[:, 0] >= T_MIN) & (data[:, 0] <= T_MAX)
    data = data[mask]
    T = data[:, 0]
    dv: dict[str, np.ndarray] = {}
    for j, col in enumerate(col_names):
        for e in ELEMENTS_ALL:
            if col.lower() == f"dv_{e.lower()}(m2/s)":
                dv[e] = data[:, j]
    return T, dv


# ══════════════════════════════════════════════════════════════════════════════
#  Arrhenius helper
# ══════════════════════════════════════════════════════════════════════════════

def _arrhenius(inv_T: np.ndarray, y: np.ndarray
               ) -> tuple[float, float, float, float] | None:
    mask = (y > 0) & ~np.isnan(y)
    if mask.sum() < 3:
        return None
    res = linregress(inv_T[mask], np.log(y[mask]))
    Q_eV = -res.slope * 8.617333e-5
    return float(res.intercept), float(res.slope), float(res.rvalue**2), Q_eV


# ══════════════════════════════════════════════════════════════════════════════
#  Main function
# ══════════════════════════════════════════════════════════════════════════════

def run(comp: dict[str, float], txt_dir: Path, out_dir: Path,
        comp_label: str = "") -> None:
    """
    Compute D*_i and D*_total; save txt and plots.

    Parameters
    ----------
    comp       {element: concentration}; elements with c=0 are skipped
    txt_dir    directory containing Cv.txt and Dv.txt
    out_dir    output directory for txt and plots
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    T, Cv = _load_cv(txt_dir)
    T_dv, dv_map = _load_dv(txt_dir)

    if not np.allclose(T, T_dv, atol=0.5):
        raise ValueError("Temperature grids in Cv.txt and Dv.txt do not match")

    inv_T = 1.0 / T

    # Active elements: have concentration > 0 and a Dv column
    active = [e for e in ELEMENTS_ALL
              if comp.get(e, 0.0) > 1e-6 and e in dv_map]

    # Per-element tracer diffusion: D*_i = Cv · Dv_i / c_i
    D: dict[str, np.ndarray] = {}
    for e in active:
        ci = comp[e]
        D[e] = (1.0 / ci) * Cv * dv_map[e]

    # Total: D*_total = Cv · Σ_i Dv_i  (equates to Σ_i c_i D*_i)
    D_total = Cv * sum(dv_map[e] for e in active)

    # ── save txt ──────────────────────────────────────────────────────────────
    header = ("T(K) 1/T(1/K) " +
              " ".join(f"D_{e}(m2/s)" for e in active) +
              " D_total(m2/s)")
    out_array = np.column_stack(
        [T, inv_T] + [D[e] for e in active] + [D_total]
    )
    txt_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_dir / "D2_components.txt", out_array,
               header=header, fmt="%15.6e")
    print(f"[D2] Saved D2_components.txt → {txt_dir}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'Element':>10}  {'D0(m²/s)':>14}  {'Q(eV)':>8}  {'R²':>8}")
    print("  " + "-" * 46)
    for e in active:
        fit = _arrhenius(inv_T, D[e])
        if fit:
            a, _, r2, Q = fit
            print(f"{e:>10}  {np.exp(a):14.4e}  {Q:8.4f}  {r2:8.6f}")
    fit_tot = _arrhenius(inv_T, D_total)
    if fit_tot:
        a, _, r2, Q = fit_tot
        print(f"{'total':>10}  {np.exp(a):14.4e}  {Q:8.4f}  {r2:8.6f}")

    # ── plot 1: per-element D* ────────────────────────────────────────────────
    FORMULA_EL = (
        r"$D^*_i = C_v \cdot D_v^{(i)} / c_i$"
        "\n" r"$c_i$ = composition fraction"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for e in active:
        d = D[e]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(inv_T[valid], d[valid], "o", color=COLORS[e],
                    ms=5, label=e)
        fit = _arrhenius(inv_T, d)
        if fit:
            a, b, r2, Q = fit
            ax.semilogy(inv_T[valid],
                        np.exp(a + b * inv_T[valid]),
                        "--", color=COLORS[e], alpha=0.75,
                        label=f"  Q={Q:.2f} eV  R²={r2:.4f}")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel(r"$D^*_i$  (m²/s)", fontsize=12)
    ax.set_title(f"Tracer diffusion — per element  {comp_label}", fontsize=11)
    ax.text(0.97, 0.04, FORMULA_EL, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "D2_vs_invT.png", dpi=150)
    plt.close(fig)

    # ── plot 2: D*_total + per-element ───────────────────────────────────────
    FORMULA_TOT = (
        r"$D^*_\mathrm{total} = C_v \cdot \sum_i D_v^{(i)}$"
        "\n" r"$= \sum_i c_i \cdot D^*_i$"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for e in active:
        d = D[e]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(inv_T[valid], d[valid], "o", color=COLORS[e],
                    ms=4, alpha=0.5, label=e)
    valid_tot = (D_total > 0) & ~np.isnan(D_total)
    ax.semilogy(inv_T[valid_tot], D_total[valid_tot], "s",
                color="black", ms=6, zorder=5,
                label=r"$D^*_\mathrm{total}$")
    if fit_tot:
        a, b, _, Q = fit_tot
        ax.semilogy(inv_T[valid_tot],
                    np.exp(a + b * inv_T[valid_tot]),
                    "-", color="black", lw=2,
                    label=f"  Q={Q:.2f} eV")
    ax.set_xlabel("1/T  (1/K)", fontsize=12)
    ax.set_ylabel(r"$D^*$  (m²/s)", fontsize=12)
    ax.set_title(r"Tracer diffusion — $D^*_\mathrm{total}$  " + comp_label,
                 fontsize=11)
    ax.text(0.97, 0.04, FORMULA_TOT, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.8))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "Dtotal_vs_invT.png", dpi=150)
    plt.close(fig)

    print(f"[D2] Plots saved → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute tracer diffusion")
    ap.add_argument("--txt_dir",  required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--comp",     nargs="+", required=True,
                    help="Element fractions e.g. W:0.25 Mo:0.25 Nb:0.25 Ta:0.25")
    ap.add_argument("--label",    default="")
    args = ap.parse_args()
    comp = {}
    for token in args.comp:
        e, c = token.split(":")
        comp[e] = float(c)
    run(comp, Path(args.txt_dir), Path(args.out_dir), args.label)

