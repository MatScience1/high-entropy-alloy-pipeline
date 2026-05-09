"""
analysis/Dv.py
==============
Stage 6 — Vacancy diffusion coefficient Dv(T) from MSD of the vacancy.

Interface (called by run_pipeline.py)
--------------------------------------
    process(sim_base: Path, res_base: Path,
            txt_dir: Path, plot_dir: Path,
            tm_override: dict[str, float] | None) -> None

    tm_override : {element: Tm}  — alloy Tm for homologous-T axis.
                  If None, falls back to elemental Tm from config.TM.

Physics
-------
The vacancy diffusion coefficient is extracted from the MSD of the
vacancy site centre of mass:

    Dv(T) = lim_{t→∞}  MSD_vac(t) / (6t)

Arrhenius fit:
    ln Dv = ln Dv0 − Qv / (kB T)

where Qv is the vacancy migration barrier and Dv0 the pre-exponential.
Plots: Dv vs 1/T (Arrhenius) and Dv vs T/Tm (homologous temperature).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Allow import from parent dir when run as standalone script
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import CONV_A2PS_TO_M2S, TM

ELEMENTS_ALL  = ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
ELEM_LOWER    = [e.lower() for e in ELEMENTS_ALL]
COLORS        = {
    "W":  "steelblue",  "Mo": "darkorange", "Nb": "forestgreen",
    "Zr": "goldenrod",  "Ti": "orchid",     "Ta": "tomato",
}
T_MIN, T_MAX  = 900, 3100   # temperature filter [K]


# ══════════════════════════════════════════════════════════════════════════════
#  Atom-count reader
# ══════════════════════════════════════════════════════════════════════════════

_ATOM_COUNT_CACHE: dict[str, tuple] | None = None

def count_atoms_by_type(sim_dir: Path) -> tuple[int, ...]:
    """
    Return (N_W, N_Mo, N_Nb, N_Zr, N_Ti, N_Ta, N_total) from log.lammps.
    Result is cached after the first successful read.
    Falls back to equiatomic 6×6×6 BCC - 1 vacancy = 431 atoms if parsing fails.
    """
    global _ATOM_COUNT_CACHE
    cache_key = str(sim_dir)
    if _ATOM_COUNT_CACHE is not None and cache_key in _ATOM_COUNT_CACHE:
        return _ATOM_COUNT_CACHE[cache_key]

    log_path = sim_dir / "log.lammps"
    n_by_type: dict[int, int] = {}
    n_total_line: int | None = None

    if log_path.is_file():
        with open(log_path) as fh:
            for i, line in enumerate(fh):
                if i >= 300:
                    break
                stripped = line.strip()
                if stripped and stripped[0].isdigit():
                    parts = stripped.split()
                    # "1999 atoms"
                    if len(parts) == 2 and parts[1] == "atoms":
                        try:
                            val = int(parts[0])
                            if val > 100:
                                n_total_line = val
                        except ValueError:
                            pass
                    # "500 atoms of type 1"
                    if "atoms of type" in stripped:
                        try:
                            n_by_type[int(parts[-1])] = int(parts[0])
                        except (ValueError, IndexError):
                            pass

    if len(n_by_type) == 6:
        counts = tuple(n_by_type.get(k, 0) for k in range(1, 7))
        result = counts + (sum(counts),)
    else:
        ntot = n_total_line if n_total_line else 431
        base = ntot // 6
        remainder = ntot - 6 * base
        counts_list = [base] * 6
        for i in range(remainder):
            counts_list[i] += 1
        result = tuple(counts_list) + (ntot,)

    if _ATOM_COUNT_CACHE is None:
        _ATOM_COUNT_CACHE = {}
    _ATOM_COUNT_CACHE[cache_key] = result
    return result


def reset_atom_cache() -> None:
    """Reset cache between compositions (different supercells)."""
    global _ATOM_COUNT_CACHE
    _ATOM_COUNT_CACHE = None


# ══════════════════════════════════════════════════════════════════════════════
#  MSD loaders  (fixed: always read from res_dir, not sim_dir)
# ══════════════════════════════════════════════════════════════════════════════

def load_msd_all(res_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ps, msd_all_A2) from res_dir/msd_all.txt."""
    path = res_dir / "msd_all.txt"
    data = np.loadtxt(path, comments="#", skiprows=1)
    return data[:, 0], data[:, 1]


def load_msd_solo(res_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Return (time_ps, {element_upper: msd_per_atom_A2}) from msd_solo.txt.
    Column header format: time_ps  c_msd_w[4]  c_msd_mo[4]  …
    """
    path = res_dir / "msd_solo.txt"
    with open(path) as fh:
        header_line = fh.readline().lstrip("#").strip()
    col_names = header_line.split()
    data = np.loadtxt(path, comments="#", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[:, 0]
    msds: dict[str, np.ndarray] = {}
    for j, col in enumerate(col_names[1:], start=1):
        # e.g. "c_msd_mo[4]" → "Mo"
        m = col.replace("c_msd_", "").replace("[4]", "")
        upper = m.capitalize()
        if upper in ELEMENTS_ALL:
            msds[upper] = data[:, j]
    return time, msds


# ══════════════════════════════════════════════════════════════════════════════
#  Linear slope
# ══════════════════════════════════════════════════════════════════════════════

def linear_slope(t: np.ndarray, msd: np.ndarray) -> float:
    """Slope [Å²/ps] from linear regression, excluding t=0."""
    mask = t > 0
    if mask.sum() < 3:
        return float("nan")
    slope, *_ = linregress(t[mask], msd[mask])
    return float(slope)


def compute_dv(slope_A2_per_ps: float, N: int) -> float:
    """
    Dv = slope · N / 6  [m²/s]
    Applies the Smirnova 2015 / Starikov 2024 formula with unit conversion.
    """
    return slope_A2_per_ps * N / 6.0 * CONV_A2PS_TO_M2S


# ══════════════════════════════════════════════════════════════════════════════
#  Arrhenius fit helper
# ══════════════════════════════════════════════════════════════════════════════

def arrhenius_fit(x: np.ndarray, y: np.ndarray,
                  color: str, label: str, ax) -> tuple | None:
    """Fit log(y) vs x with linregress; add dashed line to ax."""
    mask = (y > 0) & ~np.isnan(y)
    if mask.sum() < 3:
        return None
    res = linregress(x[mask], np.log(y[mask]))
    ax.plot(x[mask], np.exp(res.intercept + res.slope * x[mask]),
            "--", color=color, alpha=0.75,
            label=f"{label} fit  R²={res.rvalue**2:.4f}")
    Q_eV = -res.slope * 8.617333e-5
    return res.intercept, res.slope, res.rvalue**2, Q_eV


# ══════════════════════════════════════════════════════════════════════════════
#  Main processing function
# ══════════════════════════════════════════════════════════════════════════════

def process(sim_base: Path, res_base: Path,
            out_txt_dir: Path, out_plot_dir: Path,
            tm_override: dict[str, float] | None = None) -> None:
    """
    Walk sim_*/res_* paired directories; collect Dv; write Dv.txt and plots.
    tm_override: use these Tm values for homologous-T plot (composition-specific).
    """
    reset_atom_cache()
    tm_vals = tm_override if tm_override else TM

    sim_dirs = sorted(
        [d for d in sim_base.iterdir()
         if d.is_dir() and d.name.startswith("sim_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    temperatures: list[float] = []
    dv_all_list:  list[float] = []
    dv_elem: dict[str, list[float]] = {e: [] for e in ELEMENTS_ALL}

    for sd in sim_dirs:
        T = int(sd.name.split("_")[1])
        if not (T_MIN <= T <= T_MAX):
            continue

        res_dir = res_base / sd.name
        sim_dir = sd

        counts = count_atoms_by_type(sim_dir)
        # counts = (N_W, N_Mo, N_Nb, N_Zr, N_Ti, N_Ta, N_total)
        n_map = {e: counts[i] for i, e in enumerate(ELEMENTS_ALL)}
        n_total = counts[6]

        # --- total Dv from all-atoms MSD ---
        try:
            t, msd = load_msd_all(res_dir)
            slope = linear_slope(t, msd)
            dv_all_list.append(compute_dv(slope, n_total))
            temperatures.append(float(T))
        except Exception as exc:
            print(f"  [WARN] {sd.name}/msd_all.txt: {exc}")
            continue

        # --- per-element Dv from msd_solo.txt ---
        try:
            t_el, msds_el = load_msd_solo(res_dir)
            for e in ELEMENTS_ALL:
                if e in msds_el:
                    slope_e = linear_slope(t_el, msds_el[e])
                    dv_elem[e].append(compute_dv(slope_e, n_map[e]))
                else:
                    dv_elem[e].append(float("nan"))
        except Exception as exc:
            print(f"  [WARN] {sd.name}/msd_solo.txt: {exc}")
            for e in ELEMENTS_ALL:
                dv_elem[e].append(float("nan"))

    if not temperatures:
        print("[Dv] No data collected — check sim/res directories.")
        return

    temperatures_arr = np.array(temperatures)
    dv_all_arr = np.array(dv_all_list)
    dv_elem_arr = {e: np.array(dv_elem[e]) for e in ELEMENTS_ALL}
    inv_T = 1.0 / temperatures_arr

    # ── save Dv.txt ──────────────────────────────────────────────────────────
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    elem_header = "  ".join(f"{'Dv_'+e+'(m2/s)':>16}" for e in ELEMENTS_ALL)
    header_str  = f"{'T(K)':>8}  {'1/T(1/K)':>12}  {'Dv_all(m2/s)':>16}  {elem_header}"
    with open(out_txt_dir / "Dv.txt", "w") as fout:
        fout.write("# " + header_str + "\n")
        for i, T_val in enumerate(temperatures_arr):
            row = f"  {T_val:>8.0f}  {inv_T[i]:>12.6e}  {dv_all_arr[i]:>16.6e}"
            for e in ELEMENTS_ALL:
                row += f"  {dv_elem_arr[e][i]:>16.6e}"
            fout.write(row + "\n")
    print(f"[Dv] Saved Dv.txt → {out_txt_dir}")

    # ── plots ─────────────────────────────────────────────────────────────────
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    _active = [e for e in ELEMENTS_ALL
               if np.any(dv_elem_arr[e] > 0) and not np.all(np.isnan(dv_elem_arr[e]))]

    # Plot 1: Dv_total vs 1/T
    fig, ax = plt.subplots(figsize=(6, 4.5))
    valid = (dv_all_arr > 0) & ~np.isnan(dv_all_arr)
    ax.semilogy(inv_T[valid], dv_all_arr[valid], "o", color="black",
                label="all atoms", ms=5)
    arrhenius_fit(inv_T, dv_all_arr, "black", "Total", ax)
    ax.set_xlabel("1/T  (1/K)")
    ax.set_ylabel(r"$D_v$  (m²/s)")
    ax.set_title("Vacancy diffusion coefficient — total")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_total.png", dpi=150)
    plt.close(fig)

    # Plot 2: per-element vs 1/T
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for e in _active:
        d = dv_elem_arr[e]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(inv_T[valid], d[valid], "o", color=COLORS[e],
                    label=e, ms=5)
        arrhenius_fit(inv_T, d, COLORS[e], e, ax)
    ax.set_xlabel("1/T  (1/K)")
    ax.set_ylabel(r"$D_v^{(i)}$  (m²/s)")
    ax.set_title("Vacancy diffusion — per element")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_elements.png", dpi=150)
    plt.close(fig)

    # Plot 3: vs homologous temperature Tm_alloy / T
    # tm_vals is {element: alloy_Tm} — all values are equal to the alloy Tm.
    # Using the alloy Tm (not per-element) is physically correct: we compare
    # diffusivity at the same fraction of the alloy melting point.
    # Source: Starikov 2024 Fig. 3 right panel; Zhang et al. Acta Mater. 2022 Fig. 6b.
    alloy_Tm = list(tm_vals.values())[0] if tm_vals else 3000.0
    hom_T = alloy_Tm / temperatures_arr   # Tm_alloy / T for every simulated T

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for e in _active:
        d = dv_elem_arr[e]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(hom_T[valid], d[valid], "o", color=COLORS[e],
                    label=e, ms=5)
        if valid.sum() > 2:
            arrhenius_fit(hom_T, d, COLORS[e], e, ax)
    ax.set_xlabel(r"$T_m^{\rm alloy}\,/\,T$")
    ax.set_ylabel(r"$D_v^{(i)}$  (m²/s)")
    ax.set_title(r"$D_v$ vs homologous temperature  ($T_m$ = alloy Tm = "
                 f"{alloy_Tm:.0f} K)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_homologous.png", dpi=150)
    plt.close(fig)

    print(f"[Dv] Plots saved → {out_plot_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute Dv from MSD data")
    ap.add_argument("--sim_base",   default="../sim_x")
    ap.add_argument("--res_base",   default="../results/sim_x")
    ap.add_argument("--out_txt",    default="../results/txt")
    ap.add_argument("--out_plot",   default="../results/plots")
    args = ap.parse_args()
    process(
        Path(args.sim_base), Path(args.res_base),
        Path(args.out_txt),  Path(args.out_plot),
    )

