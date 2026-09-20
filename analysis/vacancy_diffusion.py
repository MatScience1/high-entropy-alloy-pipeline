"""Stage 6 - vacancy diffusion coefficient Dv(T) from the vacancy MSD.

Interface (called by ``run_pipeline.py``)
-----------------------------------------
``process(sim_base, res_base, txt_dir, plot_dir, tm_override)``

``tm_override`` maps element to the alloy melting temperature used for the
homologous-temperature axis.  If ``None``, the elemental reference values from
:data:`config.TM` are used.

Physics
-------
The vacancy diffusion coefficient is extracted from the MSD of the vacancy
site centre of mass:

.. math::

    D_v(T) = \\lim_{t \\to \\infty} \\frac{\\mathrm{MSD}_{\\text{vac}}(t)}{6t}.

LAMMPS reports the *average* atomic MSD, and the system contains exactly one
vacancy, so the vacancy MSD is the total atomic MSD multiplied by the number
of atoms :math:`N`:

.. math::

    D_v = \\frac{\\text{slope} \\cdot N}{6} \\times 10^{-8}
        \\quad [\\text{m}^2\\,\\text{s}^{-1}].

The Arrhenius fit is

.. math::

    \\ln D_v = \\ln D_{v,0} - \\frac{Q_v}{k_B T},

where :math:`Q_v` is the vacancy migration barrier and :math:`D_{v,0}` the
pre-exponential factor.

Output
------
``results/<comp_id>/txt/Dv.txt``
``results/<comp_id>/plots/Dv_total.png``, ``Dv_elements.png``,
``Dv_homologous.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from analysis.arrhenius import fit_arrhenius_wls  # noqa: E402
from config import CONV_A2PS_TO_M2S, TM  # noqa: E402
from logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

ELEMENTS_ALL: list[str] = ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
COLORS: dict[str, str] = {
    "W": "steelblue", "Mo": "darkorange", "Nb": "forestgreen",
    "Zr": "goldenrod", "Ti": "orchid", "Ta": "tomato",
}
T_MIN: int = 900
T_MAX: int = 3100

_ATOM_COUNT_CACHE: dict[str, tuple[int, ...]] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  Atom-count reader
# ══════════════════════════════════════════════════════════════════════════════

def count_atoms_by_type(sim_dir: Path) -> tuple[int, ...]:
    """Return ``(N_W, N_Mo, N_Nb, N_Zr, N_Ti, N_Ta, N_total)`` from a log.

    The result is cached per directory.  If parsing fails, the counts fall
    back to an equiatomic 6x6x6 BCC supercell minus one vacancy (431 atoms).

    Parameters
    ----------
    sim_dir : pathlib.Path
        Directory containing ``log.lammps``.

    Returns
    -------
    tuple of int
        Per-element atom counts followed by the total.
    """
    cache_key = str(sim_dir)
    if cache_key in _ATOM_COUNT_CACHE:
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
                    if len(parts) == 2 and parts[1] == "atoms":
                        try:
                            value = int(parts[0])
                            if value > 100:
                                n_total_line = value
                        except ValueError:
                            continue
                    if "atoms of type" in stripped:
                        try:
                            n_by_type[int(parts[-1])] = int(parts[0])
                        except (ValueError, IndexError):
                            continue

    if len(n_by_type) == 6:
        counts = tuple(n_by_type.get(k, 0) for k in range(1, 7))
        result = counts + (sum(counts),)
    else:
        n_total = n_total_line if n_total_line else 431
        base = n_total // 6
        remainder = n_total - 6 * base
        counts_list = [base] * 6
        for i in range(remainder):
            counts_list[i] += 1
        result = tuple(counts_list) + (n_total,)

    _ATOM_COUNT_CACHE[cache_key] = result
    return result


def reset_atom_cache() -> None:
    """Clear the atom-count cache between compositions."""
    _ATOM_COUNT_CACHE.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MSD loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_msd_all(res_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time_ps, msd_all_A2)`` from ``res_dir/msd_all.txt``."""
    data = np.loadtxt(res_dir / "msd_all.txt", comments="#", skiprows=1)
    return data[:, 0], data[:, 1]


def load_msd_solo(res_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``(time_ps, {element: msd_per_atom_A2})`` from ``msd_solo.txt``.

    The column header format is ``time_ps  c_msd_w[4]  c_msd_mo[4]  ...``.
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
        name = col.replace("c_msd_", "").replace("[4]", "")
        upper = name.capitalize()
        if upper in ELEMENTS_ALL:
            msds[upper] = data[:, j]
    return time, msds


# ══════════════════════════════════════════════════════════════════════════════
#  Slope and conversion
# ══════════════════════════════════════════════════════════════════════════════

def linear_slope(t: np.ndarray, msd: np.ndarray) -> float:
    """Return the MSD slope [Angstrom^2/ps], excluding ``t = 0``.

    Parameters
    ----------
    t : numpy.ndarray
        Time [ps].
    msd : numpy.ndarray
        Mean-square displacement [Angstrom^2].

    Returns
    -------
    float
        Least-squares slope, or ``NaN`` if fewer than three points remain.
    """
    mask = t > 0
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(t[mask], msd[mask], 1)
    return float(slope)


def compute_dv(slope_a2_per_ps: float, n_atoms: int) -> float:
    """Convert an MSD slope into a vacancy diffusion coefficient.

    Parameters
    ----------
    slope_a2_per_ps : float
        MSD slope [Angstrom^2 ps^-1].
    n_atoms : int
        Number of atoms in the supercell.

    Returns
    -------
    float
        ``Dv = slope * N / 6`` converted to m^2 s^-1.
    """
    return slope_a2_per_ps * n_atoms / 6.0 * CONV_A2PS_TO_M2S


def _arrhenius_plot(
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    ax,
) -> None:
    """Fit ``ln(y)`` against ``x`` and add the fitted line to ``ax``."""
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 3:
        return
    fit = fit_arrhenius_wls(x[mask], np.log(y[mask]))
    if fit is None:
        return
    ax.plot(x[mask], np.exp(fit.ln_D0 + fit.slope * x[mask]),
            "--", color=color, alpha=0.75,
            label=f"{label} fit  R2={fit.r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main processing function
# ══════════════════════════════════════════════════════════════════════════════

def process(
    sim_base: Path,
    res_base: Path,
    out_txt_dir: Path,
    out_plot_dir: Path,
    tm_override: dict[str, float] | None = None,
) -> None:
    """Collect Dv over all temperatures and write ``Dv.txt`` and plots.

    Parameters
    ----------
    sim_base : pathlib.Path
        Directory containing ``sim_<T>`` subdirectories.
    res_base : pathlib.Path
        Directory containing the parsed MSD files (``sim_<T>`` mirrored).
    out_txt_dir : pathlib.Path
        Output directory for ``Dv.txt``.
    out_plot_dir : pathlib.Path
        Output directory for the plots.
    tm_override : dict of str to float, optional
        Composition-specific melting temperatures for the homologous axis.
    """
    reset_atom_cache()
    tm_vals = tm_override if tm_override else TM

    if not sim_base.exists():
        logger.warning("simulation base not found: %s", sim_base)
        return

    sim_dirs = sorted(
        [d for d in sim_base.iterdir()
         if d.is_dir() and d.name.startswith("sim_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    temperatures: list[float] = []
    dv_all_list: list[float] = []
    dv_elem: dict[str, list[float]] = {e: [] for e in ELEMENTS_ALL}

    for sim_dir in sim_dirs:
        temperature = int(sim_dir.name.split("_")[1])
        if not (T_MIN <= temperature <= T_MAX):
            continue

        res_dir = res_base / sim_dir.name
        counts = count_atoms_by_type(sim_dir)
        n_map = {e: counts[i] for i, e in enumerate(ELEMENTS_ALL)}
        n_total = counts[6]

        try:
            t, msd = load_msd_all(res_dir)
            slope = linear_slope(t, msd)
            dv_all_list.append(compute_dv(slope, n_total))
            temperatures.append(float(temperature))
        except Exception as exc:  # noqa: BLE001 - skip this temperature
            logger.warning("%s/msd_all.txt: %s", sim_dir.name, exc)
            continue

        try:
            t_el, msds_el = load_msd_solo(res_dir)
            for element in ELEMENTS_ALL:
                if element in msds_el:
                    slope_e = linear_slope(t_el, msds_el[element])
                    dv_elem[element].append(compute_dv(slope_e, n_map[element]))
                else:
                    dv_elem[element].append(float("nan"))
        except Exception as exc:  # noqa: BLE001 - record NaNs for this T
            logger.warning("%s/msd_solo.txt: %s", sim_dir.name, exc)
            for element in ELEMENTS_ALL:
                dv_elem[element].append(float("nan"))

    if not temperatures:
        logger.warning("no data collected - check sim/res directories")
        return

    temperatures_arr = np.array(temperatures)
    dv_all_arr = np.array(dv_all_list)
    dv_elem_arr = {e: np.array(dv_elem[e]) for e in ELEMENTS_ALL}
    inv_T = 1.0 / temperatures_arr

    # ── save Dv.txt ──────────────────────────────────────────────────────────
    out_txt_dir.mkdir(parents=True, exist_ok=True)
    elem_header = "  ".join(f"{'Dv_'+e+'(m2/s)':>16}" for e in ELEMENTS_ALL)
    header_str = (
        f"{'T(K)':>8}  {'1/T(1/K)':>12}  {'Dv_all(m2/s)':>16}  {elem_header}"
    )
    with open(out_txt_dir / "Dv.txt", "w") as fout:
        fout.write("# " + header_str + "\n")
        for i, t_val in enumerate(temperatures_arr):
            row = f"  {t_val:>8.0f}  {inv_T[i]:>12.6e}  {dv_all_arr[i]:>16.6e}"
            for element in ELEMENTS_ALL:
                row += f"  {dv_elem_arr[element][i]:>16.6e}"
            fout.write(row + "\n")
    logger.info("Saved Dv.txt -> %s", out_txt_dir)

    # ── plots ─────────────────────────────────────────────────────────────────
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    active = [
        e for e in ELEMENTS_ALL
        if np.any(dv_elem_arr[e] > 0) and not np.all(np.isnan(dv_elem_arr[e]))
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    valid = (dv_all_arr > 0) & ~np.isnan(dv_all_arr)
    ax.semilogy(inv_T[valid], dv_all_arr[valid], "o", color="black",
                label="all atoms", ms=5)
    _arrhenius_plot(inv_T, dv_all_arr, "black", "Total", ax)
    ax.set_xlabel("1/T  (1/K)")
    ax.set_ylabel(r"$D_v$  (m$^2$/s)")
    ax.set_title("Vacancy diffusion coefficient - total")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_total.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for element in active:
        d = dv_elem_arr[element]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(inv_T[valid], d[valid], "o", color=COLORS[element],
                    label=element, ms=5)
        _arrhenius_plot(inv_T, d, COLORS[element], element, ax)
    ax.set_xlabel("1/T  (1/K)")
    ax.set_ylabel(r"$D_v^{(i)}$  (m$^2$/s)")
    ax.set_title("Vacancy diffusion - per element")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_elements.png", dpi=150)
    plt.close(fig)

    # Homologous temperature axis: Tm_alloy / T.  Using the alloy Tm (not a
    # per-element Tm) is physically correct because all species are compared at
    # the same fraction of the alloy melting point.
    alloy_tm = list(tm_vals.values())[0] if tm_vals else 3000.0
    hom_t = alloy_tm / temperatures_arr

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for element in active:
        d = dv_elem_arr[element]
        valid = (d > 0) & ~np.isnan(d)
        ax.semilogy(hom_t[valid], d[valid], "o", color=COLORS[element],
                    label=element, ms=5)
        if valid.sum() > 2:
            _arrhenius_plot(hom_t, d, COLORS[element], element, ax)
    ax.set_xlabel(r"$T_m^{\rm alloy}\,/\,T$")
    ax.set_ylabel(r"$D_v^{(i)}$  (m$^2$/s)")
    ax.set_title(
        r"$D_v$ vs homologous temperature  ($T_m$ = alloy Tm = "
        f"{alloy_tm:.0f} K)"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_plot_dir / "Dv_homologous.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved -> %s", out_plot_dir)


def main() -> None:
    """Command-line entry point for Dv extraction."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Compute Dv from MSD data.")
    parser.add_argument("--sim_base", default="../sim_x")
    parser.add_argument("--res_base", default="../results/sim_x")
    parser.add_argument("--out_txt", default="../results/txt")
    parser.add_argument("--out_plot", default="../results/plots")
    args = parser.parse_args()
    process(
        Path(args.sim_base), Path(args.res_base),
        Path(args.out_txt), Path(args.out_plot),
    )


if __name__ == "__main__":
    main()