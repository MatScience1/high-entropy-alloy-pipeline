"""
analysis/msd.py
===============
Stage 5 — Parse mean-square displacement from LAMMPS thermo output.

Interface (called by run_pipeline.py)
--------------------------------------
    run_all(sim_base: Path, res_base: Path) -> None

    sim_base : runs/<comp_id>/              (contains sim_<T>/ subdirectories)
    res_base : results/<comp_id>/sim_x/     (output destination)

Output files written per temperature
-------------------------------------
    results/<comp_id>/sim_x/<T>/msd_all.txt       MSD of all atoms vs time
    results/<comp_id>/sim_x/<T>/msd_<elem>.txt    MSD per element

Physics
-------
The LAMMPS compute msd produces:
    MSD(t) = <|r(t) - r(0)|²>   [Å²]

where r(t) is the unwrapped position at time t.
D* is extracted in analysis/D2.py via the Einstein relation:
    D* = lim_{t→∞}  MSD(t) / (6t)     [Å² ps⁻¹ → m² s⁻¹]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress

# Default element set and display properties
ELEMENTS_ALL = ["w", "mo", "nb", "zr", "ti", "ta"]
ELEMENT_LABELS = {"w": "W", "mo": "Mo", "nb": "Nb",
                  "zr": "Zr", "ti": "Ti", "ta": "Ta"}
COLORS = {
    "w":  "#e05c5c", "mo": "#5c8fe0", "nb": "#5cc47a",
    "zr": "#e0a040", "ti": "#9b5ce0", "ta": "#ff0084",
}

TIMESTEP_PS = 0.0005  # ps per step


# ══════════════════════════════════════════════════════════════════════════════
#  Log parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_log(log_path: Path) -> dict[str, np.ndarray]:
    """
    Find the MD section containing c_msd_all[4] and parse numeric rows.
    Returns dict {column_name: np.array}.
    """
    with open(log_path, "r") as fh:
        lines = fh.readlines()

    # Find the LAST header line that contains c_msd_all[4]
    # (there may be multiple runs in one log)
    header_idx = None
    for i, line in enumerate(lines):
        if "c_msd_all[1]" in line and "c_msd_all[4]" in line:
            header_idx = i

    if header_idx is None:
        raise ValueError(f"No MSD header found in {log_path}")

    headers = lines[header_idx].split()
    data: dict[str, list[float]] = {h: [] for h in headers}

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # Stop at any non-numeric first token
        try:
            int(tokens[0])
        except (ValueError, IndexError):
            break
        if len(tokens) != len(headers):
            break
        for h, v in zip(headers, tokens):
            data[h].append(float(v))

    if not data.get("Step"):
        raise ValueError(f"No data rows after MSD header in {log_path}")

    return {k: np.array(v) for k, v in data.items()}


def active_elements_in_log(data: dict[str, np.ndarray]) -> list[str]:
    """Return lowercase element names whose MSD column is present in data."""
    return [
        el for el in ELEMENTS_ALL
        if f"c_msd_{el}[4]" in data
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_msd_all(out_path: Path, time: np.ndarray,
                  msd: np.ndarray) -> None:
    header = "time_ps  c_msd_all[4]"
    np.savetxt(out_path, np.column_stack([time, msd]),
               header=header, fmt="%.8e", comments="")


def write_msd_solo(out_path: Path, time: np.ndarray,
                   msds: dict[str, np.ndarray]) -> None:
    """Write per-element MSD file; include only elements present in msds."""
    elements = list(msds.keys())
    col_names = [f"c_msd_{el}[4]" for el in elements]
    header = "time_ps  " + "  ".join(col_names)
    matrix = np.column_stack([time] + [msds[el] for el in elements])
    np.savetxt(out_path, matrix, header=header, fmt="%.8e", comments="")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _linear_fit(x: np.ndarray, y: np.ndarray
                ) -> tuple[float, float, np.ndarray]:
    """Return (slope, intercept, y_fitted)."""
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0], coeffs[1], np.polyval(coeffs, x)


def plot_msd_all(time: np.ndarray, msd: np.ndarray,
                 temperature: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time, msd, color="#aaaaaa", lw=0.8, alpha=0.5, label="raw")
    slope, _, y_fit = _linear_fit(time, msd)
    ax.plot(time, y_fit, color="#e05c5c", lw=1.5, ls="--",
            label=f"fit  slope={slope:.3e} Å²/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\AA^2$)")
    ax.set_title(f"T = {temperature} K — all atoms")
    ax.legend(fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_msd_solo(time: np.ndarray, msds: dict[str, np.ndarray],
                  temperature: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for el, msd in msds.items():
        color = COLORS.get(el, "black")
        slope, _, y_fit = _linear_fit(time, msd)
        ax.plot(time, msd, color=color, lw=0.8, alpha=0.4)
        ax.plot(time, y_fit, color=color, lw=1.5, ls="--",
                label=f"{ELEMENT_LABELS.get(el, el)}  {slope:.3e} Å²/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\AA^2$)")
    ax.set_title(f"T = {temperature} K — per element")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-directory processor
# ══════════════════════════════════════════════════════════════════════════════

def process_dir(sim_dir: Path, res_dir: Path) -> bool:
    """
    Parse sim_dir/log.lammps; write MSD txt files and plots to res_dir.
    Returns True on success, False on skip/error.
    """
    log_path = sim_dir / "log.lammps"
    if not log_path.exists():
        print(f"  [skip] no log.lammps in {sim_dir}")
        return False

    temperature = sim_dir.name.split("_")[1]
    print(f"  [msd] T={temperature} K ...")

    try:
        data = parse_log(log_path)
    except Exception as exc:
        print(f"  [WARN] parse failed: {exc}")
        return False

    steps = data["Step"]
    time = (steps - steps[0]) * TIMESTEP_PS   # shift so t=0 at run start

    # --- all-atoms MSD --------------------------------------------------------
    if "c_msd_all[4]" not in data:
        print(f"  [WARN] c_msd_all[4] not found in {log_path}")
        return False

    msd_all = data["c_msd_all[4]"]
    res_dir.mkdir(parents=True, exist_ok=True)
    write_msd_all(res_dir / "msd_all.txt", time, msd_all)
    plot_msd_all(time, msd_all, temperature, res_dir / "msd_all.png")

    # --- per-element MSD ------------------------------------------------------
    active = active_elements_in_log(data)
    if active:
        msds = {el: data[f"c_msd_{el}[4]"] for el in active}
        write_msd_solo(res_dir / "msd_solo.txt", time, msds)
        plot_msd_solo(time, msds, temperature, res_dir / "msd_solo.png")
    else:
        print(f"  [WARN] no per-element MSD columns found")

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Batch runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all(sim_base: Path, res_base: Path) -> None:
    """
    Walk all sim_* subdirectories under sim_base.
    Mirror structure into res_base for outputs.
    """
    sim_dirs = sorted(
        [d for d in sim_base.iterdir()
         if d.is_dir() and re.match(r"sim_\d+$", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not sim_dirs:
        print(f"[msd] No sim_* directories found under {sim_base}")
        return

    print(f"[msd] Found {len(sim_dirs)} directories under {sim_base}")
    ok, fail = 0, 0
    for d in sim_dirs:
        T = d.name.split("_")[1]
        res_dir = res_base / f"sim_{T}"
        if process_dir(d, res_dir):
            ok += 1
        else:
            fail += 1
    print(f"[msd] Done: {ok} processed, {fail} skipped/failed")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse LAMMPS MSD from log files")
    ap.add_argument("--sim_base", default="../sim_x",
                    help="Directory containing sim_T sub-directories")
    ap.add_argument("--res_base", default="../results/sim_x",
                    help="Output directory (mirrored structure)")
    args = ap.parse_args()
    run_all(Path(args.sim_base), Path(args.res_base))

