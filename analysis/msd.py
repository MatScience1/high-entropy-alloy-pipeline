"""Stage 5 - parse mean-square displacement from LAMMPS thermo output.

Interface (called by ``run_pipeline.py``)
-----------------------------------------
``run_all(sim_base, res_base)``

* ``sim_base``: ``runs/<comp_id>/`` (contains ``sim_<T>/`` subdirectories)
* ``res_base``: ``results/<comp_id>/sim_x/`` (output destination)

Output files written per temperature
------------------------------------
``results/<comp_id>/sim_x/<T>/msd_all.txt``
    MSD of all atoms versus time.
``results/<comp_id>/sim_x/<T>/msd_solo.txt``
    MSD per element.

Physics
-------
The LAMMPS ``compute msd`` produces

.. math::

    \\mathrm{MSD}(t) = \\langle |\\mathbf{r}(t) - \\mathbf{r}(0)|^2 \\rangle
    \\quad [\\text{\\AA}^2],

where :math:`\\mathbf{r}(t)` is the unwrapped position at time :math:`t`.
The tracer diffusivity is extracted in
:mod:`analysis.tracer_diffusion` via the Einstein relation

.. math::

    D^* = \\lim_{t \\to \\infty} \\frac{\\mathrm{MSD}(t)}{6t}.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from analysis.logparse import LogParseError, read_lines
from config import TIMESTEP_PS
from logging_config import configure_logging, get_logger

logger = get_logger(__name__)

# Default element set and display properties.
ELEMENTS_ALL: list[str] = ["w", "mo", "nb", "zr", "ti", "ta"]
ELEMENT_LABELS: dict[str, str] = {
    "w": "W", "mo": "Mo", "nb": "Nb", "zr": "Zr", "ti": "Ti", "ta": "Ta",
}
COLORS: dict[str, str] = {
    "w": "#e05c5c", "mo": "#5c8fe0", "nb": "#5cc47a",
    "zr": "#e0a040", "ti": "#9b5ce0", "ta": "#ff0084",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Log parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_log(log_path: Path) -> dict[str, np.ndarray]:
    """Parse the MD section containing ``c_msd_all[4]`` from a LAMMPS log.

    The parser locates the last thermo header that contains both
    ``c_msd_all[1]`` and ``c_msd_all[4]`` (a log may contain several runs) and
    reads the numeric rows that follow.  Malformed or truncated rows are
    skipped with a warning; parsing stops at the first non-numeric line
    (typically ``Loop time``).

    Parameters
    ----------
    log_path : pathlib.Path
        Path to ``log.lammps``.

    Returns
    -------
    dict of str to numpy.ndarray
        Column name to data array.

    Raises
    ------
    LogParseError
        If the file is missing, has no MSD header, or contains no data rows.
    """
    lines = read_lines(log_path)

    header_idx: int | None = None
    for i, line in enumerate(lines):
        if "c_msd_all[1]" in line and "c_msd_all[4]" in line:
            header_idx = i
    if header_idx is None:
        raise LogParseError(f"no MSD header found in {log_path}")

    headers = lines[header_idx].split()
    data: dict[str, list[float]] = {h: [] for h in headers}
    n_skipped = 0

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            int(tokens[0])
        except (ValueError, IndexError):
            break   # end of the thermo block (e.g. "Loop time")
        if len(tokens) != len(headers):
            n_skipped += 1
            continue
        try:
            values = [float(v) for v in tokens]
        except ValueError:
            n_skipped += 1
            continue
        for name, value in zip(headers, values):
            data[name].append(value)

    if n_skipped:
        logger.warning(
            "Skipped %d malformed row(s) in %s (truncated log?)",
            n_skipped, log_path,
        )

    if not data.get("Step"):
        raise LogParseError(f"no data rows after MSD header in {log_path}")

    return {k: np.array(v) for k, v in data.items()}


def active_elements_in_log(data: dict[str, np.ndarray]) -> list[str]:
    """Return the lowercase elements whose MSD column is present in ``data``.

    Parameters
    ----------
    data : dict of str to numpy.ndarray
        Parsed log columns.

    Returns
    -------
    list of str
        Lowercase element symbols with a ``c_msd_<el>[4]`` column.
    """
    return [el for el in ELEMENTS_ALL if f"c_msd_{el}[4]" in data]


# ══════════════════════════════════════════════════════════════════════════════
#  Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_msd_all(out_path: Path, time: np.ndarray, msd: np.ndarray) -> None:
    """Write the all-atoms MSD to a two-column text file."""
    np.savetxt(
        out_path, np.column_stack([time, msd]),
        header="time_ps  c_msd_all[4]", fmt="%.8e", comments="",
    )


def write_msd_solo(
    out_path: Path,
    time: np.ndarray,
    msds: dict[str, np.ndarray],
) -> None:
    """Write the per-element MSD file, including only elements present."""
    elements = list(msds.keys())
    col_names = [f"c_msd_{el}[4]" for el in elements]
    header = "time_ps  " + "  ".join(col_names)
    matrix = np.column_stack([time] + [msds[el] for el in elements])
    np.savetxt(out_path, matrix, header=header, fmt="%.8e", comments="")


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return ``(slope, intercept, y_fitted)`` for a first-order polynomial."""
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0]), float(coeffs[1]), np.polyval(coeffs, x)


def plot_msd_all(
    time: np.ndarray,
    msd: np.ndarray,
    temperature: str,
    out_path: Path,
) -> None:
    """Plot the all-atoms MSD with a linear fit overlay."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time, msd, color="#aaaaaa", lw=0.8, alpha=0.5, label="raw")
    slope, _, y_fit = _linear_fit(time, msd)
    ax.plot(time, y_fit, color="#e05c5c", lw=1.5, ls="--",
            label=f"fit  slope={slope:.3e} A^2/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\AA^2$)")
    ax.set_title(f"T = {temperature} K - all atoms")
    ax.legend(fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_msd_solo(
    time: np.ndarray,
    msds: dict[str, np.ndarray],
    temperature: str,
    out_path: Path,
) -> None:
    """Plot the per-element MSD with linear fit overlays."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for element, msd in msds.items():
        color = COLORS.get(element, "black")
        slope, _, y_fit = _linear_fit(time, msd)
        ax.plot(time, msd, color=color, lw=0.8, alpha=0.4)
        ax.plot(time, y_fit, color=color, lw=1.5, ls="--",
                label=f"{ELEMENT_LABELS.get(element, element)}  {slope:.3e} A^2/ps")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"MSD ($\AA^2$)")
    ax.set_title(f"T = {temperature} K - per element")
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
    """Parse ``sim_dir/log.lammps`` and write MSD files and plots to ``res_dir``.

    Parameters
    ----------
    sim_dir : pathlib.Path
        Directory containing ``log.lammps``.
    res_dir : pathlib.Path
        Output directory for the MSD text files and plots.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the run was skipped or failed.
    """
    log_path = sim_dir / "log.lammps"
    if not log_path.exists():
        logger.warning("no log.lammps in %s - skipped", sim_dir)
        return False

    temperature = sim_dir.name.split("_")[1]
    logger.info("T=%s K ...", temperature)

    try:
        data = parse_log(log_path)
    except LogParseError as exc:
        logger.warning("parse failed: %s", exc)
        return False

    steps = data["Step"]
    time = (steps - steps[0]) * TIMESTEP_PS   # shift so t=0 at run start

    if "c_msd_all[4]" not in data:
        logger.warning("c_msd_all[4] not found in %s", log_path)
        return False

    msd_all = data["c_msd_all[4]"]
    res_dir.mkdir(parents=True, exist_ok=True)
    write_msd_all(res_dir / "msd_all.txt", time, msd_all)
    plot_msd_all(time, msd_all, temperature, res_dir / "msd_all.png")

    active = active_elements_in_log(data)
    if active:
        msds = {el: data[f"c_msd_{el}[4]"] for el in active}
        write_msd_solo(res_dir / "msd_solo.txt", time, msds)
        plot_msd_solo(time, msds, temperature, res_dir / "msd_solo.png")
    else:
        logger.warning("no per-element MSD columns found in %s", log_path)

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Batch runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all(sim_base: Path, res_base: Path) -> None:
    """Process every ``sim_*`` subdirectory under ``sim_base``.

    Parameters
    ----------
    sim_base : pathlib.Path
        Directory containing ``sim_<T>`` subdirectories.
    res_base : pathlib.Path
        Output directory; the ``sim_<T>`` structure is mirrored.
    """
    if not sim_base.exists():
        logger.warning("simulation base not found: %s", sim_base)
        return

    sim_dirs = sorted(
        [d for d in sim_base.iterdir()
         if d.is_dir() and re.match(r"sim_\d+$", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not sim_dirs:
        logger.warning("no sim_* directories found under %s", sim_base)
        return

    logger.info("Found %d directories under %s", len(sim_dirs), sim_base)
    ok = 0
    failed = 0
    for sim_dir in sim_dirs:
        temperature = sim_dir.name.split("_")[1]
        res_dir = res_base / f"sim_{temperature}"
        if process_dir(sim_dir, res_dir):
            ok += 1
        else:
            failed += 1
    logger.info("Done: %d processed, %d skipped/failed", ok, failed)


def main() -> None:
    """Command-line entry point for MSD extraction."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Parse LAMMPS MSD from log files.")
    parser.add_argument("--sim_base", default="../sim_x",
                        help="Directory containing sim_T sub-directories.")
    parser.add_argument("--res_base", default="../results/sim_x",
                        help="Output directory (mirrored structure).")
    args = parser.parse_args()
    run_all(Path(args.sim_base), Path(args.res_base))


if __name__ == "__main__":
    main()
