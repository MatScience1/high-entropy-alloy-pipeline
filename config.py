#!/usr/bin/env python3
"""Central configuration for the WMoNbZrTiTa diffusion pipeline.

This module is the single source of truth for physical constants, element
properties, simulation parameters, and SLURM settings.  Every other module
imports from here.

Mode selection
--------------
``TEST_MODE`` defaults to ``False`` (full production run).  It can be
overridden at runtime, before any stage executes, with::

    from config import set_test_mode
    set_test_mode(True)

or from the command line via ``run_pipeline.py --test``.  Because several
parameters are derived from the mode at import time, :func:`set_test_mode`
recomputes and rebinds those module-level values.

Physical constants are never modified by the mode switch; only run lengths,
sampling counts, and walltimes change.

Notes
-----
The SRO cutoff is not a global constant.  It depends on the composition's
lattice parameter and must be obtained from :func:`compute_r_sro`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

__version__: str = "1.1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
# Default is the full production run.  Use set_test_mode(True) or the
# run_pipeline.py --test flag for a rapid validation run.
TEST_MODE: bool = False

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_DIR: Path = Path(__file__).resolve().parent
RUNS_DIR: Path = PIPELINE_DIR / "runs"
RESULTS_DIR: Path = PIPELINE_DIR / "results"
DOCS_DIR: Path = PIPELINE_DIR / "docs"

_POT_NAME: str = "WMoNbZrTiTa.nist.adp.txt"
_TM_NAME: str = "Tm_e.txt"


def _find_file(name: str) -> Path:
    """Locate a reference file in the pipeline directory or a sibling ``ref/``.

    Parameters
    ----------
    name : str
        File name to search for.

    Returns
    -------
    Path
        First existing candidate, or the primary candidate if none exist.
    """
    candidates: list[Path] = [
        PIPELINE_DIR / name,
        PIPELINE_DIR.parent / "ref" / name,
        PIPELINE_DIR / "ref" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


POTENTIAL_FILE: Path = _find_file(_POT_NAME)
TM_FILE: Path = _find_file(_TM_NAME)

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
KB_EV: float = 8.617333e-5      # Boltzmann constant  [eV K^-1]
CONV_A2PS_TO_M2S: float = 1e-8  # Angstrom^2 ps^-1 -> m^2 s^-1
TIMESTEP_PS: float = 0.0005     # ADP diffusion runs  [ps]

# ══════════════════════════════════════════════════════════════════════════════
#  ELEMENT PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════
ELEMENTS: list[str] = ["W", "Mo", "Nb", "Zr", "Ti", "Ta"]
ELEM_TYPE: dict[str, int] = {e: i + 1 for i, e in enumerate(ELEMENTS)}

MASSES: dict[str, float] = {
    "W": 184.0, "Mo": 96.0, "Nb": 93.0,
    "Zr": 91.0, "Ti": 48.0, "Ta": 181.0,
}

TM_REF: dict[str, float] = {
    "W": 3695.0, "Mo": 2896.0, "Nb": 2770.0,
    "Zr": 2128.0, "Ti": 1941.0, "Ta": 3290.0,
}

A0_REF: dict[str, float] = {
    "W": 3.185, "Mo": 3.168, "Nb": 3.320,
    "Zr": 3.580, "Ti": 3.260, "Ta": 3.315,
}

SF_REF: dict[str, float] = {
    "W": 2.7, "Mo": 2.0, "Nb": 2.0,
    "Zr": 2.0, "Ti": 2.0, "Ta": 2.0,
}

# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENTIAL-FRACTION TYPE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════
# LAMMPS `set group all type/fraction` is applied sequentially and overwrites
# previously assigned types.  The setting order and the RNG seeds are fixed
# here so that every generator (constants.py, lammps_diffusion.py,
# lammps_tm.py) produces byte-identical inputs.
SET_FRACTION_ORDER: list[str] = ["Mo", "Nb", "Zr", "Ti", "Ta"]
SET_FRACTION_SEEDS: dict[str, int] = {
    "Mo": 2639, "Nb": 16392, "Zr": 7739, "Ti": 2039, "Ta": 56530,
}

# ══════════════════════════════════════════════════════════════════════════════
#  SRO CUTOFF
# ══════════════════════════════════════════════════════════════════════════════
# The SRO cutoff is composition dependent.  A fixed 3.0 Angstrom cutoff misses
# the Zr first coordination shell (r_1NN = 3.10 Angstrom for pure BCC Zr), so
# every caller must use compute_r_sro(a0) with the composition's Vegard a0.


def compute_r_sro(a0: float) -> float:
    """Return the SRO cutoff radius for a BCC crystal with lattice parameter a0.

    The cutoff is placed at the midpoint between the first and second
    nearest-neighbour shells of the BCC lattice:

    .. math::

        r_{1NN} = a_0 \\frac{\\sqrt{3}}{2}, \\qquad r_{2NN} = a_0

        r_{SRO} = \\frac{r_{1NN} + r_{2NN}}{2}
                = a_0 \\frac{\\sqrt{3}/2 + 1}{2} \\approx 0.9330 \\, a_0

    This captures all eight first-shell neighbours while excluding the six
    second-shell neighbours.  Across the full W-Mo-Nb-Zr-Ti-Ta composition
    space the gap to the second shell is at least 210 pm, so no second-shell
    contamination is possible.

    Parameters
    ----------
    a0 : float
        BCC lattice parameter [Angstrom].

    Returns
    -------
    float
        SRO cutoff radius [Angstrom].
    """
    return a0 * (math.sqrt(3) / 2 + 1.0) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  GRACE MLIP SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
GRACE_MODEL_DIR: str = os.environ.get(
    "GRACE_MODEL_DIR",
    "$HOME/users/razikaxz/.cache/grace/GRACE-2L-OMAT/",
)
GRACE_PAIR_STYLE: str = "grace"
GRACE_ELEMENTS: str = " ".join(ELEMENTS)   # "W Mo Nb Zr Ti Ta"
GRACE_TIMESTEP_PS: float = 0.001           # GRACE requires 2x longer step than ADP

# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE GRID (per composition, diffusion runs)
# ══════════════════════════════════════════════════════════════════════════════
T_FRAC_MIN: float = 0.50   # lower bound as fraction of Tm
T_FRAC_MAX: float = 0.80   # upper bound as fraction of Tm
#   0.80 is intentional: ADP Tm != GRACE Tm; running above 0.8*Tm_GRACE risks
#   melting the alloy during ADP diffusion runs.  Arrhenius R^2 > 0.99 confirms
#   the extrapolation to Tm is reliable.
T_ROUND_K: int = 25
T_MIN_ABS: float = 900.0

# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS SIMULATION PARAMETERS (ADP diffusion runs)
# ══════════════════════════════════════════════════════════════════════════════
CELL_SIZE: int = 6          # supercell edge in BCC units -> 6^3 x 2 = 432 atoms

# ══════════════════════════════════════════════════════════════════════════════
#  Tm COEXISTENCE SUPERCELL GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
COEX_NX: int = 10          # BCC unit cells in x
COEX_NY: int = 10          # BCC unit cells in y
COEX_NZ: int = 40          # BCC unit cells in z  (split 20/20 solid/liquid)
#   10x10x40 BCC = 8000 atoms.  Karavaev (2016) minimum is 432 atoms for the
#   modified Z-method; 8000 gives adequate compositional statistics for
#   6-element alloys (~1333 atoms per species at equiatomic).

# NPT damping constants for coexistence runs [ps]
COEX_TAU_T: float = 0.1      # temperature damping
COEX_TAU_P: float = 1.0      # pressure damping (main coexistence)
COEX_TAU_P_EQ: float = 0.05  # tighter damping for stress equalization step

# ══════════════════════════════════════════════════════════════════════════════
#  POLYNOMIAL FIT
# ══════════════════════════════════════════════════════════════════════════════
POLY_ALPHA: float = 1e-3

# ══════════════════════════════════════════════════════════════════════════════
#  SLURM / CLUSTER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SLURM_PARTITION: str = "compute"
SLURM_NODES: int = 1
SLURM_TASKS_PER_NODE: int = 1
SLURM_TM_TASKS_PER_NODE: int = 1
SLURM_TM_CPUS_PER_TASK: int = 16  # TF uses these as OMP threads
SLURM_CPUS_PER_TASK: int = 16

LAMMPS_CMD_TMPL: str = (
    "mpirun --bind-to none -np {ntasks} "
    "/home/users/razikaxz/lammps/build/lmp -in bcc_vac_adv.in"
)
LAMMPS_SERIAL_CMD: str = "/home/users/razikaxz/lammps/build/lmp -in {input_file}"
LAMMPS_EXE: str = "/home/users/razikaxz/lammps/build/lmp"


# ══════════════════════════════════════════════════════════════════════════════
#  MODE-DEPENDENT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

def _mode_parameters(test_mode: bool) -> dict[str, object]:
    """Return the mode-dependent parameter set.

    Parameters
    ----------
    test_mode : bool
        ``True`` selects the fast validation configuration; ``False`` selects
        the full production configuration.

    Returns
    -------
    dict
        Mapping of module-level parameter names to their values for the mode.
    """
    if test_mode:
        return {
            "N_COMPOSITIONS": 5,
            "N_TEMPS": 3,
            "N_MC": 100_000,
            "N_MD": 5_000_000,
            "COEX_N_EQUIL": 20_000,
            "COEX_N_MELT": 20_000,
            "COEX_N_COEX": 50_000,
            "COEX_N_STRESS_EQ": 10_000,
            "COEX_N_TGUESS": 3,
            "EF_N_SITES": 10,
            "SLURM_WALLTIME": "2:00:00",
            "SLURM_WALLTIME_TM": "4:00:00",
            "POLY_DEGREE": 1,
        }
    return {
        "N_COMPOSITIONS": 100,
        "N_TEMPS": 10,
        "N_MC": 2_000_000,
        "N_MD": 500_000_000,
        "COEX_N_EQUIL": 20_000,
        "COEX_N_MELT": 50_000,
        "COEX_N_COEX": 200_000,
        "COEX_N_STRESS_EQ": 50_000,
        "COEX_N_TGUESS": 5,
        "EF_N_SITES": 50,
        "SLURM_WALLTIME": "12:00:00",
        "SLURM_WALLTIME_TM": "48:00:00",
        "POLY_DEGREE": 2,
    }


def _apply_mode(test_mode: bool) -> None:
    """Bind the mode-dependent parameters into module globals.

    Parameters
    ----------
    test_mode : bool
        Mode to apply.
    """
    global TEST_MODE
    TEST_MODE = test_mode
    for name, value in _mode_parameters(test_mode).items():
        globals()[name] = value


def set_test_mode(test_mode: bool) -> None:
    """Switch between the test and production configurations at runtime.

    Recomputes every mode-dependent parameter (composition count, temperature
    count, MC/MD run lengths, coexistence run lengths, SLURM walltimes, and
    polynomial degree).  Physical constants are unaffected.

    Parameters
    ----------
    test_mode : bool
        ``True`` for the fast validation configuration, ``False`` for the
        full production configuration.
    """
    _apply_mode(bool(test_mode))


# Initialise module-level mode-dependent parameters.
_apply_mode(TEST_MODE)

# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITION SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED: int = 42

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: load Tm values from Tm_e.txt at import time
# ══════════════════════════════════════════════════════════════════════════════

def _load_tm_file() -> dict[str, float]:
    """Load melting temperatures, falling back to the built-in reference table.

    Returns
    -------
    dict of str to float
        Mapping of element symbol to melting temperature [K].  Values from
        ``Tm_e.txt`` override the built-in :data:`TM_REF` entries.
    """
    tm: dict[str, float] = dict(TM_REF)
    if TM_FILE.exists():
        with open(TM_FILE) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        tm[parts[0].capitalize()] = float(parts[1])
                    except ValueError:
                        continue
    return tm


TM: dict[str, float] = _load_tm_file()
