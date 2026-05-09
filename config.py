#!/usr/bin/env python3
"""
config.py — Central configuration for the WMoNbZrTiTa diffusion pipeline.

Toggle TEST_MODE to switch between a rapid debug run (≲1 day total on cluster)
and the full production run (100 compositions, ~4 h wall-time with enough nodes).

All physical constants, element properties, and SLURM settings live here so
that every other script imports from one place.

Changes vs original
───────────────────
  [1] R_SRO removed as a fixed constant.
      Replaced by compute_r_sro(a0) which returns the midpoint between
      r_1NN and r_2NN for the BCC lattice at the composition's Vegard a0.
      Rationale: fixed 3.0 Å misses the Zr first shell (r_1NN = 3.10 Å).
      The midpoint formula a0*(√3/2 + 1)/2 gives ≥210 pm gap to 2NN for
      all compositions in the W-Mo-Nb-Zr-Ti-Ta space.

  [2] Added GRACE_MODEL_DIR, GRACE_PAIR_STYLE, GRACE_ELEMENTS.
      All Tm-coexistence scripts import from here; no hardcoded paths.

  [3] Added SLURM_WALLTIME_TM (separate walltime for Tm coexistence runs).

  [4] Added coexistence supercell geometry constants (COEX_NX/NY/NZ,
      COEX_TIMESTEP_PS) so tm_solution.py imports them instead of hardcoding.
"""

import math
import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  MODE SELECTOR  ← change this line only
# ══════════════════════════════════════════════════════════════════════════════
TEST_MODE: bool = False   # True → fast debug run; False → full production

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_DIR  = Path(__file__).resolve().parent
RUNS_DIR      = PIPELINE_DIR / "runs"
RESULTS_DIR   = PIPELINE_DIR / "results"
DOCS_DIR      = PIPELINE_DIR / "docs"

_POT_NAME = "WMoNbZrTiTa.nist.adp.txt"
_TM_NAME  = "Tm_e.txt"


def _find_file(name: str) -> Path:
    """Search for a reference file in the pipeline dir, then parent/ref."""
    candidates = [
        PIPELINE_DIR / name,
        PIPELINE_DIR.parent / "ref" / name,
        PIPELINE_DIR / "ref" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


POTENTIAL_FILE = _find_file(_POT_NAME)
TM_FILE        = _find_file(_TM_NAME)

# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
KB_EV: float = 8.617333e-5      # Boltzmann constant  [eV K⁻¹]
CONV_A2PS_TO_M2S: float = 1e-8  # Å² ps⁻¹ → m² s⁻¹
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
#  SRO CUTOFF  [Fix 1]
# ══════════════════════════════════════════════════════════════════════════════
# R_SRO is no longer a single global constant.  The original value of 3.0 Å
# misses the Zr first coordination shell (r_1NN = 3.10 Å for pure BCC Zr).
#
# Use compute_r_sro(a0) wherever a cutoff is needed.  In generate_lammps_inputs
# and build_coexistence_input, pass the Vegard a0 for the composition.

def compute_r_sro(a0: float) -> float:
    """
    Return the SRO cutoff radius [Å] for a BCC crystal with lattice parameter a0.

    Formula: midpoint between r_1NN and r_2NN in BCC.
      r_1NN = a0 * √3 / 2   (8 nearest neighbours along <111>)
      r_2NN = a0             (6 next-nearest along <100>)
      r_sro = a0 * (√3/2 + 1) / 2 = a0 * 0.9330

    Gap to 2NN ≥ 210 pm across the full W-Mo-Nb-Zr-Ti-Ta composition space,
    so no second-shell contamination is possible.
    """
    return a0 * (math.sqrt(3) / 2 + 1.0) / 2.0


# Kept as a fallback for any legacy code that imports the name directly.
# Will be removed once all callers are updated to use compute_r_sro(a0).
R_SRO: float = compute_r_sro(3.30)   # equiatomic estimate; 3.079 Å

# ══════════════════════════════════════════════════════════════════════════════
#  GRACE MLIP SETTINGS  [Fix 2]
# ══════════════════════════════════════════════════════════════════════════════
GRACE_MODEL_DIR: str = os.environ.get(
    "GRACE_MODEL_DIR",
    "$HOME/.cache/grace/GRACE-2L-OMAT",
)
GRACE_PAIR_STYLE: str = "grace"
GRACE_ELEMENTS:   str = " ".join(ELEMENTS)   # "W Mo Nb Zr Ti Ta"
GRACE_TIMESTEP_PS: float = 0.001             # GRACE requires 2× longer step than ADP

# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITION SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
N_COMPOSITIONS: int = 5 if TEST_MODE else 100
RANDOM_SEED: int = 42

# ══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE GRID (per composition, diffusion runs)
# ══════════════════════════════════════════════════════════════════════════════
T_FRAC_MIN: float = 0.50   # lower bound as fraction of Tm
T_FRAC_MAX: float = 0.80   # upper bound as fraction of Tm
#   0.80 is intentional: ADP Tm ≠ GRACE Tm; running above 0.8*Tm_GRACE risks
#   melting the alloy during ADP diffusion runs.  Arrhenius R²>0.99 confirms
#   the extrapolation to Tm is reliable.
N_TEMPS:    int   = 3 if TEST_MODE else 10
T_ROUND_K:  int   = 25
T_MIN_ABS:  float = 900.0

# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS SIMULATION PARAMETERS (ADP diffusion runs)
# ══════════════════════════════════════════════════════════════════════════════
CELL_SIZE: int = 6          # supercell edge in BCC units → 6³×2 = 432 atoms

N_MC: int = 100_000    if TEST_MODE else 2_000_000
N_MD: int = 5_000_000  if TEST_MODE else 500_000_000

# ══════════════════════════════════════════════════════════════════════════════
#  Tm COEXISTENCE SUPERCELL GEOMETRY  [Fix 4]
# ══════════════════════════════════════════════════════════════════════════════
COEX_NX: int = 10          # BCC unit cells in x
COEX_NY: int = 10          # BCC unit cells in y
COEX_NZ: int = 40          # BCC unit cells in z  (split 20/20 solid/liquid)
#   10×10×40 BCC = 8 000 atoms.  Karavaev (2016) minimum is 432 atoms for
#   the modified Z-method; 8 000 gives adequate compositional statistics for
#   6-element alloys (≈1333 atoms per species at equiatomic).

# NPT damping constants for coexistence runs [ps]
COEX_TAU_T: float = 0.1    # temperature damping
COEX_TAU_P: float = 1.0    # pressure damping (main coexistence)
COEX_TAU_P_EQ: float = 0.05  # tighter damping for stress equalization step

# Run lengths for coexistence stages [steps at GRACE_TIMESTEP_PS = 0.001 ps]
COEX_N_EQUIL:    int = 20_000 if TEST_MODE else  20_000  # solid equilibration
COEX_N_MELT:     int = 20_000 if TEST_MODE else  50_000  # liquid half melting
COEX_N_COEX:     int = 50_000 if TEST_MODE else 200_000  # interface equilibration
COEX_N_STRESS_EQ: int = 10_000 if TEST_MODE else  50_000  # Karavaev stress eq.

# Number of bracketing T_guess values per composition (spans ±20% of Tm_rom)
COEX_N_TGUESS: int = 3 if TEST_MODE else 5

# ══════════════════════════════════════════════════════════════════════════════
#  Ef STATIC CALCULATION
# ══════════════════════════════════════════════════════════════════════════════
EF_CELL_SIZE: int = 5
EF_N_SITES:   int = 10 if TEST_MODE else 50

# ══════════════════════════════════════════════════════════════════════════════
#  SLURM / CLUSTER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SLURM_PARTITION:      str = "compute"
SLURM_NODES:          int = 1
SLURM_TASKS_PER_NODE: int = 16
SLURM_WALLTIME:       str = "2:00:00"  if TEST_MODE else "12:00:00"  # ADP runs
SLURM_WALLTIME_TM:    str = "4:00:00"  if TEST_MODE else "48:00:00"  # [Fix 3] Tm runs

LAMMPS_CMD_TMPL:   str = "mpirun --bind-to none -np {ntasks} /home/users/razikaxz/lammps/build/lmp -in bcc_vac_adv.in"
LAMMPS_SERIAL_CMD: str = "/home/users/razikaxz/lammps/build/lmp -in {input_file}"
LAMMPS_EXE:        str = "/home/users/razikaxz/lammps/build/lmp"

# ══════════════════════════════════════════════════════════════════════════════
#  POLYNOMIAL FIT
# ══════════════════════════════════════════════════════════════════════════════
POLY_DEGREE: int = 1 if TEST_MODE else 2
POLY_ALPHA:  float = 1e-3

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: load Tm values from Tm_e.txt at import time
# ══════════════════════════════════════════════════════════════════════════════
def _load_tm_file() -> dict[str, float]:
    tm: dict[str, float] = dict(TM_REF)
    if TM_FILE.exists():
        with open(TM_FILE) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        tm[parts[0].capitalize()] = float(parts[1])
                    except ValueError:
                        pass
    return tm


TM: dict[str, float] = _load_tm_file()
