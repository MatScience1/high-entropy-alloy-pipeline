"""Stages 3b and 3c - melting temperature via phase coexistence (GRACE MLIP).

Why a separate module from :mod:`pipeline.lammps_diffusion`
-----------------------------------------------------------
The ADP potential (diffusion runs) and the GRACE MLIP (Tm runs) serve
different physical purposes:

* ADP: fast structural dynamics, MSD convergence, SRO.  Tm is approximate.
* GRACE: high-fidelity energy surface, used only for the Tm determination.

The real Tm from coexistence feeds back into ``constants_all.csv`` and sets
the homologous temperature axis ``T / Tm`` for all D*(T/Tm) plots.

Method: Modified Z-method (Karavaev et al. 2016)
------------------------------------------------
1. Build an elongated BCC supercell (``NX x NY x NZ``, z is the long axis).
2. Assign the alloy composition via the sequential-fraction formula.
3. Minimise, then equilibrate the full crystal at ``T_solid = 0.5 * Tm_rom``.
4. Freeze the solid half (``z in [0, NZ/2]``) and heat the liquid half
   (``z in [NZ/2, NZ]``) at ``2 * Tm_rom`` until it melts.
5. Release all atoms and run NPT at ``T_guess`` with independent x, y, z
   barostats.  An isotropic barostat violates Karavaev's stress
   equalization requirement.
6. Stress equalization: tight per-axis NPT until ``Pxx ~ Pyy ~ Pzz ~ 0``.
7. Read volume and pressure from the log:

   * ``dV > 0`` -> melting (``T_guess > Tm``) -> try a lower ``T_guess``
   * ``dV < 0`` -> freezing (``T_guess < Tm``) -> try a higher ``T_guess``
   * ``dV ~ 0`` and ``|P| < 500`` bar -> coexistence -> ``T_guess ~ Tm``

``COEX_N_TGUESS`` values bracket ``Tm_rom`` by +/-20%.  After all runs finish,
:func:`parse_coexistence_log` identifies the stable one and
:func:`patch_constants_with_tm` writes the real Tm and rebuilt T_grid back to
``constants_all.csv``.

References
----------
Karavaev et al., J. Chem. Phys. 144, 194507 (2016) - modified Z-method.
Zhu et al., npj Comput. Mater. 10, 60 (2024) - MLIP for Tm.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd

import config
from config import (
    COEX_NX,
    COEX_NY,
    COEX_NZ,
    COEX_TAU_P,
    COEX_TAU_P_EQ,
    COEX_TAU_T,
    ELEMENTS,
    GRACE_TIMESTEP_PS,
    LAMMPS_EXE,
    PIPELINE_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    SLURM_CPUS_PER_TASK,
    SLURM_NODES,
    SLURM_PARTITION,
    SLURM_TASKS_PER_NODE,
    SLURM_WALLTIME_TM,
)
from logging_config import get_logger
from pipeline.constants import make_temperature_grid
from pipeline.lammps_common import mass_block, set_fraction_block

logger = get_logger(__name__)

COEX_SOLID_HALF: int = COEX_NZ // 2   # solid occupies z in [0, NZ/2] in lattice units

# Centred bracketing: always includes 1.0 x Tm_rom, in steps of 0.10.
_BRACKET_FACTORS: list[float] = [0.80, 0.90, 1.00, 1.10, 1.20]


# ══════════════════════════════════════════════════════════════════════════════
#  T_guess bracketing helper
# ══════════════════════════════════════════════════════════════════════════════

def tguess_values(tm_rom: float) -> list[int]:
    """Return the bracketing ``T_guess`` values around ``Tm_rom``.

    Parameters
    ----------
    tm_rom : float
        Rule-of-mixtures melting temperature [K].

    Returns
    -------
    list of int
        ``COEX_N_TGUESS`` temperatures.  In test mode (N=3) these are
        ``[0.90, 1.00, 1.10] * Tm_rom``; in production (N=5) they are
        ``[0.80, 0.90, 1.00, 1.10, 1.20] * Tm_rom``.
    """
    n = int(config.COEX_N_TGUESS)
    mid = len(_BRACKET_FACTORS) // 2
    start = mid - n // 2
    return [
        int(round(_BRACKET_FACTORS[i] * tm_rom))
        for i in range(start, start + n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS coexistence input
# ══════════════════════════════════════════════════════════════════════════════

def build_coexistence_input(
    comp: dict[str, float],
    a0: float,
    T_guess: int,
    Tm_rom: float,
) -> str:
    """Return a complete ``coexistence.in`` for one (composition, T_guess).

    ``T_guess`` and ``Tm_rom`` are kept separate: ``Tm_rom`` sets the solid
    (``0.5 * Tm_rom``) and liquid (``2 * Tm_rom``) preparation temperatures,
    while ``T_guess`` is the NPT coexistence temperature being tested.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    a0 : float
        Lattice parameter [Angstrom].
    T_guess : int
        Coexistence test temperature [K].
    Tm_rom : float
        Rule-of-mixtures melting temperature [K].

    Returns
    -------
    str
        Complete LAMMPS input script.
    """
    t_solid = int(round(0.50 * Tm_rom))
    t_liquid = int(round(2.00 * Tm_rom))
    n_atoms = COEX_NX * COEX_NY * COEX_NZ * 2
    header = ", ".join(f"{e}:{comp.get(e, 0):.3f}" for e in ELEMENTS)

    return f"""\
# ═══════════════════════════════════════════════════════════════════════════
# coexistence.in  —  Phase Coexistence Tm
# Composition : {header}
# T_guess     : {T_guess} K   |   Tm_rom = {int(round(Tm_rom))} K
# Method      : Modified Z-method + stress equalization (Karavaev 2016)
# Potential   : GRACE-2L-OMAT (Zhu 2024)
# Supercell   : {COEX_NX}x{COEX_NY}x{COEX_NZ} BCC = {n_atoms} atoms
# ═══════════════════════════════════════════════════════════════════════════

variable  GRACE_MODEL_DIR  getenv  GRACE_MODEL_DIR

units        metal
boundary     p p p
atom_style   atomic
neighbor     1.0 bin
neigh_modify every 2 delay 4 check yes
timestep     {GRACE_TIMESTEP_PS}

# ── supercell (BCC — all refractory HEA phases are BCC) ──────────────────
lattice       bcc {a0:.5f}
region        box block 0 {COEX_NX} 0 {COEX_NY} 0 {COEX_NZ}
create_box    6 box        # 6 element types for GRACE
create_atoms  1 box        # all atoms start as W

{mass_block()}

# ── alloy composition ─────────────────────────────────────────────────────
{set_fraction_block(comp)}

# ── GRACE MLIP ────────────────────────────────────────────────────────────
pair_style  grace
pair_coeff  * * ${{GRACE_MODEL_DIR}} {" ".join(ELEMENTS)}

# ── define solid / liquid half-regions ───────────────────────────────────
region   solid_reg   block INF INF  INF INF  0                  {COEX_SOLID_HALF}  units lattice
region   liquid_reg  block INF INF  INF INF  {COEX_SOLID_HALF}  {COEX_NZ}          units lattice
group    solid_grp   region solid_reg
group    liquid_grp  region liquid_reg

# ── Step 1: minimise to remove composition-assignment overlaps ────────────
thermo       500
thermo_style custom step temp pe ke etotal press vol pxx pyy pzz
min_style    hftn
minimize     1.0e-8 1.0e-8 2000 2000

# ── Step 2: equilibrate full crystal at T_solid = {t_solid} K ────────────
velocity  all create {t_solid} 482751 dist gaussian
fix       eq_solid all nvt temp {t_solid} {t_solid} {COEX_TAU_T}
run       {int(config.COEX_N_EQUIL)}
unfix     eq_solid

# ── Step 3: melt liquid half at T_liquid = {t_liquid} K (2 x Tm_rom) ─────
#    Solid half is frozen (setforce 0) to preserve BCC order during melting.
fix   freeze_solid solid_grp setforce 0.0 0.0 0.0
velocity  liquid_grp create {t_liquid} 918273 dist gaussian
fix       melt_liq liquid_grp nvt temp {t_liquid} {t_liquid} {COEX_TAU_T}
run       {int(config.COEX_N_MELT)}
unfix     melt_liq
unfix     freeze_solid

# ── Step 4: coexistence NPT at T_guess = {T_guess} K ─────────────────────
#    Independent x, y, z barostats (Karavaev requirement).
#    NO velocity rescaling — it would destroy the solid/liquid interface.
#    Convergence diagnostic (logged every 1000 steps):
#      vol increasing  -> melting (T_guess > Tm)   -> run at lower T_guess
#      vol decreasing  -> freezing (T_guess < Tm)  -> run at higher T_guess
#      vol stable, |press| < 500 bar               -> interface stable -> Tm ~ T_guess

fix   coex_npt all npt \\
      temp  {T_guess}  {T_guess}  {COEX_TAU_T} \\
      x     0.0  0.0  {COEX_TAU_P} \\
      y     0.0  0.0  {COEX_TAU_P} \\
      z     0.0  0.0  {COEX_TAU_P}

thermo       1000
thermo_style custom step temp pe ke etotal vol press pxx pyy pzz lx ly lz

dump  coex_dump all custom 10000 dump.coexistence id type x y z vx vy vz
run   {int(config.COEX_N_COEX)}
undump coex_dump
unfix  coex_npt

# ── Step 5: stress equalization (Karavaev 2016 mandate) ──────────────────
#    After coexistence, Pxx/Pyy/Pzz may be unequal (anisotropic stress).
#    Tighten the barostat damping (tau = {COEX_TAU_P_EQ} ps) to equalize
#    normal stresses without changing temperature.
#    Without this step, Tm is systematically biased.

fix   stress_eq all npt \\
      temp  {T_guess}  {T_guess}  {COEX_TAU_T} \\
      x     0.0  0.0  {COEX_TAU_P_EQ} \\
      y     0.0  0.0  {COEX_TAU_P_EQ} \\
      z     0.0  0.0  {COEX_TAU_P_EQ}

thermo       500
thermo_style custom step temp pe ke etotal vol press pxx pyy pzz
run   {int(config.COEX_N_STRESS_EQ)}
unfix stress_eq

# ── End marker (read by parse_coexistence_log) ────────────────────────────
print "COEXISTENCE_RESULT T_guess={T_guess} comp={header}"
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SLURM scripts
# ══════════════════════════════════════════════════════════════════════════════

def _env_block() -> str:
    """Return the shared SLURM environment setup (micromamba + GRACE)."""
    return """\
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate grace

export GRACE_MODEL_DIR="$HOME/.cache/grace/GRACE-2L-OMAT"
export CUDA_VISIBLE_DEVICES="-1"
export OMP_NUM_THREADS=16
export TF_NUM_INTRAOP_THREADS=16
export TF_NUM_INTEROP_THREADS=2
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1"""


def build_tm_array_script(
    job_list: list[dict[str, object]],
    max_concurrent: int = 20,
) -> None:
    """Write ``slurm/submit_Tm_array.sh`` as one SLURM array job.

    The path dispatch table is embedded in the script as bash associative
    arrays, so there is no runtime dependency on CSV files.

    Parameters
    ----------
    job_list : list of dict
        Records with keys ``comp_id``, ``T_guess``, and ``path``.
    max_concurrent : int, optional
        Maximum number of concurrently running array tasks.
    """
    n = len(job_list)
    n_comps = len({j["comp_id"] for j in job_list})

    path_arr = "\n".join(
        f'JOB_PATHS[{i}]="{j["path"]}"' for i, j in enumerate(job_list)
    )
    label_arr = "\n".join(
        f'JOB_LABELS[{i}]="{j["comp_id"]}_T{j["T_guess"]}"'
        for i, j in enumerate(job_list)
    )

    script = f"""#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# submit_Tm_array.sh — SLURM Job Array for Tm Phase Coexistence
# {n} runs : {n_comps} compositions x {int(config.COEX_N_TGUESS)} T_guess values
# Submit  : sbatch slurm/submit_Tm_array.sh
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=Tm_coex
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks-per-node={SLURM_TASKS_PER_NODE}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --time={SLURM_WALLTIME_TM}
#SBATCH --array=0-{n - 1}%{max_concurrent}
#SBATCH --output=logs/Tm_%A_%a.out
#SBATCH --error=logs/Tm_%A_%a.err

# ── dispatch table ────────────────────────────────────────────────────────
declare -A JOB_PATHS
declare -A JOB_LABELS
{path_arr}
{label_arr}

# ── environment ───────────────────────────────────────────────────────────
{_env_block()}

# ── run ───────────────────────────────────────────────────────────────────
IDX=$SLURM_ARRAY_TASK_ID
RUN_DIR="${{JOB_PATHS[$IDX]}}"
LABEL="${{JOB_LABELS[$IDX]}}"

echo "[$IDX] $LABEL  host=$(hostname)  start=$(date)"

[ -d "$RUN_DIR" ]             || {{ echo "[$IDX] ERROR: $RUN_DIR not found"; exit 1; }}
[ -f "$RUN_DIR/coexistence.in" ] || {{ echo "[$IDX] ERROR: coexistence.in missing"; exit 1; }}

cd "$RUN_DIR"
mkdir -p dump

srun {LAMMPS_EXE} -in coexistence.in -log log.coexistence
EXIT_CODE=$?

echo "[$IDX] $LABEL  exit=$EXIT_CODE  end=$(date)"
exit $EXIT_CODE
"""

    out = PIPELINE_DIR / "slurm" / "submit_Tm_array.sh"
    out.parent.mkdir(exist_ok=True)
    out.write_text(script)
    out.chmod(0o755)
    logger.info("Wrote %s (%d array tasks)", out, n)


# ══════════════════════════════════════════════════════════════════════════════
#  Log parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_coexistence_log(log_path: Path) -> float | None:
    """Return the Tm estimate from ``log.coexistence``, or ``None``.

    Reads all thermo rows (columns ``step temp pe ke etotal vol press pxx pyy
    pzz``) and stops at the ``COEXISTENCE_RESULT`` marker.  The stress
    equalization run is always last, so its rows form the tail.

    Convergence criterion: the mean absolute pressure over the final 50 rows
    is below 500 bar (0.05 GPa).

    Parameters
    ----------
    log_path : pathlib.Path
        Path to the LAMMPS coexistence log.

    Returns
    -------
    float or None
        Mean temperature of the final 50 rows if converged, else ``None``.
        Truncated or unreadable logs return ``None`` rather than raising.
    """
    if not log_path.exists():
        return None

    temps: list[float] = []
    presses: list[float] = []
    found_result = False

    try:
        with log_path.open() as fh:
            for line in fh:
                stripped = line.strip()
                if "COEXISTENCE_RESULT" in stripped:
                    found_result = True
                    break
                parts = stripped.split()
                # Thermo data line: integer step number, at least 10 columns.
                if len(parts) >= 10 and parts[0].isdigit():
                    try:
                        temps.append(float(parts[1]))    # temp
                        presses.append(float(parts[6]))  # press
                    except ValueError:
                        continue
    except OSError as exc:
        logger.warning("Cannot read %s (%s)", log_path, exc)
        return None

    if not found_result or len(temps) < 10:
        return None   # run crashed or did not complete

    tail_p = presses[-50:]
    if abs(statistics.mean(tail_p)) < 500.0:
        return statistics.mean(temps[-50:])
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Directory builder - Stage 3b
# ══════════════════════════════════════════════════════════════════════════════

def create_tm_directories(
    compositions_df: pd.DataFrame,
    constants_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Write ``runs/<comp_id>/tm_coexistence/T_<guess>/coexistence.in``.

    Parameters
    ----------
    compositions_df : pandas.DataFrame
        Output of :func:`pipeline.compositions.generate_compositions`.
    constants_df : pandas.DataFrame
        Output of :func:`pipeline.constants.compute_all_constants`.

    Returns
    -------
    list of dict
        Records with keys ``comp_id``, ``T_guess``, ``Tm_rom``, and ``path``.
        Also written to ``results/tm_job_list.csv``.
    """
    job_list: list[dict[str, object]] = []

    for _, crow in compositions_df.iterrows():
        comp_id = str(crow["comp_id"])
        comp = {e: float(crow[f"x_{e}"]) for e in ELEMENTS}

        const = constants_df[constants_df["comp_id"] == comp_id]
        if const.empty:
            logger.warning("%s: no constants row - skipped", comp_id)
            continue

        a0 = float(const.iloc[0]["a0_vegard"])
        tm_rom = float(const.iloc[0]["Tm"])

        for t_guess in tguess_values(tm_rom):
            sim_dir = RUNS_DIR / comp_id / "tm_coexistence" / f"T_{t_guess}"
            sim_dir.mkdir(parents=True, exist_ok=True)
            (sim_dir / "dump").mkdir(exist_ok=True)

            (sim_dir / "coexistence.in").write_text(
                build_coexistence_input(comp, a0, t_guess, tm_rom)
            )

            job_list.append({
                "comp_id": comp_id,
                "T_guess": t_guess,
                "Tm_rom": round(tm_rom, 1),
                "path": str(sim_dir),
            })

    out_csv = RESULTS_DIR / "tm_job_list.csv"
    pd.DataFrame(job_list).to_csv(out_csv, index=False)
    logger.info("Created %d coexistence directories -> %s", len(job_list), out_csv)
    return job_list


# ══════════════════════════════════════════════════════════════════════════════
#  Patch constants - Stage 3c
# ══════════════════════════════════════════════════════════════════════════════

def patch_constants_with_tm(
    constants_csv: Path,
    job_list: list[dict[str, object]],
) -> None:
    """Patch ``constants_all.csv`` with the real Tm from coexistence logs.

    Reads the completed ``log.coexistence`` files, identifies the stable
    ``T_guess`` for each composition, and writes the real Tm and the rebuilt
    temperature grid back to ``constants_all.csv``.  Multiple converged
    ``T_guess`` values per composition are averaged; a spread above 100 K
    triggers a warning that the bracketing may have missed the window.

    Parameters
    ----------
    constants_csv : pathlib.Path
        Path to ``results/constants_all.csv``.
    job_list : list of dict
        Records with keys ``comp_id`` and ``path``.
    """
    df = pd.read_csv(constants_csv)

    by_comp: dict[str, list[dict[str, object]]] = {}
    for job in job_list:
        by_comp.setdefault(str(job["comp_id"]), []).append(job)

    updated = 0
    for comp_id, jobs in by_comp.items():
        converged: list[float] = []
        for job in sorted(jobs, key=lambda x: x["T_guess"]):
            tm = parse_coexistence_log(Path(str(job["path"])) / "log.coexistence")
            if tm is not None:
                converged.append(tm)

        if not converged:
            logger.warning("%s: no converged run - keeping Tm_rom", comp_id)
            continue

        real_tm = statistics.mean(converged)
        spread = max(converged) - min(converged) if len(converged) > 1 else 0.0
        if spread > 100.0:
            logger.warning(
                "%s: Tm spread = %.0f K - check bracketing", comp_id, spread
            )

        t_grid = make_temperature_grid(real_tm)
        mask = df["comp_id"] == comp_id
        df.loc[mask, "Tm"] = round(real_tm, 1)
        df.loc[mask, "T_grid"] = json.dumps(t_grid)
        updated += 1
        logger.info(
            "%s: Tm = %.0f K (spread %.0f K, %d pts)  T_grid = %s",
            comp_id, real_tm, spread, len(converged), t_grid,
        )

    df.to_csv(constants_csv, index=False)
    logger.info(
        "Patched %d/%d compositions -> %s", updated, len(by_comp), constants_csv
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline entry points (called from run_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def stage_3b(
    comps_df: pd.DataFrame,
    consts_df: pd.DataFrame,
    max_concurrent: int = 20,
) -> list[dict[str, object]]:
    """Generate coexistence inputs and ``slurm/submit_Tm_array.sh``.

    Parameters
    ----------
    comps_df : pandas.DataFrame
        Compositions table.
    consts_df : pandas.DataFrame
        Constants table.
    max_concurrent : int, optional
        Maximum number of concurrently running array tasks.

    Returns
    -------
    list of dict
        Coexistence job records.
    """
    job_list = create_tm_directories(comps_df, consts_df)
    build_tm_array_script(job_list, max_concurrent)
    logger.info("Next: sbatch slurm/submit_Tm_array.sh")
    logger.info("After jobs finish: python run_pipeline.py --only_stage 32")
    return job_list


def stage_3c() -> None:
    """Patch ``constants_all.csv`` with the real Tm from completed logs.

    Raises
    ------
    FileNotFoundError
        If ``results/tm_job_list.csv`` is missing (Stage 3b not yet run).
    """
    job_csv = RESULTS_DIR / "tm_job_list.csv"
    if not job_csv.exists():
        raise FileNotFoundError(f"{job_csv} not found - run Stage 3b first")
    jobs = pd.read_csv(job_csv).to_dict("records")
    patch_constants_with_tm(RESULTS_DIR / "constants_all.csv", jobs)
