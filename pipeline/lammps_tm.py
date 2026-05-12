"""
pipeline/lammps_tm.py
=====================
Stages 3b & 3c — Melting temperature via phase coexistence (GRACE MLIP).

Why a separate module from lammps_diffusion.py
-----------------------------------------------
The ADP potential (diffusion runs) and GRACE MLIP (Tm runs) are used for
different physical purposes:
  - ADP  : fast structural dynamics, MSD convergence, SRO.  Tm is approximate.
  - GRACE: high-fidelity energy surface.  Used ONLY for Tm determination.

The real Tm from coexistence feeds back to constants_all.csv and sets the
homologous temperature axis T/Tm for all D*(T/Tm) plots.

Method: Modified Z-method (Karavaev et al. 2016)
-------------------------------------------------
1.  Build elongated BCC supercell (NX × NY × NZ, z-axis is the long axis).
2.  Assign alloy composition via sequential-fraction formula.
3.  Minimise, then equilibrate the full crystal at T_solid = 0.5·Tm_rom.
4.  Freeze the solid half (z ∈ [0, NZ/2]).
    Heat the liquid half (z ∈ [NZ/2, NZ]) at 2·Tm_rom until it melts.
5.  Release all atoms.  Run NPT at T_guess with INDEPENDENT x, y, z barostats.
    (iso barostat violates Karavaev's stress equalization requirement.)
6.  Stress equalization: tight per-axis NPT until Pxx ≈ Pyy ≈ Pzz ≈ 0.
7.  Read volume and pressure from log:
      ΔV > 0  →  melting   (T_guess > Tm) → try lower T_guess
      ΔV < 0  →  freezing  (T_guess < Tm) → try higher T_guess
      ΔV ≈ 0, |press| < 500 bar → coexistence → T_guess ≈ Tm

Five T_guess values bracket Tm_rom ± 20%.  After all runs finish,
parse_coexistence_log() identifies the stable one and patch_constants()
writes real_Tm + rebuilt T_grid back to constants_all.csv.

References
----------
  Karavaev et al., J. Chem. Phys. 144, 194507 (2016) — modified Z-method
  Zhu et al., npj Comput. Mater. 10, 60 (2024)       — MLIP for Tm
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd

from config import (
    COEX_N_EQUIL, COEX_N_MELT, COEX_N_COEX, COEX_N_STRESS_EQ,
    COEX_N_TGUESS, COEX_NX, COEX_NY, COEX_NZ,
    COEX_TAU_P, COEX_TAU_P_EQ, COEX_TAU_T,
    ELEMENTS, GRACE_MODEL_DIR, GRACE_TIMESTEP_PS, LAMMPS_EXE, MASSES,
    PIPELINE_DIR, RESULTS_DIR, RUNS_DIR,
    SLURM_NODES, SLURM_PARTITION, SLURM_TASKS_PER_NODE, SLURM_CPUS_PER_TASK, SLURM_WALLTIME_TM,
)
from pipeline.constants import make_temperature_grid, sequential_fracs

ELEM_TYPE       = {e: i + 1 for i, e in enumerate(ELEMENTS)}
COEX_SOLID_HALF = COEX_NZ // 2   # solid occupies z ∈ [0, NZ/2] in lattice units


# ══════════════════════════════════════════════════════════════════════════════
#  Block builders
# ══════════════════════════════════════════════════════════════════════════════

def _mass_block() -> str:
    return "\n".join(
        f"mass  {ELEM_TYPE[e]}  {MASSES[e]}   # {e}" for e in ELEMENTS
    )


def _set_block(comp: dict[str, float]) -> str:
    fracs = sequential_fracs(comp)
    order = ["Mo", "Nb", "Zr", "Ti", "Ta"]
    seeds = {"Mo": 2639, "Nb": 16392, "Zr": 7739, "Ti": 2039, "Ta": 56530}
    lines = []
    for e in order:
        f = fracs.get(e, 0.0)
        if f > 1e-9:
            lines.append(
                f"set group all type/fraction {ELEM_TYPE[e]} "
                f"{f:.8f} {seeds[e]}   # {e}: target {comp.get(e,0):.4f}"
            )
    return "\n".join(lines) if lines else "# (pure W)"


# ══════════════════════════════════════════════════════════════════════════════
#  T_guess bracketing helper
# ══════════════════════════════════════════════════════════════════════════════

# Centred bracketing: always includes 1.0 × Tm_rom, ± steps of 0.10
_BRACKET_FACTORS = [0.80, 0.90, 1.00, 1.10, 1.20]


def tguess_values(Tm_rom: float) -> list[int]:
    """
    Return COEX_N_TGUESS temperatures bracketing Tm_rom.
    In TEST_MODE (N=3): [0.90, 1.00, 1.10] × Tm_rom
    In production (N=5): [0.80, 0.90, 1.00, 1.10, 1.20] × Tm_rom
    """
    n     = COEX_N_TGUESS
    mid   = len(_BRACKET_FACTORS) // 2
    start = mid - n // 2
    return [int(round(_BRACKET_FACTORS[i] * Tm_rom))
            for i in range(start, start + n)]


# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS coexistence input
# ══════════════════════════════════════════════════════════════════════════════

def build_coexistence_input(
    comp:    dict[str, float],
    a0:      float,
    T_guess: int,
    Tm_rom:  float,
) -> str:
    """
    Return a complete LAMMPS coexistence.in for one (composition, T_guess).

    T_guess and Tm_rom are kept separate:
      Tm_rom  → sets T_solid (0.5·Tm_rom) and T_liquid (2·Tm_rom)
      T_guess → the NPT coexistence temperature being tested
    """
    T_solid  = int(round(0.50 * Tm_rom))
    T_liquid = int(round(2.00 * Tm_rom))
    n_atoms  = COEX_NX * COEX_NY * COEX_NZ * 2
    header   = ", ".join(f"{e}:{comp.get(e,0):.3f}" for e in ELEMENTS)

    return f"""\
# ═══════════════════════════════════════════════════════════════════════════
# coexistence.in  —  Phase Coexistence Tm
# Composition : {header}
# T_guess     : {T_guess} K   |   Tm_rom = {int(round(Tm_rom))} K
# Method      : Modified Z-method + stress equalization (Karavaev 2016)
# Potential   : GRACE-2L-OMAT (Zhu 2024)
# Supercell   : {COEX_NX}×{COEX_NY}×{COEX_NZ} BCC = {n_atoms} atoms
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

{_mass_block()}

# ── alloy composition ─────────────────────────────────────────────────────
{_set_block(comp)}

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

# ── Step 2: equilibrate full crystal at T_solid = {T_solid} K ────────────
velocity  all create {T_solid} 482751 dist gaussian
fix       eq_solid all nvt temp {T_solid} {T_solid} {COEX_TAU_T}
run       {COEX_N_EQUIL}
unfix     eq_solid

# ── Step 3: melt liquid half at T_liquid = {T_liquid} K (2 × Tm_rom) ─────
#    Solid half is frozen (setforce 0) to preserve BCC order during melting.
fix   freeze_solid solid_grp setforce 0.0 0.0 0.0
velocity  liquid_grp create {T_liquid} 918273 dist gaussian
fix       melt_liq liquid_grp nvt temp {T_liquid} {T_liquid} {COEX_TAU_T}
run       {COEX_N_MELT}
unfix     melt_liq
unfix     freeze_solid

# ── Step 4: coexistence NPT at T_guess = {T_guess} K ─────────────────────
#    Independent x, y, z barostats (Karavaev requirement).
#    NO velocity rescaling — it would destroy the solid/liquid interface.
#    Convergence diagnostic (logged every 1000 steps):
#      vol increasing  → melting (T_guess > Tm)   → run at lower T_guess
#      vol decreasing  → freezing (T_guess < Tm)  → run at higher T_guess
#      vol stable, |press| < 500 bar              → interface stable → Tm ≈ T_guess

fix   coex_npt all npt \\
      temp  {T_guess}  {T_guess}  {COEX_TAU_T} \\
      x     0.0  0.0  {COEX_TAU_P} \\
      y     0.0  0.0  {COEX_TAU_P} \\
      z     0.0  0.0  {COEX_TAU_P}

thermo       1000
thermo_style custom step temp pe ke etotal vol press pxx pyy pzz lx ly lz

dump  coex_dump all custom 10000 dump.coexistence id type x y z vx vy vz
run   {COEX_N_COEX}
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
run   {COEX_N_STRESS_EQ}
unfix stress_eq

# ── End marker (read by parse_coexistence_log) ────────────────────────────
print "COEXISTENCE_RESULT T_guess={T_guess} comp={header}"
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SLURM scripts
# ══════════════════════════════════════════════════════════════════════════════

def _env_block() -> str:
    """Shared SLURM environment setup (micromamba + GRACE)."""
    return f"""\
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


def build_tm_array_script(job_list: list[dict], max_concurrent: int = 20) -> None:
    """
    Write slurm/submit_Tm_array.sh — one SLURM array job for all Tm runs.

    The path dispatch table is embedded in the script as a bash associative
    array so there is no runtime dependency on CSV files.
    """
    n       = len(job_list)
    n_comps = len({j["comp_id"] for j in job_list})

    path_arr  = "\n".join(
        f'JOB_PATHS[{i}]="{j["path"]}"' for i, j in enumerate(job_list)
    )
    label_arr = "\n".join(
        f'JOB_LABELS[{i}]="{j["comp_id"]}_T{j["T_guess"]}"'
        for i, j in enumerate(job_list)
    )

    script = f"""#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# submit_Tm_array.sh — SLURM Job Array for Tm Phase Coexistence
# {n} runs : {n_comps} compositions × {COEX_N_TGUESS} T_guess values
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
    print(f"[lammps_tm] Written: {out}  ({n} array tasks)")


# ══════════════════════════════════════════════════════════════════════════════
#  Log parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_coexistence_log(log_path: Path) -> float | None:
    """
    Return Tm estimate from log.coexistence, or None if not converged.

    Reads all thermo rows from the log (columns: step temp pe ke etotal
    vol press pxx pyy pzz).  Stops at the COEXISTENCE_RESULT print line.
    The stress-equalization run is always last, so its rows are the tail.

    Convergence: mean |press| over the final 50 rows < 500 bar (0.05 GPa).
    """
    if not log_path.exists():
        return None

    temps:   list[float] = []
    presses: list[float] = []
    found_result = False

    with log_path.open() as fh:
        for line in fh:
            s = line.strip()
            if "COEXISTENCE_RESULT" in s:
                found_result = True
                break
            parts = s.split()
            # Thermo data line: first token is an integer step number, ≥10 cols
            if len(parts) >= 10 and parts[0].isdigit():
                try:
                    temps.append(float(parts[1]))    # temp
                    presses.append(float(parts[6]))  # press
                except ValueError:
                    continue

    if not found_result or len(temps) < 10:
        return None   # run crashed or didn't complete

    tail_p = presses[-50:]
    if abs(statistics.mean(tail_p)) < 500.0:
        return statistics.mean(temps[-50:])
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  Directory builder — Stage 3b
# ══════════════════════════════════════════════════════════════════════════════

def create_tm_directories(compositions_df: pd.DataFrame,
                           constants_df:   pd.DataFrame) -> list[dict]:
    """
    Write runs/<comp_id>/tm_coexistence/T_<guess>/coexistence.in.
    Returns job list and saves results/tm_job_list.csv.
    """
    job_list: list[dict] = []

    for _, crow in compositions_df.iterrows():
        comp_id = crow["comp_id"]
        comp    = {e: float(crow[f"x_{e}"]) for e in ELEMENTS}

        const = constants_df[constants_df["comp_id"] == comp_id]
        if const.empty:
            print(f"[WARN] {comp_id}: no constants row — skipped")
            continue

        a0     = float(const.iloc[0]["a0_vegard"])
        Tm_rom = float(const.iloc[0]["Tm"])

        for T_guess in tguess_values(Tm_rom):
            sim_dir = RUNS_DIR / comp_id / "tm_coexistence" / f"T_{T_guess}"
            sim_dir.mkdir(parents=True, exist_ok=True)
            (sim_dir / "dump").mkdir(exist_ok=True)

            (sim_dir / "coexistence.in").write_text(
                build_coexistence_input(comp, a0, T_guess, Tm_rom)
            )

            job_list.append({
                "comp_id": comp_id,
                "T_guess": T_guess,
                "Tm_rom":  round(Tm_rom, 1),
                "path":    str(sim_dir),
            })

    out_csv = RESULTS_DIR / "tm_job_list.csv"
    pd.DataFrame(job_list).to_csv(out_csv, index=False)
    print(f"[lammps_tm] {len(job_list)} coexistence directories  →  {out_csv}")
    return job_list


# ══════════════════════════════════════════════════════════════════════════════
#  Patch constants — Stage 3c
# ══════════════════════════════════════════════════════════════════════════════

def patch_constants_with_tm(constants_csv: Path,
                             job_list:      list[dict]) -> None:
    """
    Read completed log.coexistence files, identify stable T_guess for each
    composition, and write real_Tm + make_temperature_grid(real_Tm) back to
    constants_all.csv.

    Accepts multiple converged T_guess values per composition and averages
    them.  Warns if spread > 100 K (bracketing may have missed the window).
    """
    df = pd.read_csv(constants_csv)

    by_comp: dict[str, list[dict]] = {}
    for j in job_list:
        by_comp.setdefault(j["comp_id"], []).append(j)

    updated = 0
    for comp_id, jobs in by_comp.items():
        converged = []
        for j in sorted(jobs, key=lambda x: x["T_guess"]):
            Tm = parse_coexistence_log(Path(j["path"]) / "log.coexistence")
            if Tm is not None:
                converged.append(Tm)

        if not converged:
            print(f"[WARN] {comp_id}: no converged run — keeping Tm_rom")
            continue

        real_Tm = statistics.mean(converged)
        spread  = max(converged) - min(converged) if len(converged) > 1 else 0.0
        if spread > 100.0:
            print(f"[WARN] {comp_id}: Tm spread = {spread:.0f} K — "
                  f"check bracketing")

        T_grid = make_temperature_grid(real_Tm)
        mask   = df["comp_id"] == comp_id
        df.loc[mask, "Tm"]     = round(real_Tm, 1)
        df.loc[mask, "T_grid"] = json.dumps(T_grid)
        updated += 1
        print(f"  {comp_id}: Tm = {real_Tm:.0f} K  "
              f"(spread {spread:.0f} K, {len(converged)} pts)  "
              f"T_grid = {T_grid}")

    df.to_csv(constants_csv, index=False)
    print(f"[lammps_tm] Patched {updated}/{len(by_comp)} compositions  "
          f"→  {constants_csv}")


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline entry points (called from run_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def stage_3b(comps_df: pd.DataFrame, consts_df: pd.DataFrame,
             max_concurrent: int = 20) -> list[dict]:
    """Generate coexistence inputs and submit_Tm_array.sh."""
    job_list = create_tm_directories(comps_df, consts_df)
    build_tm_array_script(job_list, max_concurrent)
    print("\n  Next: sbatch slurm/submit_Tm_array.sh")
    print("  After jobs finish: python run_pipeline.py --only_stage 32\n")
    return job_list


def stage_3c() -> None:
    """Patch constants_all.csv with real Tm from completed coexistence logs."""
    job_csv = RESULTS_DIR / "tm_job_list.csv"
    if not job_csv.exists():
        raise FileNotFoundError(
            f"{job_csv} not found — run Stage 3b first"
        )
    jobs = pd.read_csv(job_csv).to_dict("records")
    patch_constants_with_tm(RESULTS_DIR / "constants_all.csv", jobs)
