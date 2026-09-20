"""Stage 3 - generate LAMMPS inputs for the ADP-potential diffusion runs.

One directory is created per (composition, temperature)::

    runs/<comp_id>/sim_<T>/
        bcc_vac_adv.in   LAMMPS script (MC equilibration + MD diffusion)
        submit.sh        SLURM single-job script
        dump/            directory for LAMMPS dump files

The ADP potential (``WMoNbZrTiTa.nist.adp.txt``) is used for the diffusion
runs; GRACE is used only for the melting-temperature determination
(see :mod:`pipeline.lammps_tm`).

Physics of the generated script
-------------------------------
1. Build a BCC supercell (``CELL_SIZE``^3 x 2 atoms).
2. Assign the composition via the sequential-fraction formula
   (see :func:`pipeline.lammps_common.sequential_fracs`).
3. Minimise (HFTN) to relax overlaps.
4. MC equilibration: ``N_MC`` steps of atom/swap moves over all 15 element
   pairs via ``fix atom/swap`` under NPT, developing short-range order.
5. MD diffusion: ``N_MD`` steps under NPT with a single vacancy.  The MSD is
   tracked per element, giving D* through the Einstein relation.
6. SRO: Warren-Cowley parameters ``alpha_ij = 1 - P_ij / x_j`` are tracked
   during the MD run with the composition-dependent cutoff ``r_sro``.

Output
------
``results/job_list.csv`` with columns ``comp_id, T, path``.
"""

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import pandas as pd

import config
from config import (
    CELL_SIZE,
    ELEM_TYPE,
    ELEMENTS,
    LAMMPS_CMD_TMPL,
    POTENTIAL_FILE,
    RESULTS_DIR,
    RUNS_DIR,
    SLURM_NODES,
    SLURM_PARTITION,
    SLURM_TASKS_PER_NODE,
    SLURM_WALLTIME,
    compute_r_sro,
)
from logging_config import get_logger
from pipeline.lammps_common import mass_block, set_fraction_block

logger = get_logger(__name__)

# All 15 unique element pairs, used for the atom/swap MC fixes.
_SWAP_PAIRS: list[tuple[str, str]] = list(itertools.combinations(ELEMENTS, 2))


# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS script block builders
# ══════════════════════════════════════════════════════════════════════════════

def _group_block() -> str:
    """Return dynamic per-element groups, refreshed every 100 steps."""
    lines: list[str] = []
    for element in ELEMENTS:
        lines.append(
            f"variable  is{element}  atom  \"type=={ELEM_TYPE[element]}\"\n"
            f"group     {element}_type  dynamic all var is{element}  every 100"
        )
    return "\n".join(lines)


def _comp_variable_block() -> str:
    """Return instantaneous concentration variables from dynamic group counts."""
    lines: list[str] = ["variable  tot_numb  equal  count(all)"]
    for element in ELEMENTS:
        lines.append(f"variable  {element}_numb  equal  count({element}_type)")
        lines.append(
            f"variable  {element}_comp  equal  v_{element}_numb/v_tot_numb"
        )
    return "\n".join(lines)


def _sro_block(r_sro: float) -> str:
    """Return the coordination and Warren-Cowley SRO block.

    For each ordered pair (A, B):

    * ``coord_AB`` = mean number of B atoms within ``r_sro`` of each A atom,
    * ``z_A`` = total coordination of A (sum over all B),
    * ``alpha_AB = 1 - (coord_AB / z_A) / x_B`` (Warren-Cowley, Cowley 1950).

    ``alpha = 0`` indicates random mixing, ``alpha > 0`` unlike-pair depletion,
    and ``alpha < 0`` unlike-pair enrichment.

    Parameters
    ----------
    r_sro : float
        SRO cutoff radius [Angstrom].

    Returns
    -------
    str
        LAMMPS ``compute``/``variable`` block.
    """
    coord = "\n".join(
        f"compute  c{a}{b}  {a}_type  coord/atom  "
        f"cutoff {r_sro:.4f}  group {b}_type"
        for a in ELEMENTS for b in ELEMENTS
    )
    avg = "\n".join(
        f"compute  avg{a}{b}  {a}_type  reduce ave  c_c{a}{b}"
        for a in ELEMENTS for b in ELEMENTS
    )
    z = "\n".join(
        f"variable  z{a}  equal  "
        + "+".join(f"c_avg{a}{b}" for b in ELEMENTS)
        + "+0.00001"          # avoid division by zero for dilute elements
        for a in ELEMENTS
    )
    alpha = "\n".join(
        f"variable  alpha{a}{b}  equal  "
        f"1.0-((c_avg{a}{b}/v_z{a})/(v_{b}_comp+0.00001))"
        for a in ELEMENTS for b in ELEMENTS
    )
    return f"{coord}\n\n{avg}\n\n{z}\n\n{alpha}"


def _av_columns() -> str:
    """Return the column list for the ``fix ave/time`` SRO output."""
    cols = ["c_T_c"]
    cols += [f"v_{e}_comp" for e in ELEMENTS]
    cols += [f"v_alpha{a}{b}" for a in ELEMENTS for b in ELEMENTS]
    return " ".join(cols)


def _mc_block() -> str:
    """Return the 15 ``fix atom/swap`` commands, one per element pair."""
    lines: list[str] = []
    for idx, (e1, e2) in enumerate(_SWAP_PAIRS):
        seed = 1709 + idx * 1031
        lines.append(
            f"fix  mc_{e1}{e2}  body  atom/swap  1000  100  {seed}  ${{T}}  "
            f"ke yes  types {ELEM_TYPE[e1]} {ELEM_TYPE[e2]}"
        )
    return "\n".join(lines)


def _unfix_mc() -> str:
    """Return the ``unfix`` commands for all atom/swap fixes."""
    return "\n".join(f"unfix mc_{e1}{e2}" for e1, e2 in _SWAP_PAIRS)


def _msd_block() -> str:
    """Return the MSD computes for all atoms and for each element."""
    lines = ["compute  msd_all  all  msd"]
    for element in ELEMENTS:
        lines.append(f"compute  msd_{element.lower()}  {element}_dif  msd")
    return "\n".join(lines)


def _msd_thermo() -> str:
    """Return the thermo column list for the MD production run."""
    cols = "step temp pe ke etotal pxx pyy pzz press vol lx ly lz"
    cols += "".join(f" v_{e}_comp" for e in ELEMENTS)
    cols += " c_msd_all[1] c_msd_all[4]"
    cols += "".join(
        f" c_msd_{e.lower()}[1] c_msd_{e.lower()}[4]" for e in ELEMENTS
    )
    return cols


# ══════════════════════════════════════════════════════════════════════════════
#  Main template
# ══════════════════════════════════════════════════════════════════════════════

def build_diffusion_input(comp: dict[str, float], T: float, a0: float) -> str:
    """Return a complete ``bcc_vac_adv.in`` LAMMPS script.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    T : float
        Simulation temperature [K].
    a0 : float
        Lattice parameter [Angstrom].

    Returns
    -------
    str
        Complete LAMMPS input script.
    """
    temperature = int(round(T))
    t_init = 2 * temperature                    # initial velocity temperature
    r_sro = round(compute_r_sro(a0), 4)         # composition-specific cutoff
    lx = ly = lz = CELL_SIZE
    n_mc = int(config.N_MC)
    n_md = int(config.N_MD)

    header = ", ".join(f"{e}:{comp.get(e, 0):.3f}" for e in ELEMENTS)

    return f"""# ─────────────────────────────────────────────────────────────────────────
# bcc_vac_adv.in  —  WMoNbZrTiTa ADP diffusion run
# Composition : {header}
# Temperature : {temperature} K
# Potential   : ADP (WMoNbZrTiTa.nist.adp.txt)
# ─────────────────────────────────────────────────────────────────────────

units        metal
boundary     p p p
atom_style   atomic
neighbor     0.5 bin
neigh_modify every 2 delay 10 check yes

# ── simulation parameters ─────────────────────────────────────────────────
variable  T     equal  {temperature}
variable  r_sro equal  {r_sro}    # BCC 1NN/2NN midpoint for a0={a0:.4f} A
variable  lx    equal  {lx}
variable  ly    equal  {ly}
variable  lz    equal  {lz}
variable  N_mc  equal  {n_mc}
variable  N_md  equal  {n_md}
variable  T_in  equal  {t_init}   # initial velocity temperature (2xT)

# ── BCC supercell ({lx}x{ly}x{lz} unit cells = {lx*ly*lz*2} atoms before vacancy) ──
lattice       bcc {a0:.5f}
region        box block 0.0 ${{lx}} 0.0 ${{ly}} 0.0 ${{lz}}
create_box    6 box
create_atoms  1 box    # all atoms start as W (type 1)

{mass_block()}

# ── alloy creation (sequential-fraction formula — see pipeline/lammps_common.py) ─
{set_fraction_block(comp)}

velocity all create ${{T_in}} 130784

timestep 0.0005    # 0.5 fs  (ADP is stiff near a vacancy)

pair_style  adp
pair_coeff  * * {POTENTIAL_FILE.name} W Mo Nb Zr Ti Ta

# ── basic diagnostics ─────────────────────────────────────────────────────
compute  pe_at  all  pe/atom
compute  cna    all  cna/atom 3.4      # common neighbour analysis
compute  cen_at all  centro/atom bcc
compute  T_c    all  temp

thermo       100
thermo_style custom step temp pe ke etotal press vol

dump snap_init all cfg 100000 ./dump/alloy_creation.*.cfg \\
     mass type xs ys zs type c_pe_at c_cen_at c_cna
dump_modify snap_init element {" ".join(ELEMENTS)}

min_style  hftn
minimize   1.0e-10 1.0e-10 1000 1000    # relax alloy overlaps

# ── per-element dynamic groups ────────────────────────────────────────────
{_group_block()}

{_comp_variable_block()}

# ── SRO computes (Warren-Cowley; cutoff = BCC 1NN/2NN midpoint) ──────────
{_sro_block(r_sro)}

variable  volume equal vol

thermo_style custom step temp pe ke etotal pxx pyy pzz press vol lx ly lz \\
             v_W_comp v_Mo_comp v_Nb_comp v_Zr_comp v_Ti_comp v_Ta_comp

# time-averaged SRO output: one row per 2000 steps
fix av_GB all ave/time 20 100 2000 {_av_columns()} file MD_T_${{T}}K.txt

# ── MC + NPT equilibration (chemical ordering) ────────────────────────────
group anchor id 20       # prevent COM drift during NPT
group body   subtract all anchor

fix main body npt temp ${{T}} ${{T}} 5.0 \\
              x 0.0 0.0 5.0  y 0.0 0.0 5.0  z 0.0 0.0 5.0

{_mc_block()}

run ${{N_mc}}    # MC equilibration: {n_mc:,} steps

unfix main
{_unfix_mc()}

# ── MD diffusion run (single vacancy) ────────────────────────────────────
compute disp all displace/atom
undump  snap_init

dump snap all cfg 100000 ./dump/alloy_diffusion.*.cfg \\
     mass type xs ys zs type c_pe_at c_cen_at c_cna c_disp[4]
dump_modify snap element {" ".join(ELEMENTS)}

fix main_1 body npt temp ${{T}} ${{T}} 5.0 \\
               x 0.0 0.0 5.0  y 0.0 0.0 5.0  z 0.0 0.0 5.0

# Create one vacancy (atom 110 is near the cell centre for a {lx}^3 BCC cell)
group del id 110
delete_atoms group del

# Per-element diffusion groups (static — type labels fixed after vacancy)
{chr(10).join(f"group  {e}_dif  type  {ELEM_TYPE[e]}" for e in ELEMENTS)}

{_msd_block()}

thermo_style custom {_msd_thermo()}

run ${{N_md}}    # MD production: {n_md:,} steps
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SLURM script
# ══════════════════════════════════════════════════════════════════════════════

def _submit_script(comp_id: str, T: int) -> str:
    """Return the SLURM submission script for one diffusion run.

    Parameters
    ----------
    comp_id : str
        Composition identifier.
    T : int
        Simulation temperature [K].

    Returns
    -------
    str
        SLURM batch script.
    """
    ntasks = SLURM_NODES * SLURM_TASKS_PER_NODE
    return f"""#!/bin/bash
#SBATCH --job-name={comp_id}_T{T}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks-per-node={SLURM_TASKS_PER_NODE}
#SBATCH --time={SLURM_WALLTIME}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate grace

export CUDA_VISIBLE_DEVICES="-1"
export OMP_NUM_THREADS=1

echo "Starting {comp_id} T={T} K  host=$(hostname)  $(date)"
mkdir -p dump

{LAMMPS_CMD_TMPL.format(ntasks=ntasks)} -in bcc_vac_adv.in

echo "Finished $(date)"
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Directory builder
# ══════════════════════════════════════════════════════════════════════════════

def create_run_directories(
    compositions_df: pd.DataFrame,
    constants_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create ``runs/<comp_id>/sim_<T>/`` with inputs and submit scripts.

    Parameters
    ----------
    compositions_df : pandas.DataFrame
        Output of :func:`pipeline.compositions.generate_compositions`.
    constants_df : pandas.DataFrame
        Output of :func:`pipeline.constants.compute_all_constants`.

    Returns
    -------
    list of dict
        Records with keys ``comp_id``, ``T``, and ``path``.  Also written to
        ``results/job_list.csv``.
    """
    job_list: list[dict[str, object]] = []

    for _, crow in compositions_df.iterrows():
        comp_id = str(crow["comp_id"])
        comp = {e: float(crow[f"x_{e}"]) for e in ELEMENTS}

        const = constants_df[constants_df["comp_id"] == comp_id].iloc[0]
        a0 = float(const["a0_vegard"])
        t_grid = (
            json.loads(const["T_grid"])
            if isinstance(const["T_grid"], str)
            else const["T_grid"]
        )

        for T in t_grid:
            sim_dir = RUNS_DIR / comp_id / f"sim_{T}"
            sim_dir.mkdir(parents=True, exist_ok=True)
            (sim_dir / "dump").mkdir(exist_ok=True)

            # Symlink (or copy) the potential file so LAMMPS finds it locally.
            pot_link = sim_dir / POTENTIAL_FILE.name
            if not pot_link.exists():
                try:
                    pot_link.symlink_to(POTENTIAL_FILE)
                except OSError:
                    shutil.copy(POTENTIAL_FILE, pot_link)

            (sim_dir / "bcc_vac_adv.in").write_text(
                build_diffusion_input(comp, T, a0)
            )
            (sim_dir / "submit.sh").write_text(_submit_script(comp_id, int(T)))

            job_list.append({"comp_id": comp_id, "T": T, "path": str(sim_dir)})

    pd.DataFrame(job_list).to_csv(RESULTS_DIR / "job_list.csv", index=False)
    logger.info("Created %d diffusion run directories", len(job_list))
    return job_list
