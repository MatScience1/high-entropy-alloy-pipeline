"""Stage 2 - compute per-composition physical constants.

Constants computed
------------------
a0 : float
    Equilibrium BCC lattice parameter [Angstrom].  Obtained from a LAMMPS
    ``box/relax`` at P = 0 on a 5x5x5 BCC supercell; falls back to Vegard's
    law if LAMMPS fails.
Tm : float
    Preliminary melting temperature [K] from the linear mixing rule,
    ``Tm = sum_i x_i Tm_i``.  Replaced by the GRACE coexistence result in
    Stage 3c.
Ef : float
    Mean vacancy formation energy [eV], Starikov 2024 Eq. 7.
Sf : float
    Composition-weighted vacancy formation entropy [kB].
C0 : float
    Arrhenius prefactor for the vacancy concentration, ``C0 = exp(Sf)``
    (Sf is already expressed in units of kB).
T_grid : list of int
    Simulation temperatures [K].

References
----------
Starikov et al., Phys. Rev. Materials 8, 043603 (2024) - Ef formula, Sf.
Vegard, Z. Phys. 5, 17 (1921) - Vegard's law.

Output
------
``results/constants/<comp_id>.json`` (per composition)
``results/constants_all.csv`` (aggregated)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import (
    A0_REF,
    ELEMENTS,
    LAMMPS_SERIAL_CMD,
    POTENTIAL_FILE,
    RESULTS_DIR,
    SF_REF,
    TM,
    T_FRAC_MAX,
    T_FRAC_MIN,
    T_MIN_ABS,
    T_ROUND_K,
)
from logging_config import get_logger
from pipeline.compositions import active_elements, row_to_comp
from pipeline.lammps_common import mass_block, set_fraction_block
from reporting import log_dataframe

logger = get_logger(__name__)

# Fallback vacancy formation energies [eV] used when the LAMMPS calculation
# fails.  Source: DFT literature (approximate).
_BCC_STABLE: set[str] = {"W", "Mo", "Nb", "Ta"}  # Zr, Ti are HCP at 0 K
_EF_FALLBACK: dict[str, float] = {
    "W": 3.56,   # Starikov 2024
    "Mo": 2.88,  # Starikov 2024
    "Nb": 2.91,  # Starikov 2024
    "Zr": 2.05,  # DFT-GGA BCC
    "Ti": 1.97,  # DFT-GGA BCC
    "Ta": 3.03,  # Satta et al. PRB 1999
}


# ══════════════════════════════════════════════════════════════════════════════
#  Analytical estimates (no LAMMPS)
# ══════════════════════════════════════════════════════════════════════════════

def vegard_a0(comp: dict[str, float]) -> float:
    """Return the Vegard's-law lattice parameter.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.

    Returns
    -------
    float
        ``a0 = sum_i x_i a0_i`` [Angstrom].
    """
    return sum(comp.get(e, 0.0) * A0_REF[e] for e in ELEMENTS)


def linear_tm(comp: dict[str, float]) -> float:
    """Return the rule-of-mixtures melting temperature.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.

    Returns
    -------
    float
        ``Tm = sum_i x_i Tm_i`` [K].
    """
    return sum(comp.get(e, 0.0) * TM[e] for e in ELEMENTS)


def composition_sf(comp: dict[str, float]) -> float:
    """Return the composition-weighted vacancy formation entropy.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.

    Returns
    -------
    float
        ``Sf = sum_i x_i Sf_i`` [kB].
    """
    return sum(comp.get(e, 0.0) * SF_REF[e] for e in ELEMENTS)


def make_temperature_grid(tm: float) -> list[int]:
    """Return the simulation temperature grid for a composition.

    The grid spans ``[max(T_MIN_ABS, T_FRAC_MIN * Tm), T_FRAC_MAX * Tm]`` with
    ``N_TEMPS`` points, each rounded to the nearest ``T_ROUND_K`` kelvin.

    Notes
    -----
    ``T_FRAC_MAX = 0.80`` is intentional: the ADP melting temperature differs
    from the GRACE melting temperature, and running above ``0.8 * Tm_GRACE``
    risks melting the alloy during the ADP diffusion runs.  The Arrhenius fit
    (R^2 > 0.99) supports reliable extrapolation to Tm.

    Parameters
    ----------
    tm : float
        Melting temperature [K] used to set the grid bounds.

    Returns
    -------
    list of int
        Sorted, de-duplicated simulation temperatures [K].
    """
    t_lo = max(T_MIN_ABS, T_FRAC_MIN * tm)
    t_hi = T_FRAC_MAX * tm
    if t_lo >= t_hi:
        t_lo = 0.5 * t_hi
    raw = np.linspace(t_lo, t_hi, int(config.N_TEMPS))
    return sorted({int(round(t / T_ROUND_K) * T_ROUND_K) for t in raw})


# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_lammps(inp: Path, cwd: Path, timeout: int = 300) -> str:
    """Run LAMMPS serially on an input file and return its combined output.

    Parameters
    ----------
    inp : pathlib.Path
        LAMMPS input file, relative to ``cwd``.
    cwd : pathlib.Path
        Working directory for the run.
    timeout : int, optional
        Wall-clock timeout [s].

    Returns
    -------
    str
        Combined stdout and stderr.

    Raises
    ------
    RuntimeError
        If LAMMPS exits with a non-zero status.
    """
    cmd = LAMMPS_SERIAL_CMD.format(input_file=inp.name).split()
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    out = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS exit {result.returncode}:\n{out[-2000:]}")
    return out


def compute_a0_lammps(comp: dict[str, float], a0_guess: float) -> float:
    """Return the equilibrium BCC lattice parameter via ``box/relax`` at P = 0.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    a0_guess : float
        Initial lattice parameter [Angstrom], typically the Vegard estimate.

    Returns
    -------
    float
        Relaxed lattice parameter [Angstrom], or ``a0_guess`` if LAMMPS fails.
    """
    n = int(config.EF_CELL_SIZE)
    pot = POTENTIAL_FILE

    lammps_in = f"""units metal
boundary p p p
atom_style atomic
lattice bcc {a0_guess:.5f}
region box block 0 {n} 0 {n} 0 {n}
create_box 6 box
create_atoms 1 box
{mass_block()}
{set_fraction_block(comp)}
pair_style adp
pair_coeff * * {pot} W Mo Nb Zr Ti Ta
fix relax all box/relax iso 0.0 vmax 0.001
minimize 1e-12 1e-12 2000 20000
variable a0_eq equal lx/{n}
print "EQUILIBRIUM_A0 ${{a0_eq}}"
"""
    with tempfile.TemporaryDirectory(prefix="a0_") as tmp:
        work = Path(tmp)
        shutil.copy(pot, work / pot.name)
        inp = work / "find_a0.in"
        inp.write_text(lammps_in)
        try:
            out = _run_lammps(inp, work, timeout=120)
            for line in out.splitlines():
                if "EQUILIBRIUM_A0" in line:
                    return float(line.split()[1])
        except Exception as exc:  # noqa: BLE001 - fall back to Vegard's law
            logger.warning("a0 LAMMPS failed (%s); using Vegard's law", exc)
    return a0_guess


def _lammps_perfect_input(comp: dict[str, float], a0: float, pot: Path, n: int) -> str:
    """Return a LAMMPS input that builds and minimises the perfect alloy.

    The script prints ``PERFECT_ENERGY N E Lx`` and writes ``perfect.data``
    for the subsequent vacancy runs.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    a0 : float
        BCC lattice parameter [Angstrom].
    pot : pathlib.Path
        ADP potential file.
    n : int
        Supercell edge length in BCC unit cells.

    Returns
    -------
    str
        Complete LAMMPS input script.
    """
    return f"""units metal
boundary p p p
atom_style atomic
lattice bcc {a0:.5f}
region box block 0 {n} 0 {n} 0 {n}
create_box 6 box
create_atoms 1 box
{mass_block()}
{set_fraction_block(comp)}
pair_style adp
pair_coeff * * {pot} W Mo Nb Zr Ti Ta
min_style cg
minimize 1e-12 1e-12 10000 10000
variable N    equal count(all)
variable Etot equal pe
variable Lbox equal lx
print "PERFECT_ENERGY ${{N}} ${{Etot}} ${{Lbox}}"
write_data perfect.data
"""


def _lammps_vacancy_input(atom_id: int, pot: Path) -> str:
    """Return a LAMMPS input that removes one atom and minimises the crystal.

    Parameters
    ----------
    atom_id : int
        Identifier of the atom to delete.
    pot : pathlib.Path
        ADP potential file.

    Returns
    -------
    str
        Complete LAMMPS input script.
    """
    return f"""units metal
boundary p p p
atom_style atomic
read_data perfect.data
group del id {atom_id}
delete_atoms group del bond no
pair_style adp
pair_coeff * * {pot} W Mo Nb Zr Ti Ta
min_style cg
minimize 1e-12 1e-12 10000 10000
variable Evac equal pe
print "VACANCY_ENERGY {atom_id} ${{Evac}}"
"""


def _parse_perfect(out: str) -> tuple[int, float]:
    """Parse ``PERFECT_ENERGY N E Lx`` from LAMMPS output.

    Parameters
    ----------
    out : str
        Combined LAMMPS output.

    Returns
    -------
    tuple of (int, float)
        Atom count and total potential energy [eV].

    Raises
    ------
    ValueError
        If the marker line is absent.
    """
    for line in out.splitlines():
        if line.strip().startswith("PERFECT_ENERGY"):
            parts = line.split()
            return int(parts[1]), float(parts[2])
    raise ValueError("PERFECT_ENERGY not found in LAMMPS output")


def _parse_vacancy(out: str) -> float:
    """Parse ``VACANCY_ENERGY atom_id E`` from LAMMPS output.

    Parameters
    ----------
    out : str
        Combined LAMMPS output.

    Returns
    -------
    float
        Total potential energy of the defected crystal [eV].

    Raises
    ------
    ValueError
        If the marker line is absent.
    """
    for line in out.splitlines():
        if line.strip().startswith("VACANCY_ENERGY"):
            return float(line.split()[2])
    raise ValueError("VACANCY_ENERGY not found in LAMMPS output")


def compute_ef(comp: dict[str, float], a0: float) -> float:
    """Compute the mean vacancy formation energy [eV].

    Implements Starikov 2024 Eq. 7, averaged over ``EF_N_SITES`` sites:

    .. math::

        \\langle E_f \\rangle = \\frac{1}{N_s} \\sum_i E(N-1, i)
            - \\frac{N-1}{N} E(N)

    where :math:`E(N)` is the perfect-crystal energy and :math:`E(N-1, i)` is
    the energy after removing atom :math:`i` and re-minimising.  The chemical
    potential of the removed atom is taken as :math:`E(N)/N`.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    a0 : float
        BCC lattice parameter [Angstrom].

    Returns
    -------
    float
        Mean vacancy formation energy [eV].

    Raises
    ------
    RuntimeError
        If no vacancy energies could be collected.
    """
    # Skip LAMMPS for HCP elements: ADP gives unphysical BCC energies.
    active = [e for e in ELEMENTS if comp.get(e, 0.0) > 0.99]
    if len(active) == 1 and active[0] not in _BCC_STABLE:
        return _EF_FALLBACK[active[0]]

    pot = POTENTIAL_FILE
    with tempfile.TemporaryDirectory(prefix="ef_") as tmp:
        work = Path(tmp)
        shutil.copy(pot, work / pot.name)

        inp = work / "perfect.in"
        inp.write_text(_lammps_perfect_input(comp, a0, pot, int(config.EF_CELL_SIZE)))
        out = _run_lammps(inp, work)
        n_atoms, e0 = _parse_perfect(out)

        n_sites = int(config.EF_N_SITES)
        step = max(1, n_atoms // n_sites)
        ids = list(range(1, n_atoms + 1, step))[:n_sites]
        e_vacs: list[float] = []
        for atom_id in ids:
            inp2 = work / f"vac_{atom_id}.in"
            inp2.write_text(_lammps_vacancy_input(atom_id, pot))
            e_vacs.append(_parse_vacancy(_run_lammps(inp2, work)))

    if not e_vacs:
        raise RuntimeError("No vacancy energies collected")

    return float(np.mean(e_vacs)) - (n_atoms - 1) / n_atoms * e0


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_constants(compositions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a0, Tm, Ef, Sf, C0, and T_grid for every composition.

    Parameters
    ----------
    compositions_df : pandas.DataFrame
        Output of :func:`pipeline.compositions.generate_compositions`.

    Returns
    -------
    pandas.DataFrame
        One row per composition.  Written to ``results/constants_all.csv``;
        per-composition JSON is written to ``results/constants/``.
    """
    const_dir = RESULTS_DIR / "constants"
    const_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    for _, row in compositions_df.iterrows():
        comp_id = str(row["comp_id"])
        comp = row_to_comp(row)
        logger.info("%s  active=%s", comp_id, active_elements(comp))

        a0_vegard = vegard_a0(comp)
        a0 = compute_a0_lammps(comp, a0_vegard)
        logger.info("  a0: Vegard=%.4f A  LAMMPS=%.4f A", a0_vegard, a0)

        tm = linear_tm(comp)
        sf = composition_sf(comp)
        c0 = float(np.exp(sf))  # C0 = exp(Sf/kB); Sf in kB units
        t_grid = make_temperature_grid(tm)

        try:
            ef = compute_ef(comp, a0)
        except Exception as exc:  # noqa: BLE001 - fall back to literature values
            logger.warning("Ef LAMMPS failed for %s (%s)", comp_id, exc)
            ef = sum(comp.get(e, 0.0) * _EF_FALLBACK.get(e, 2.5) for e in ELEMENTS)
            logger.warning("Using fallback Ef = %.4f eV for %s", ef, comp_id)

        rec: dict[str, object] = {
            "comp_id": comp_id,
            "a0_vegard": round(a0, 5),
            "Tm": round(tm, 1),
            "Ef": round(ef, 5),
            "Sf": round(sf, 4),
            "C0": round(c0, 6),
            "T_grid": t_grid,
            **{f"x_{e}": round(comp.get(e, 0.0), 6) for e in ELEMENTS},
        }

        (const_dir / f"{comp_id}.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
        logger.info(
            "  Tm=%.0f K  Ef=%.4f eV  C0=%.4f  T_grid=%s", tm, ef, c0, t_grid
        )

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "constants_all.csv"
    df.to_csv(out, index=False)
    logger.info("Saved %d rows -> %s", len(df), out)

    summary_cols = ["comp_id", "a0_vegard", "Tm", "Ef", "Sf", "C0"]
    log_dataframe(logger, df[summary_cols], title="Computed constants:")
    return df
