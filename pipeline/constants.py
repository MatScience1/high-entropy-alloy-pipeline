"""
pipeline/constants.py
=====================
Stage 2 — Compute per-composition physical constants.

Constants computed
------------------
  a0   [Å]    Vegard's-law lattice parameter: a0 = Σᵢ xᵢ a0ᵢ
               Used as the BCC lattice parameter for all LAMMPS inputs.

  Tm   [K]    Preliminary melting temperature via linear mixing rule:
               Tm = Σᵢ xᵢ Tmᵢ
               Replaced by GRACE coexistence result in Stage 3c.

  Ef   [eV]   Mean vacancy formation energy — Starikov 2024 Eq. 7:
               <Ef> = mean_i [ E(N−1, site i removed) ] − (N−1)/N · E(N)
               Averaged over EF_N_SITES random vacancy sites.
               Chemical potential μ = E(N)/N (energy per atom in the alloy).

  Sf   [kB]   Vacancy formation entropy — composition-weighted:
               Sf = Σᵢ xᵢ Sf,i    (Sf,i from Starikov 2024: W=2.7, others≈2.0)

  C0   [—]    Arrhenius prefactor for vacancy concentration:
               Cv = C0 · exp(−Ef / kB T)    →    C0 = exp(Sf / kB)
               Since Sf is already stored in units of kB, C0 = exp(Sf).

  T_grid [K]  Simulation temperatures: N_TEMPS points in
               [max(T_MIN_ABS, T_FRAC_MIN·Tm), T_FRAC_MAX·Tm],
               rounded to T_ROUND_K.

References
----------
  Starikov et al., Phys. Rev. Materials 8, 043603 (2024)  — Ef formula, Sf
  Vegard, Z. Phys. 5, 17 (1921)                           — Vegard's law

Output
------
  results/constants/<comp_id>.json   (per-composition)
  results/constants_all.csv          (aggregated)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    A0_REF, EF_CELL_SIZE, EF_N_SITES, ELEMENTS,
    KB_EV, LAMMPS_SERIAL_CMD, MASSES, POTENTIAL_FILE,
    RESULTS_DIR, SF_REF, TM,
    T_FRAC_MIN, T_FRAC_MAX, N_TEMPS, T_MIN_ABS, T_ROUND_K,
)
from pipeline.compositions import active_elements, row_to_comp


# ══════════════════════════════════════════════════════════════════════════════
#  Sequential-fraction formula
#  (shared with lammps_diffusion.py and lammps_tm.py)
# ══════════════════════════════════════════════════════════════════════════════

def sequential_fracs(comp: dict[str, float]) -> dict[str, float]:
    """
    Compute conditional probabilities for LAMMPS 'set type/fraction'.

    LAMMPS applies set operations sequentially on ALL atoms:
      Step 1 (Mo): set fraction f_Mo of all atoms → Mo
      Step 2 (Nb): set fraction f_Nb of all atoms → Nb  (overwrites some Mo)
      ...
      Step 5 (Ta): set fraction f_Ta of all atoms → Ta  (last, no overwrites)

    The probability of ending as element X is:
      P(X) = f_X · ∏_{Y set after X} (1 − f_Y)

    Solving for f_X in reverse order of setting:
      f_Ta  = c_Ta
      f_Ti  = c_Ti  / (1 − c_Ta)
      f_Zr  = c_Zr  / (1 − c_Ta − c_Ti)
      f_Nb  = c_Nb  / (1 − c_Ta − c_Ti − c_Zr)
      f_Mo  = c_Mo  / (1 − c_Ta − c_Ti − c_Zr − c_Nb)

    The original bcc_vac_adv.in used products (1−cTa)(1−cTi)… which is
    incorrect: it treats fractions as independent when they are not, causing
    ~6–12% concentration error for equiatomic alloys.
    """
    setting_order = ["Mo", "Nb", "Zr", "Ti", "Ta"]
    fracs: dict[str, float] = {}
    claimed = 0.0
    for e in reversed(setting_order):   # Ta first (set last → not overwritten)
        c = comp.get(e, 0.0)
        denom = 1.0 - claimed
        fracs[e] = c / denom if denom > 1e-9 else 0.0
        claimed += c
    return fracs

# ══════════════════════════════════════════════════════════════════════════════
# A0 calculation
# ══════════════════════════════════════════════════════════════════════════════

def compute_a0_lammps(comp: dict[str, float], a0_guess: float) -> float:
    """
    Equilibrium BCC lattice parameter via box/relax at P=0.
    EF_CELL_SIZE³×2 supercell with alloy composition assigned.
    Returns a0 [Å]. Falls back to Vegard's law on LAMMPS failure.
    """
    N   = EF_CELL_SIZE   # 5 → 250 atoms
    pot = POTENTIAL_FILE

    lammps_in = f"""
units metal
boundary p p p
atom_style atomic
lattice bcc {a0_guess:.5f}
region box block 0 {N} 0 {N} 0 {N}
create_box 6 box
create_atoms 1 box
{chr(10).join(f"mass {i+1} {MASSES[e]}  # {e}" for i,e in enumerate(ELEMENTS))}
{chr(10).join(
    f"set group all type/fraction {ELEMENTS.index(e)+1} {f:.8f} {abs(hash(e))%90000+10000}"
    for e,f in sequential_fracs(comp).items() if f > 1e-9
)}
pair_style adp
pair_coeff * * {pot} W Mo Nb Zr Ti Ta
fix relax all box/relax iso 0.0 vmax 0.001
minimize 1e-12 1e-12 2000 20000
variable a0_eq equal lx/{N}
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
        except Exception as exc:
            print(f"  [WARN] a0 LAMMPS failed: {exc} — using Vegard's law")
    return a0_guess   # fallback

# ══════════════════════════════════════════════════════════════════════════════
#  Analytical estimates (no LAMMPS)
# ══════════════════════════════════════════════════════════════════════════════

def vegard_a0(comp: dict[str, float]) -> float:
    """Vegard's law: a0 = Σᵢ xᵢ a0ᵢ  [Å]"""
    return sum(comp.get(e, 0.0) * A0_REF[e] for e in ELEMENTS)


def linear_tm(comp: dict[str, float]) -> float:
    """Linear mixing rule: Tm = Σᵢ xᵢ Tmᵢ  [K]  (preliminary estimate)"""
    return sum(comp.get(e, 0.0) * TM[e] for e in ELEMENTS)


def composition_sf(comp: dict[str, float]) -> float:
    """Composition-weighted vacancy formation entropy: Sf = Σᵢ xᵢ Sf,i  [kB]"""
    return sum(comp.get(e, 0.0) * SF_REF[e] for e in ELEMENTS)


def make_temperature_grid(Tm: float) -> list[int]:
    """
    Return N_TEMPS simulation temperatures in [max(T_MIN_ABS, T_FRAC_MIN·Tm),
    T_FRAC_MAX·Tm], each rounded to the nearest T_ROUND_K K.

    T_FRAC_MAX = 0.80 is intentional: ADP Tm ≠ GRACE Tm.  Running above
    0.8·Tm_GRACE risks melting the alloy during ADP diffusion runs.
    The Arrhenius fit (R²>0.99 achieved) allows reliable extrapolation to Tm.
    """
    T_lo = max(T_MIN_ABS, T_FRAC_MIN * Tm)
    T_hi = T_FRAC_MAX * Tm
    if T_lo >= T_hi:
        T_lo = 0.5 * T_hi
    raw  = np.linspace(T_lo, T_hi, N_TEMPS)
    grid = sorted({int(round(t / T_ROUND_K) * T_ROUND_K) for t in raw})
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  Ef calculation via LAMMPS (ADP potential, serial)
# ══════════════════════════════════════════════════════════════════════════════

def _lammps_perfect_input(comp: dict[str, float], a0: float,
                           pot: Path, N: int) -> str:
    """
    LAMMPS input: build BCC alloy, minimise, print PERFECT_ENERGY N E Lx,
    and write perfect.data for subsequent vacancy runs.
    """
    fracs = sequential_fracs(comp)
    mass_lines = "\n".join(
        f"mass {i+1} {MASSES[e]}  # {e}" for i, e in enumerate(ELEMENTS)
    )
    set_lines = "".join(
        f"set group all type/fraction {ELEMENTS.index(e)+1} "
        f"{f:.8f} {abs(hash(e)) % 90000 + 10000}\n"
        for e, f in fracs.items() if f > 1e-9
    )
    return f"""units metal
boundary p p p
atom_style atomic
lattice bcc {a0:.5f}
region box block 0 {N} 0 {N} 0 {N}
create_box 6 box
create_atoms 1 box
{mass_lines}
{set_lines}
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
    """
    LAMMPS input: read perfect.data, delete atom_id, minimise,
    print VACANCY_ENERGY atom_id N-1 E.
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


def _run_lammps(inp: Path, cwd: Path, timeout: int = 300) -> str:
    cmd = LAMMPS_SERIAL_CMD.format(input_file=inp.name).split()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                            text=True, timeout=timeout)
    out = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS exit {result.returncode}:\n{out[-2000:]}")
    return out


def compute_ef(comp: dict[str, float], a0: float) -> float:
    """
    Compute mean vacancy formation energy [eV].

    Implements Starikov 2024 Eq. 7, averaged over EF_N_SITES sites:

        <Ef> = mean_i [ E(N−1, site i) ] − (N−1)/N · E(N)

    where E(N) is the perfect-crystal energy and E(N−1, i) is the energy
    after removing atom i.  The chemical potential for the removed atom is
    taken as E(N)/N (energy per atom in the alloy).
    """
    # Skip LAMMPS for HCP elements — ADP gives unphysical BCC energies
    active = [e for e in ELEMENTS if comp.get(e, 0.0) > 0.99]
    if len(active) == 1 and active[0] not in _BCC_STABLE:
        return _EF_FALLBACK[active[0]]

    pot = POTENTIAL_FILE
    with tempfile.TemporaryDirectory(prefix="ef_") as tmp:
        work = Path(tmp)
        shutil.copy(pot, work / pot.name)

        # Perfect crystal
        inp = work / "perfect.in"
        inp.write_text(_lammps_perfect_input(comp, a0, pot, EF_CELL_SIZE))
        out = _run_lammps(inp, work)
        N, E0 = _parse_perfect(out)

        # Vacancy calculations at EF_N_SITES evenly spaced atom IDs
        step    = max(1, N // EF_N_SITES)
        ids     = list(range(1, N + 1, step))[:EF_N_SITES]
        E_vacs  = []
        for aid in ids:
            inp2 = work / f"vac_{aid}.in"
            inp2.write_text(_lammps_vacancy_input(aid, pot))
            E_vacs.append(_parse_vacancy(_run_lammps(inp2, work)))

    if not E_vacs:
        raise RuntimeError("No vacancy energies collected")

    # Starikov Eq. 7:  <Ef> = mean(E_i) − (N−1)/N · E0
    return float(np.mean(E_vacs)) - (N - 1) / N * E0


def _parse_perfect(out: str) -> tuple[int, float]:
    for line in out.splitlines():
        if line.strip().startswith("PERFECT_ENERGY"):
            p = line.split()
            return int(p[1]), float(p[2])
    raise ValueError("PERFECT_ENERGY not found in LAMMPS output")


def _parse_vacancy(out: str) -> float:
    for line in out.splitlines():
        if line.strip().startswith("VACANCY_ENERGY"):
            return float(line.split()[2])
    raise ValueError("VACANCY_ENERGY not found in LAMMPS output")


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

# Fallback Ef values [eV] if LAMMPS calculation fails.
# Source: DFT literature (approximate; update when site-specific data available)
_BCC_STABLE = {"W", "Mo", "Nb", "Ta"}  # Zr, Ti are HCP at 0K — skip ADP

_EF_FALLBACK = {
    "W":  3.56,   # Starikov 2024
    "Mo": 2.88,   # Starikov 2024
    "Nb": 2.91,   # Starikov 2024
    "Zr": 2.05,   # DFT-GGA BCC
    "Ti": 1.97,   # DFT-GGA BCC
    "Ta": 3.03,   # Satta et al. PRB 1999
}


def compute_all_constants(compositions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a0, Tm, Ef, Sf, C0, T_grid for every composition.

    Writes:
      results/constants/<comp_id>.json   (one file per composition)
      results/constants_all.csv          (all compositions, one row each)
    """
    const_dir = RESULTS_DIR / "constants"
    const_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for _, row in compositions_df.iterrows():
        comp_id = row["comp_id"]
        comp    = row_to_comp(row)
        print(f"[constants] {comp_id}  active={active_elements(comp)}")

        a0_vegard = vegard_a0(comp)
        a0        = compute_a0_lammps(comp, a0_vegard)
        print(f"  a0: Vegard={a0_vegard:.4f} Å  LAMMPS={a0:.4f} Å")
        Tm     = linear_tm(comp)
        Sf     = composition_sf(comp)
        C0     = float(np.exp(Sf))      # C0 = exp(Sf/kB); Sf in kB units
        T_grid = make_temperature_grid(Tm)

        try:
            Ef = compute_ef(comp, a0)
        except Exception as exc:
            print(f"  [WARN] Ef LAMMPS failed: {exc}")
            Ef = sum(comp.get(e, 0.0) * _EF_FALLBACK.get(e, 2.5)
                     for e in ELEMENTS)
            print(f"  [WARN] Using fallback Ef = {Ef:.4f} eV")

        rec = {
            "comp_id":   comp_id,
            "a0_vegard": round(a0, 5),
            "Tm":        round(Tm, 1),
            "Ef":        round(Ef, 5),
            "Sf":        round(Sf, 4),
            "C0":        round(C0, 6),
            "T_grid":    T_grid,
            **{f"x_{e}": round(comp.get(e, 0.0), 6) for e in ELEMENTS},
        }

        (const_dir / f"{comp_id}.json").write_text(json.dumps(rec, indent=2))
        records.append(rec)
        print(f"  Tm={Tm:.0f} K  Ef={Ef:.4f} eV  C0={C0:.4f}  T_grid={T_grid}")

    df = pd.DataFrame(records)
    out = RESULTS_DIR / "constants_all.csv"
    df.to_csv(out, index=False)
    print(f"[constants] Saved {len(df)} rows  →  {out}")
    return df
