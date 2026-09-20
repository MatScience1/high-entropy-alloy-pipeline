"""Compute the vacancy formation entropy Sf [kB] for each pure BCC element.

Uses phonopy with the ADP potential.  Run once before Stage 2.

Method (Starikov 2024, Sec. III.A)
----------------------------------
.. math::

    S_f = S_{\\text{vac}} - \\frac{N-1}{N} S_{\\text{perf}}
        = S_{\\text{vac}} - (N-1) S_{\\text{id}}

where :math:`S_{\\text{id}} = S_{\\text{perf}} / N` is the entropy per atom.

For each element:

1. Relax the perfect BCC supercell.
2. Generate phonopy displaced structures and collect forces (``FORCE_SETS``).
3. Read ``thermal_properties.yaml`` to obtain the entropy at ``T0``.
4. Repeat for the vacancy supercell.
5. ``Sf = S_vac - (N-1) * S_id``.

BCC-stable elements (W, Mo, Nb, Ta) use ``T0 = 300 K``.  BCC-unstable
elements at 300 K (Ti, Zr) use a temperature inside their BCC stability
region (1200 K and 1100 K respectively); imaginary frequencies may appear and
should be checked in ``thermal_properties.yaml``.

Output
------
``results/Sf_phonopy.txt``.  Update ``SF_REF`` in :mod:`config` with these
values.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402 - path bootstrap must precede this import
    A0_REF,
    ELEMENTS,
    LAMMPS_EXE,
    MASSES,
    POTENTIAL_FILE,
    RESULTS_DIR,
)
from logging_config import configure_logging, get_logger  # noqa: E402
from reporting import log_mapping  # noqa: E402

logger = get_logger(__name__)

# Temperature for the phonopy Sf calculation per element [K].
T0_PER_ELEMENT: dict[str, int] = {
    "W": 300,
    "Mo": 300,
    "Nb": 300,
    "Zr": 1100,   # BCC stable above 863 C
    "Ti": 1200,   # BCC stable above 882 C
    "Ta": 300,
}

SUPERCELL: list[int] = [6, 6, 6]   # 6^3 x 2 = 432 atoms
DISP: float = 0.02                 # displacement amplitude [Angstrom]
MESH: list[int] = [8, 8, 8]
_GAS_CONSTANT: float = 8.314462618  # J mol^-1 K^-1


# ══════════════════════════════════════════════════════════════════════════════
#  LAMMPS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _write_data(
    path: Path,
    coords: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    n: int,
    mass: float,
) -> None:
    """Write a minimal LAMMPS data file for a single-species supercell."""
    with open(path, "w") as fh:
        fh.write(
            f"LAMMPS data\n\n{n} atoms\n1 atom types\n\n"
            f"0.0 {lx:.12f} xlo xhi\n0.0 {ly:.12f} ylo yhi\n"
            f"0.0 {lz:.12f} zlo zhi\n\nMasses\n\n1 {mass}\n\nAtoms\n\n"
        )
        for i, p in enumerate(coords, 1):
            fh.write(f"{i} 1 {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")


def _write_input(
    path: Path,
    data_file: str,
    dump_file: str,
    element: str,
    relax: bool = False,
) -> None:
    """Write a LAMMPS input that either relaxes positions or dumps forces."""
    pot = POTENTIAL_FILE.name
    with open(path, "w") as fh:
        fh.write(
            "units metal\nboundary p p p\natom_style atomic\n"
            "atom_modify map array sort 0 0.0\n"
            f"read_data {data_file}\n"
            f"pair_style adp\npair_coeff * * {pot} {element}\n"
        )
        if relax:
            fh.write(
                "min_style fire\nminimize 1e-12 1e-12 10000 100000\n"
                f"dump 1 all custom 1 {dump_file} id x y z\n"
            )
        else:
            fh.write(f"dump 1 all custom 1 {dump_file} id fx fy fz\n")
        fh.write("dump_modify 1 sort id\nrun 0\n")


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess, raising on failure."""
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _parse_dump(path: Path, n: int) -> np.ndarray:
    """Parse the final ``n`` rows of a LAMMPS dump into an ``(n, 3)`` array."""
    with open(path) as fh:
        lines = fh.readlines()
    data = np.zeros((n, 3))
    for line in lines[-n:]:
        parts = line.split()
        data[int(parts[0]) - 1] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return data


def _read_poscar_direct(path: Path, n: int, lx: float, ly: float, lz: float) -> np.ndarray:
    """Read Cartesian coordinates from a phonopy-generated POSCAR file."""
    with open(path) as fh:
        lines = fh.readlines()
    coord_type = lines[7].strip()[0].lower()
    coords = np.array([line.split()[:3] for line in lines[8:8 + n]], dtype=float)
    if coord_type == "d":
        return coords * np.array([lx, ly, lz])
    return coords * float(lines[1])


def _bcc_supercell(a0: float, nx: int, ny: int, nz: int) -> list[list[float]]:
    """Return Cartesian coordinates of an ``nx x ny x nz`` BCC supercell."""
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return [
        ((np.array([i, j, k]) + b) * a0).tolist()
        for i in range(nx) for j in range(ny) for k in range(nz) for b in basis
    ]


def _parse_entropy(yaml_path: Path, t0: float) -> float:
    """Return the vibrational entropy [J K^-1] at ``t0`` from a phonopy YAML."""
    in_block = False
    with open(yaml_path) as fh:
        for line in fh:
            if "temperature:" in line:
                try:
                    t_val = float(line.split(":")[1])
                    in_block = abs(t_val - t0) < 0.1
                except ValueError:
                    continue
            elif in_block and "entropy:" in line and "free_energy" not in line:
                return float(line.split(":")[-1].strip())
    raise RuntimeError(f"Entropy not found in {yaml_path} at T={t0}")


# ══════════════════════════════════════════════════════════════════════════════
#  Core Sf computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_S_supercell(
    label: str,
    coords: list[list[float]],
    n: int,
    element: str,
    lx: float,
    ly: float,
    lz: float,
    workdir: Path,
    t0: float,
    nosym: bool = False,
) -> float:
    """Return the vibrational entropy [kB/supercell] at ``t0`` via phonopy.

    Parameters
    ----------
    label : str
        Human-readable label used in log messages (``"perfect"``/``"vacancy"``).
    coords : list of list of float
        Atomic coordinates [Angstrom].
    n : int
        Number of atoms.
    element : str
        Element symbol.
    lx, ly, lz : float
        Box dimensions [Angstrom].
    workdir : pathlib.Path
        Working directory for the phonopy run.
    t0 : float
        Temperature at which the entropy is evaluated [K].
    nosym : bool, optional
        Disable symmetry reduction (used for the vacancy supercell).

    Returns
    -------
    float
        Vibrational entropy [kB/supercell].
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(POTENTIAL_FILE, workdir / POTENTIAL_FILE.name)
    mass = MASSES[element]
    box = np.array([lx, ly, lz])

    # Relax the structure.
    _write_data(workdir / "init.data", coords, lx, ly, lz, n, mass)
    _write_input(workdir / "in.relax", "init.data", "relax.dump", element, relax=True)
    _run([LAMMPS_EXE, "-in", "in.relax"], workdir)
    relaxed = _parse_dump(workdir / "relax.dump", n)

    # Residual forces on the relaxed structure.
    _write_data(workdir / "relaxed.data", relaxed, lx, ly, lz, n, mass)
    _write_input(workdir / "in.res", "relaxed.data", "res.dump", element, relax=False)
    _run([LAMMPS_EXE, "-in", "in.res"], workdir)
    res_forces = _parse_dump(workdir / "res.dump", n)

    # POSCAR in direct coordinates.
    with open(workdir / "POSCAR", "w") as fh:
        fh.write(
            f"{element} {label}\n1.0\n{lx:.10f} 0 0\n0 {ly:.10f} 0\n0 0 {lz:.10f}\n"
            f"{element}\n{n}\nDirect\n"
        )
        for p in relaxed:
            fh.write(f"{p[0]/lx:.10f} {p[1]/ly:.10f} {p[2]/lz:.10f}\n")

    # Generate displaced structures.
    nosym_flag = ["--nosym"] if nosym else []
    _run(
        ["phonopy", "--dim=1 1 1", f"--amplitude={DISP}",
         "--tolerance=1e-2", "-d"] + nosym_flag,
        workdir,
    )

    disp_poscars = sorted(workdir.glob("POSCAR-*"))
    logger.info("[%s] %d displaced structures", label, len(disp_poscars))

    # Collect forces for each displacement.
    for dp in disp_poscars:
        d_id = dp.name.split("-")[-1]
        d_dir = workdir / f"disp-{d_id}"
        d_dir.mkdir(exist_ok=True)
        shutil.copy(POTENTIAL_FILE, d_dir / POTENTIAL_FILE.name)

        coords_d = _read_poscar_direct(dp, n, lx, ly, lz)
        _write_data(d_dir / "d.data", coords_d, lx, ly, lz, n, mass)
        _write_input(d_dir / "in.f", "d.data", "f.dump", element, relax=False)
        _run([LAMMPS_EXE, "-in", "in.f"], d_dir)

    # Assemble FORCE_SETS.
    with open(workdir / "FORCE_SETS", "w") as fh:
        fh.write(f"{n}\n{len(disp_poscars)}\n")
        for dp in disp_poscars:
            d_id = dp.name.split("-")[-1]
            forces = _parse_dump(workdir / f"disp-{d_id}" / "f.dump", n) - res_forces
            coords_d = _read_poscar_direct(dp, n, lx, ly, lz)
            diff = coords_d - relaxed
            diff -= np.round(diff / box) * box
            atom_idx = int(np.argmax(np.linalg.norm(diff, axis=1)))
            vec = diff[atom_idx]
            fh.write(f"\n{atom_idx+1}\n{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n")
            for fx, fy, fz in forces:
                fh.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")

    # Thermal properties.
    fc_sym = [] if nosym else ["--fc-symmetry"]
    mesh_str = f"--mesh={MESH[0]} {MESH[1]} {MESH[2]}"
    cmd = (
        ["phonopy", "--dim=1 1 1", mesh_str]
        + fc_sym + nosym_flag
        + [f"--tmin={t0}", f"--tmax={t0}", "--tstep=1", "--tolerance=1e-2", "-t"]
    )
    try:
        _run(cmd, workdir)
    except subprocess.CalledProcessError:
        _run(
            ["phonopy", "--dim=1 1 1", mesh_str] + nosym_flag
            + [f"--tmin={t0}", f"--tmax={t0}", "--tstep=1", "--tolerance=1e-2", "-t"],
            workdir,
        )

    s_joule = _parse_entropy(workdir / "thermal_properties.yaml", t0)
    s_kb = s_joule / _GAS_CONSTANT
    logger.info("[%s] S = %.4f kB/supercell at %.0f K", label, s_kb, t0)
    return s_kb


def _get_relaxed_a0(element: str, a0_guess: float, workdir: Path) -> float:
    """Return the ADP-relaxed lattice parameter for a pure element.

    Parameters
    ----------
    element : str
        Element symbol.
    a0_guess : float
        Initial lattice parameter [Angstrom].
    workdir : pathlib.Path
        Working directory for the LAMMPS run.

    Returns
    -------
    float
        Relaxed lattice parameter [Angstrom], or ``a0_guess`` on failure.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(POTENTIAL_FILE, workdir / POTENTIAL_FILE.name)
    inp = f"""units metal
boundary p p p
atom_style atomic
lattice bcc {a0_guess}
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 {MASSES[element]}
pair_style adp
pair_coeff * * {POTENTIAL_FILE.name} {element}
fix 1 all box/relax iso 0.0 vmax 0.001
minimize 1e-12 1e-12 10000 10000
variable a0 equal lx
print "ADP_A0 ${{a0}}"
"""
    (workdir / "a0.in").write_text(inp)
    _run([LAMMPS_EXE, "-in", "a0.in"], workdir)
    for line in (workdir / "log.lammps").read_text().splitlines():
        if line.startswith("ADP_A0"):
            return float(line.split()[1])
    return a0_guess


def compute_sf_element(element: str, workdir_base: Path) -> float:
    """Return the vacancy formation entropy Sf [kB] for one pure element.

    Implements ``Sf = S_vac - (N-1) * S_id`` with
    ``S_id = S_perfect / N`` (Starikov 2024, Sec. III.A).

    Parameters
    ----------
    element : str
        Element symbol.
    workdir_base : pathlib.Path
        Base directory for the phonopy working directories.

    Returns
    -------
    float
        Vacancy formation entropy [kB].
    """
    a0 = _get_relaxed_a0(element, A0_REF[element], workdir_base / f"{element}_a0")
    logger.info("[%s] Relaxed ADP a0 = %.5f A (ref %.5f A)", element, a0, A0_REF[element])

    t0 = T0_PER_ELEMENT[element]
    nx, ny, nz = SUPERCELL
    lx, ly, lz = nx * a0, ny * a0, nz * a0
    n = nx * ny * nz * 2   # 432

    all_coords = _bcc_supercell(a0, nx, ny, nz)
    vac_coords = all_coords[1:]   # remove atom 0

    logger.info("[%s] T0=%d K  a0=%.5f A  N=%d", element, t0, a0, n)

    s_perf = _compute_S_supercell(
        "perfect", all_coords, n, element, lx, ly, lz,
        workdir_base / f"{element}_perfect", t0,
    )
    s_id = s_perf / n   # entropy per atom

    s_vac = _compute_S_supercell(
        "vacancy", vac_coords, n - 1, element, lx, ly, lz,
        workdir_base / f"{element}_vacancy", t0,
    )

    sf = s_vac - (n - 1) * s_id
    logger.info(
        "[%s] Sf = %.6f kB (S_id=%.4f, S_vac=%.4f)", element, sf, s_id, s_vac
    )
    return sf


def run_all_elements(
    elements: list[str] | None = None,
    workdir_base: Path | None = None,
) -> dict[str, float]:
    """Compute Sf for each element, skipping those already in the output file.

    Parameters
    ----------
    elements : list of str, optional
        Elements to process.  Defaults to all six pipeline elements.
    workdir_base : pathlib.Path, optional
        Base directory for phonopy working directories.  Defaults to
        ``results/sf_phonopy``.

    Returns
    -------
    dict of str to float
        Mapping of element symbol to Sf [kB].
    """
    if elements is None:
        elements = ELEMENTS
    if workdir_base is None:
        workdir_base = RESULTS_DIR / "sf_phonopy"
    workdir_base.mkdir(parents=True, exist_ok=True)

    out_file = RESULTS_DIR / "Sf_phonopy.txt"
    done: dict[str, float] = {}
    if out_file.exists():
        with open(out_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 2:
                    done[parts[0]] = float(parts[1])

    results: dict[str, float] = dict(done)
    for element in elements:
        if element in done:
            logger.info("[%s] already computed: Sf = %.6f kB - skip", element, done[element])
            continue
        try:
            sf = compute_sf_element(element, workdir_base)
            results[element] = sf
            # Append immediately so the run is resume-safe.
            with open(out_file, "a") as fh:
                fh.write(f"{element}  {sf:.6f}\n")
        except Exception as exc:  # noqa: BLE001 - continue with other elements
            logger.error("[%s] FAILED: %s", element, exc)

    log_mapping(logger, results, title="Sf_phonopy results [kB]:")
    logger.info("Update SF_REF in config.py with these values. File: %s", out_file)
    return results


def main() -> None:
    """Command-line entry point for the phonopy Sf calculation."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Compute vacancy formation entropy Sf via phonopy + ADP."
    )
    parser.add_argument(
        "--elements", nargs="+", default=None,
        help="Subset of elements, e.g. W Mo Ta (default: all six).",
    )
    args = parser.parse_args()
    run_all_elements(args.elements)


if __name__ == "__main__":
    main()
