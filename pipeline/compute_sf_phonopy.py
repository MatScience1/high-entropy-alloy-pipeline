"""
pipeline/compute_sf_phonopy.py
==============================
Compute vacancy formation entropy Sf [kB] for each pure BCC element
using phonopy + ADP potential. Run once before Stage 2.

Method (Starikov 2024, Sec. III.A):
    Sf = S_vac_supercell - N_vac/N_perf * S_perf_supercell
       = S_vac - N_vac * S_id_per_atom

For each element:
    1. Relax perfect BCC supercell
    2. Phonopy displaced structures -> FORCE_SETS
    3. thermal_properties.yaml -> S_id (entropy per atom at T0)
    4. Repeat for vacancy supercell (--nosym)
    5. Sf = S_vac - (N-1) * S_id

BCC-stable elements (W, Mo, Nb, Ta): T0 = 300 K
BCC-unstable at 300K (Ti, Zr): T0 at BCC stability region (1200 K / 1100 K)
    Warning: imaginary frequencies may appear — check thermal_properties.yaml

Output: results/Sf_phonopy.txt  (update SF_REF in config.py with these values)
"""

from __future__ import annotations
import os, shutil, subprocess, glob
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    A0_REF, ELEMENTS, LAMMPS_EXE, MASSES, POTENTIAL_FILE, RESULTS_DIR,
)

# Temperature for phonopy Sf calculation per element [K]
# BCC-stable elements use 300 K; Ti and Zr use temperature inside BCC phase
T0_PER_ELEMENT = {
    "W":  300,
    "Mo": 300,
    "Nb": 300,
    "Zr": 1100,   # BCC stable above 863°C; use 1100K
    "Ti": 1200,   # BCC stable above 882°C; use 1200K
    "Ta": 300,
}

SUPERCELL = [6, 6, 6]   # 6³×2 = 432 atoms
DISP      = 0.02         # Angstrom
MESH      = [8, 8, 8]


# ── LAMMPS helpers ─────────────────────────────────────────────────────────────

def _write_data(path: Path, coords, lx, ly, lz, n, mass):
    with open(path, "w") as f:
        f.write(f"LAMMPS data\n\n{n} atoms\n1 atom types\n\n"
                f"0.0 {lx:.12f} xlo xhi\n0.0 {ly:.12f} ylo yhi\n"
                f"0.0 {lz:.12f} zlo zhi\n\nMasses\n\n1 {mass}\n\nAtoms\n\n")
        for i, p in enumerate(coords, 1):
            f.write(f"{i} 1 {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}\n")


def _write_input(path: Path, data_file, dump_file, element, relax=False):
    pot = POTENTIAL_FILE.name
    with open(path, "w") as f:
        f.write("units metal\nboundary p p p\natom_style atomic\n"
                "atom_modify map array sort 0 0.0\n"
                f"read_data {data_file}\n"
                f"pair_style adp\npair_coeff * * {pot} {element}\n")
        if relax:
            f.write("min_style fire\nminimize 1e-12 1e-12 10000 100000\n"
                    f"dump 1 all custom 1 {dump_file} id x y z\n")
        else:
            f.write(f"dump 1 all custom 1 {dump_file} id fx fy fz\n")
        f.write("dump_modify 1 sort id\nrun 0\n")


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _parse_dump(path: Path, n):
    data = np.zeros((n, 3))
    lines = open(path).readlines()
    for line in lines[-n:]:
        p = line.split()
        data[int(p[0]) - 1] = [float(p[1]), float(p[2]), float(p[3])]
    return data


def _read_poscar_direct(path: Path, n, lx, ly, lz):
    lines = open(path).readlines()
    coord_type = lines[7].strip()[0].lower()
    coords = np.array([l.split()[:3] for l in lines[8:8+n]], dtype=float)
    return coords * np.array([lx, ly, lz]) if coord_type == "d" else coords * float(lines[1])


def _bcc_supercell(a0, nx, ny, nz):
    basis = np.array([[0., 0., 0.], [.5, .5, .5]])
    return [((np.array([i, j, k]) + b) * a0).tolist()
            for i in range(nx) for j in range(ny) for k in range(nz) for b in basis]


def _parse_entropy(yaml_path: Path, T0: float) -> float:
    in_block = False
    for line in open(yaml_path):
        if "temperature:" in line:
            try:
                t_val = float(line.split(":")[1])
                in_block = abs(t_val - T0) < 0.1
            except ValueError:
                pass
        elif in_block and "entropy:" in line and "free_energy" not in line:
            return float(line.split(":")[-1].strip())
    raise RuntimeError(f"Entropy not found in {yaml_path} at T={T0}")


# ── Core Sf computation ────────────────────────────────────────────────────────

def _compute_S_supercell(label: str, coords, n: int, element: str,
                          lx, ly, lz, workdir: Path, T0: float,
                          nosym: bool = False) -> float:
    """
    Return vibrational entropy [kB/supercell] at T0 using phonopy.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(POTENTIAL_FILE, workdir / POTENTIAL_FILE.name)
    mass = MASSES[element]
    box  = np.array([lx, ly, lz])

    # relax
    _write_data(workdir/"init.data", coords, lx, ly, lz, n, mass)
    _write_input(workdir/"in.relax", "init.data", "relax.dump", element, relax=True)
    _run([LAMMPS_EXE, "-in", "in.relax"], workdir)
    relaxed = _parse_dump(workdir/"relax.dump", n)

    # residual forces
    _write_data(workdir/"relaxed.data", relaxed, lx, ly, lz, n, mass)
    _write_input(workdir/"in.res", "relaxed.data", "res.dump", element, relax=False)
    _run([LAMMPS_EXE, "-in", "in.res"], workdir)
    res_forces = _parse_dump(workdir/"res.dump", n)

    # POSCAR (Direct)
    with open(workdir/"POSCAR", "w") as f:
        f.write(f"{element} {label}\n1.0\n{lx:.10f} 0 0\n0 {ly:.10f} 0\n0 0 {lz:.10f}\n"
                f"{element}\n{n}\nDirect\n")
        for p in relaxed:
            f.write(f"{p[0]/lx:.10f} {p[1]/ly:.10f} {p[2]/lz:.10f}\n")

    # displaced structures
    nosym_flag = ["--nosym"] if nosym else []
    _run(["phonopy", "--dim=1 1 1", f"--amplitude={DISP}",
          "--tolerance=1e-2", "-d"] + nosym_flag, workdir)

    disp_poscars = sorted((workdir).glob("POSCAR-*"))
    print(f"  [{label}] {len(disp_poscars)} displaced structures")

    # forces for each displacement
    for dp in disp_poscars:
        d_id  = dp.name.split("-")[-1]
        d_dir = workdir / f"disp-{d_id}"
        d_dir.mkdir(exist_ok=True)
        shutil.copy(POTENTIAL_FILE, d_dir / POTENTIAL_FILE.name)

        coords_d = _read_poscar_direct(dp, n, lx, ly, lz)
        _write_data(d_dir/"d.data", coords_d, lx, ly, lz, n, mass)
        _write_input(d_dir/"in.f", "d.data", "f.dump", element, relax=False)
        _run([LAMMPS_EXE, "-in", "in.f"], d_dir)

    # FORCE_SETS
    with open(workdir/"FORCE_SETS", "w") as f:
        f.write(f"{n}\n{len(disp_poscars)}\n")
        for dp in disp_poscars:
            d_id     = dp.name.split("-")[-1]
            forces   = _parse_dump(workdir/f"disp-{d_id}"/"f.dump", n) - res_forces
            coords_d = _read_poscar_direct(dp, n, lx, ly, lz)
            diff     = coords_d - relaxed
            diff    -= np.round(diff / box) * box
            atom_idx = int(np.argmax(np.linalg.norm(diff, axis=1)))
            vec      = diff[atom_idx]
            f.write(f"\n{atom_idx+1}\n{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}\n")
            for fx, fy, fz in forces:
                f.write(f"{fx:.8f} {fy:.8f} {fz:.8f}\n")

    # thermal properties
    fc_sym = [] if nosym else ["--fc-symmetry"]
    mesh_str = f"--mesh={MESH[0]} {MESH[1]} {MESH[2]}"
    cmd = (["phonopy", "--dim=1 1 1", mesh_str]
           + fc_sym + nosym_flag
           + [f"--tmin={T0}", f"--tmax={T0}", "--tstep=1",
              "--tolerance=1e-2", "-t"])
    try:
        _run(cmd, workdir)
    except subprocess.CalledProcessError:
        _run(["phonopy", "--dim=1 1 1", mesh_str] + nosym_flag
             + [f"--tmin={T0}", f"--tmax={T0}", "--tstep=1", 
                "--tolerance=1e-2", "-t"], workdir)

    S_J = _parse_entropy(workdir/"thermal_properties.yaml", T0)
    S_kB = S_J / 8.314462618
    print(f"  [{label}] S = {S_kB:.4f} kB/supercell at {T0:.0f} K")
    return S_kB


def _get_relaxed_a0(element: str, a0_guess: float, workdir: Path) -> float:
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
    (workdir/"a0.in").write_text(inp)
    _run([LAMMPS_EXE, "-in", "a0.in"], workdir)
    for line in (workdir/"log.lammps").read_text().splitlines():
        if line.startswith("ADP_A0"): 
            return float(line.split()[1])
    return a0_guess

def compute_sf_element(element: str, workdir_base: Path) -> float:
    """
    Sf_v [kB] for one pure element via phonopy.

    Formula (Starikov 2024 Sec. III.A):
        Sf = S_vac_supercell - N_vac * S_id_per_atom
    where S_id_per_atom = S_perfect_supercell / N_perfect.
    """
    a0 = _get_relaxed_a0(element, A0_REF[element], workdir_base / f"{element}_a0")
    print(f"[{element}] Relaxed ADP a0 = {a0:.5f} Å (Ref: {A0_REF[element]} Å)")
    T0  = T0_PER_ELEMENT[element]
    nx, ny, nz = SUPERCELL
    lx  = nx * a0
    ly  = ny * a0
    lz  = nz * a0
    N   = nx * ny * nz * 2   # 432

    all_coords  = _bcc_supercell(a0, nx, ny, nz)
    vac_coords  = all_coords[1:]   # remove atom 0

    print(f"\n[{element}] T0={T0} K  a0={a0} Å  N={N}")

    S_perf = _compute_S_supercell("perfect", all_coords, N, element,
                                   lx, ly, lz, workdir_base/f"{element}_perfect", T0)
    S_id   = S_perf / N   # entropy per atom

    S_vac  = _compute_S_supercell("vacancy", vac_coords, N-1, element,
                                   lx, ly, lz, workdir_base/f"{element}_vacancy", T0,
                                   ) #nosym=True for full analysis 

    Sf = S_vac - (N-1) * S_id
    print(f"[{element}] Sf = {Sf:.6f} kB  (S_id={S_id:.4f}, S_vac={S_vac:.4f})")
    return Sf


def run_all_elements(elements=None, workdir_base: Path = None) -> dict[str, float]:
    """
    Compute Sf for each element. Skip elements already in output file.
    """
    if elements is None:
        elements = ELEMENTS
    if workdir_base is None:
        workdir_base = RESULTS_DIR / "sf_phonopy"
    workdir_base.mkdir(parents=True, exist_ok=True)

    out_file = RESULTS_DIR / "Sf_phonopy.txt"
    # Load existing results to allow resume
    done: dict[str, float] = {}
    if out_file.exists():
        for line in open(out_file):
            parts = line.strip().split()
            if len(parts) == 2:
                done[parts[0]] = float(parts[1])

    results = dict(done)
    for elem in elements:
        if elem in done:
            print(f"[{elem}] already computed: Sf = {done[elem]:.6f} kB — skip")
            continue
        try:
            Sf = compute_sf_element(elem, workdir_base)
            results[elem] = Sf
            # Append to file immediately (resume-safe)
            with open(out_file, "a") as f:
                f.write(f"{elem}  {Sf:.6f}\n")
        except Exception as e:
            print(f"[{elem}] FAILED: {e}")

    print("\n=== Sf_phonopy results ===")
    for elem in ELEMENTS:
        val = results.get(elem, float("nan"))
        print(f"  {elem}: {val:.4f} kB")
    print(f"\nUpdate SF_REF in config.py with these values.")
    print(f"File: {out_file}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", nargs="+", default=None,
                    help="Subset of elements, e.g. W Mo Ta")
    args = ap.parse_args()
    run_all_elements(args.elements)
