#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master orchestrator for the WMoNbZrTiTa diffusion pipeline.

Quick start
-----------
    # 1. Edit config.py (paths, TEST_MODE flag)
    # 2. Run stages 0–2 to set up compositions and constants:
    python run_pipeline.py --from_stage 0 --to_stage 2

    # 3. Generate and submit Tm coexistence runs (GRACE MLIP):
    python run_pipeline.py --only_stage 31
    sbatch slurm/submit_Tm_array.sh

    # 4. After Tm jobs finish, patch constants and generate diffusion inputs:
    python run_pipeline.py --only_stage 32
    python run_pipeline.py --only_stage 3
    python run_pipeline.py --only_stage 4
    # sbatch slurm/submit_diffusion.sh   (or submit_all.sh from Stage 4)

    # 5. After diffusion jobs finish, run all analysis:
    python run_pipeline.py --from_stage 5

Stage map
---------
    0   Validate environment (packages, LAMMPS binary, potential file)
    1   Generate compositions     → results/compositions.csv
    2   Compute constants         → results/constants_all.csv  (Tm_rom, Ef, Sf…)
    31  Generate Tm inputs        → runs/*/tm_coexistence/  + slurm/submit_Tm_array.sh
        [manual: sbatch slurm/submit_Tm_array.sh and wait]
    32  Patch Tm into constants   → results/constants_all.csv  (real Tm, new T_grid)
    3   Generate diffusion inputs → runs/*/sim_*/bcc_vac_adv.in
    4   Write submit_diffusion.sh
        [manual: sbatch slurm/submit_diffusion.sh and wait]
    5   Parse MSD
    6   Compute Dv
    7   Compute Cv
    8   Compute D*
    9   Compute SRO
    10  Post-process (polynomial fit, plots)

CLI examples
------------
    python run_pipeline.py                          # full run from stage 0
    python run_pipeline.py --from_stage 5           # analysis only
    python run_pipeline.py --only_stage 31          # Tm input generation only
    python run_pipeline.py --only_stage 32          # Tm patch only
    python run_pipeline.py --from_stage 3 --to_stage 4   # diffusion inputs only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path



import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import (
    LAMMPS_SERIAL_CMD, PIPELINE_DIR, POTENTIAL_FILE,
    RESULTS_DIR, RUNS_DIR, SLURM_PARTITION, TEST_MODE,
)

# ── Stage ordering ─────────────────────────────────────────────────────────────
# 31 = Stage 3b (generate Tm inputs)
# 32 = Stage 3c (patch Tm into constants_all.csv)
# They sit between 2 and 3 in execution order.
_STAGE_ORDER = [0, 1, 2, 31, 32, 3, 4, 5, 6, 7, 8, 9, 10]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _require(path: Path, name: str) -> pd.DataFrame:
    """Load a required CSV or exit with a helpful message."""
    if not path.exists():
        print(f"\n[ERROR] {name} not found at {path}")
        print(f"  Run the stage that produces it first.\n")
        sys.exit(1)
    return pd.read_csv(path)


def _analysis_dirs(comp_id: str) -> tuple[Path, Path, Path]:
    """Return (sim_base, txt_dir, plot_dir) for a composition."""
    txt_dir  = RESULTS_DIR / comp_id / "txt"
    plot_dir = RESULTS_DIR / comp_id / "plots"
    txt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / comp_id, txt_dir, plot_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Stage implementations
# ══════════════════════════════════════════════════════════════════════════════

def _stage_0():
    import re, shutil
    print("=" * 60)
    print("Stage 0: Environment validation")
    ok = True
    for pkg in ["numpy", "pandas", "matplotlib", "scipy", "sklearn"]:
        try:
            __import__(pkg); print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} — pip install {pkg}"); ok = False

    if POTENTIAL_FILE.exists():
        print(f"  [OK] ADP potential: {POTENTIAL_FILE.name}")
    else:
        print(f"  [FAIL] potential not found: {POTENTIAL_FILE}"); ok = False

    m   = re.search(r"(lmp[\w_-]*)", LAMMPS_SERIAL_CMD)
    lmp = m.group(1) if m else LAMMPS_SERIAL_CMD.split()[0]
    if shutil.which(lmp):
        print(f"  [OK] LAMMPS binary: {lmp}")
    else:
        print(f"  [WARN] '{lmp}' not in PATH (expected on compute nodes)")

    print(f"\n  Mode: {'TEST' if TEST_MODE else 'PRODUCTION'}")
    if not ok:
        sys.exit(1)
    print("Stage 0 complete.\n")


def _stage_1() -> pd.DataFrame:
    print("=" * 60)
    print("Stage 1: Generating compositions")
    from pipeline.compositions import generate_compositions
    df = generate_compositions()
    print(f"Stage 1 complete — {len(df)} compositions.\n")
    return df


def _stage_2(comps_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("Stage 2: Computing physical constants (a0, Tm_rom, Ef, Sf, C0)")
    from pipeline.constants import compute_all_constants
    df = compute_all_constants(comps_df)
    print("Stage 2 complete.\n")
    return df


def _stage_31(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> list[dict]:
    """Stage 3b — generate Tm coexistence inputs + SLURM array script."""
    print("=" * 60)
    print("Stage 3b: Generating Tm coexistence inputs (GRACE MLIP)")
    from pipeline.lammps_tm import stage_3b
    job_list = stage_3b(comps_df, consts_df)
    print("Stage 3b complete.\n")
    return job_list


def _stage_32():
    """Stage 3c — patch real Tm + T_grid into constants_all.csv."""
    print("=" * 60)
    print("Stage 3c: Patching constants_all.csv with real Tm from coexistence")
    from pipeline.lammps_tm import stage_3c
    stage_3c()
    print("Stage 3c complete.\n")


def _stage_3(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> list[dict]:
    print("=" * 60)
    print("Stage 3: Generating ADP diffusion inputs (bcc_vac_adv.in)")
    from pipeline.lammps_diffusion import create_run_directories
    job_list = create_run_directories(comps_df, consts_df)
    print(f"Stage 3 complete — {len(job_list)} run directories.\n")
    return job_list


def _stage_4(job_list: list[dict]):
    print("=" * 60)
    print("Stage 4: Writing slurm/submit_diffusion.sh")
    slurm_dir = PIPELINE_DIR / "slurm"
    slurm_dir.mkdir(exist_ok=True)
    script = PIPELINE_DIR / "slurm" / "submit_diffusion.sh"
    lines  = [
        "#!/bin/bash",
        f"#SBATCH --job-name=diffusion_all",
        f"#SBATCH --partition={SLURM_PARTITION}",
        "# Auto-generated — submit from the pipeline root directory",
        "",
    ]
    for job in job_list:
        lines += [f"cd {job['path']}", "sbatch submit.sh", ""]
    script.write_text("\n".join(lines))
    script.chmod(0o755)
    print(f"  Written: {script}")
    print("  Submit:  sbatch slurm/submit_diffusion.sh")
    print("  After all jobs finish: python run_pipeline.py --from_stage 5")
    print("Stage 4 complete.\n")


def _stage_5(comps_df: pd.DataFrame):
    print("=" * 60); print("Stage 5: Parsing MSD from LAMMPS logs")
    from analysis.msd import run_all
    for _, crow in comps_df.iterrows():
        cid = crow["comp_id"]
        sim_base, _, _ = _analysis_dirs(cid)
        res_base = RESULTS_DIR / cid / "sim_x"
        print(f"  {cid}")
        run_all(sim_base, res_base)
    print("Stage 5 complete.\n")


def _stage_6(comps_df: pd.DataFrame, consts_df: pd.DataFrame):
    print("=" * 60); print("Stage 6: Computing Dv (vacancy diffusion)")
    from analysis.Dv import process
    for _, crow in comps_df.iterrows():
        cid     = crow["comp_id"]
        sim_base, txt_dir, plot_dir = _analysis_dirs(cid)
        res_base = RESULTS_DIR / cid / "sim_x"
        cst      = consts_df[consts_df["comp_id"] == cid]
        tm_over  = ({e: float(cst.iloc[0]["Tm"])
                     for e in ["W","Mo","Nb","Zr","Ti","Ta"]}
                    if not cst.empty else None)
        print(f"  {cid}")
        process(sim_base, res_base, txt_dir, plot_dir, tm_over)
    print("Stage 6 complete.\n")


def _stage_7(comps_df: pd.DataFrame, consts_df: pd.DataFrame):
    print("=" * 60); print("Stage 7: Computing Cv (vacancy concentration)")
    from analysis.Cv import run
    for _, crow in comps_df.iterrows():
        cid = crow["comp_id"]
        cst = consts_df[consts_df["comp_id"] == cid]
        if cst.empty:
            continue
        c      = cst.iloc[0]
        T_grid = json.loads(c["T_grid"]) if isinstance(c["T_grid"], str) \
                 else list(c["T_grid"])
        _, txt_dir, _ = _analysis_dirs(cid)
        print(f"  {cid}")
        run(float(c["Ef"]), float(c["C0"]), T_grid, txt_dir, comp_label=cid)
    print("Stage 7 complete.\n")


def _stage_8(comps_df: pd.DataFrame):
    print("=" * 60); print("Stage 8: Computing D* (tracer self-diffusion)")
    from analysis.D2 import run
    for _, crow in comps_df.iterrows():
        cid  = crow["comp_id"]
        comp = {e: float(crow[f"x_{e}"]) for e in ["W","Mo","Nb","Zr","Ti","Ta"]}
        _, txt_dir, plot_dir = _analysis_dirs(cid)
        print(f"  {cid}")
        try:
            run(comp, txt_dir, plot_dir, comp_label=cid)
        except Exception as exc:
            print(f"  [WARN] D2 failed for {cid}: {exc}")
    print("Stage 8 complete.\n")


def _stage_9(comps_df: pd.DataFrame):
    print("=" * 60); print("Stage 9: Computing SRO (Warren-Cowley parameters)")
    from analysis.sro import process
    for _, crow in comps_df.iterrows():
        cid = crow["comp_id"]
        sim_base, txt_dir, plot_dir = _analysis_dirs(cid)
        print(f"  {cid}")
        process(sim_base, txt_dir, plot_dir, comp_label=cid)
    print("Stage 9 complete.\n")


def _stage_10():
    print("=" * 60); print("Stage 10: Post-processing and plots")
    from analysis.postprocess import run_postprocess
    run_postprocess()
    print("Stage 10 complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parsing + dispatch
# ══════════════════════════════════════════════════════════════════════════════

def _parse_run_stages(args: argparse.Namespace) -> list[int]:
    if args.only_stage is not None:
        return [args.only_stage]
    idx = {s: i for i, s in enumerate(_STAGE_ORDER)}
    start = idx.get(args.from_stage)
    end   = idx.get(args.to_stage)
    if start is None:
        print(f"[ERROR] Unknown --from_stage {args.from_stage}"); sys.exit(1)
    if end is None:
        print(f"[ERROR] Unknown --to_stage {args.to_stage}");   sys.exit(1)
    return _STAGE_ORDER[start: end + 1]


def main():
    ap = argparse.ArgumentParser(
        description="WMoNbZrTiTa diffusion pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--from_stage", type=int, default=0,
                    metavar="N",
                    help="Start from stage N (default 0). "
                         "Use 31 for Stage 3b, 32 for Stage 3c.")
    ap.add_argument("--to_stage", type=int, default=10,
                    metavar="N", help="Stop after stage N (default 10).")
    ap.add_argument("--only_stage", type=int, default=None,
                    metavar="N", help="Run exactly one stage.")
    args = ap.parse_args()

    run_stages = _parse_run_stages(args)
    print(f"Running stages: {run_stages}\n")

    # Cached DataFrames — loaded lazily and shared across stages
    comp_csv  = RESULTS_DIR / "compositions.csv"
    const_csv = RESULTS_DIR / "constants_all.csv"
    job_csv   = RESULTS_DIR / "job_list.csv"

    comps_df = consts_df = None
    job_list = None
    t0 = time.time()

    for s in run_stages:

        if s == 0:
            _stage_0()

        elif s == 1:
            comps_df = _stage_1()

        elif s == 2:
            if comps_df is None:  comps_df  = _require(comp_csv,  "compositions.csv")
            consts_df = _stage_2(comps_df)

        elif s == 31:
            if comps_df  is None: comps_df  = _require(comp_csv,  "compositions.csv")
            if consts_df is None: consts_df = _require(const_csv, "constants_all.csv")
            _stage_31(comps_df, consts_df)

        elif s == 32:
            _stage_32()
            consts_df = None      # force reload of patched CSV at next use

        elif s == 3:
            if comps_df  is None: comps_df  = _require(comp_csv,  "compositions.csv")
            if consts_df is None: consts_df = _require(const_csv, "constants_all.csv")
            job_list = _stage_3(comps_df, consts_df)

        elif s == 4:
            if job_list is None:
                job_list = _require(job_csv, "job_list.csv").to_dict("records")
            _stage_4(job_list)

        elif s == 5:
            if comps_df is None: comps_df = _require(comp_csv, "compositions.csv")
            _stage_5(comps_df)

        elif s == 6:
            if comps_df  is None: comps_df  = _require(comp_csv,  "compositions.csv")
            if consts_df is None: consts_df = _require(const_csv, "constants_all.csv")
            _stage_6(comps_df, consts_df)

        elif s == 7:
            if comps_df  is None: comps_df  = _require(comp_csv,  "compositions.csv")
            if consts_df is None: consts_df = _require(const_csv, "constants_all.csv")
            _stage_7(comps_df, consts_df)

        elif s == 8:
            if comps_df is None: comps_df = _require(comp_csv, "compositions.csv")
            _stage_8(comps_df)

        elif s == 9:
            if comps_df is None: comps_df = _require(comp_csv, "compositions.csv")
            _stage_9(comps_df)

        elif s == 10:
            _stage_10()

        else:
            print(f"[WARN] Unknown stage {s} — skipped")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
