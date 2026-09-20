#!/usr/bin/env python3
"""Master orchestrator for the WMoNbZrTiTa diffusion pipeline.

Quick start
-----------
::

    # 1. Edit config.py (paths, SLURM settings)
    # 2. Run stages 0-2 to set up compositions and constants:
    python run_pipeline.py --from_stage 0 --to_stage 2

    # 3. Generate and submit Tm coexistence runs (GRACE MLIP):
    python run_pipeline.py --only_stage 31
    sbatch slurm/submit_Tm_array.sh

    # 4. After Tm jobs finish, patch constants and generate diffusion inputs:
    python run_pipeline.py --only_stage 32
    python run_pipeline.py --only_stage 3
    python run_pipeline.py --only_stage 4
    # sbatch slurm/submit_diffusion.sh

    # 5. After diffusion jobs finish, run all analysis:
    python run_pipeline.py --from_stage 5

Stage map
---------
=====  ==========================================================
0      Validate environment (packages, LAMMPS binary, potential)
1      Generate compositions     -> results/compositions.csv
2      Compute constants         -> results/constants_all.csv
31     Generate Tm inputs        -> runs/*/tm_coexistence/
       [manual: sbatch slurm/submit_Tm_array.sh and wait]
32     Patch Tm into constants   -> results/constants_all.csv
3      Generate diffusion inputs -> runs/*/sim_*/bcc_vac_adv.in
4      Write slurm/submit_diffusion.sh
       [manual: sbatch slurm/submit_diffusion.sh and wait]
5      Parse MSD
6      Compute Dv (vacancy diffusion)
7      Compute Cv (vacancy concentration)
8      Compute D* (tracer diffusion)
9      Compute SRO
10     Post-process (polynomial fit, plots)
=====  ==========================================================

CLI examples
------------
::

    python run_pipeline.py                          # full run from stage 0
    python run_pipeline.py --from_stage 5           # analysis only
    python run_pipeline.py --only_stage 31          # Tm input generation only
    python run_pipeline.py --only_stage 32          # Tm patch only
    python run_pipeline.py --from_stage 3 --to_stage 4
    python run_pipeline.py --test --from_stage 0 --to_stage 2
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import config
from config import PIPELINE_DIR, SLURM_PARTITION
from logging_config import configure_logging, get_logger
from reporting import log_dataframe

logger = get_logger(__name__)

# Stage ordering.  31 = Stage 3b (generate Tm inputs), 32 = Stage 3c (patch Tm
# into constants_all.csv); both sit between 2 and 3 in execution order.
_STAGE_ORDER: list[int] = [0, 1, 2, 31, 32, 3, 4, 5, 6, 7, 8, 9, 10]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _require(path: Path, name: str) -> pd.DataFrame:
    """Load a required CSV or exit with a helpful message.

    Parameters
    ----------
    path : pathlib.Path
        CSV file to load.
    name : str
        Human-readable name used in the error message.

    Returns
    -------
    pandas.DataFrame
        The loaded table.
    """
    if not path.exists():
        logger.error("%s not found at %s", name, path)
        logger.error("Run the stage that produces it first.")
        raise SystemExit(1)
    return pd.read_csv(path)


def _analysis_dirs(comp_id: str) -> tuple[Path, Path, Path]:
    """Return ``(sim_base, txt_dir, plot_dir)`` for a composition.

    Parameters
    ----------
    comp_id : str
        Composition identifier.

    Returns
    -------
    tuple of pathlib.Path
        Simulation base directory, text output directory, and plot directory.
    """
    txt_dir = config.RESULTS_DIR / comp_id / "txt"
    plot_dir = config.RESULTS_DIR / comp_id / "plots"
    txt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return config.RUNS_DIR / comp_id, txt_dir, plot_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Stage implementations
# ══════════════════════════════════════════════════════════════════════════════

def _stage_0() -> None:
    """Validate the environment: packages, potential file, LAMMPS binary."""
    logger.info("Stage 0: environment validation")
    ok = True

    for pkg in ["numpy", "pandas", "matplotlib", "scipy", "sklearn"]:
        try:
            importlib.import_module(pkg)
            logger.info("  %s: available", pkg)
        except ImportError:
            logger.error("  %s: missing (pip install %s)", pkg, pkg)
            ok = False

    if config.POTENTIAL_FILE.exists():
        logger.info("  ADP potential: %s", config.POTENTIAL_FILE.name)
    else:
        logger.error("  potential not found: %s", config.POTENTIAL_FILE)
        ok = False

    match = re.search(r"(lmp[\w_-]*)", config.LAMMPS_SERIAL_CMD)
    lmp = match.group(1) if match else config.LAMMPS_SERIAL_CMD.split()[0]
    if shutil.which(lmp):
        logger.info("  LAMMPS binary: %s", lmp)
    else:
        logger.warning("  '%s' not in PATH (expected on compute nodes)", lmp)

    logger.info("  Mode: %s", "TEST" if config.TEST_MODE else "PRODUCTION")
    if not ok:
        raise SystemExit(1)


def _stage_1() -> pd.DataFrame:
    """Generate compositions (Stage 1)."""
    logger.info("Stage 1: generating compositions")
    from pipeline.compositions import generate_compositions

    df = generate_compositions()
    log_dataframe(logger, df.head(10), title="First compositions:")
    logger.info("Stage 1 complete - %d compositions", len(df))
    return df


def _stage_2(comps_df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical constants (Stage 2)."""
    logger.info("Stage 2: computing physical constants (a0, Tm_rom, Ef, Sf, C0)")
    from pipeline.constants import compute_all_constants

    df = compute_all_constants(comps_df)
    logger.info("Stage 2 complete")
    return df


def _stage_31(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> list[dict]:
    """Generate Tm coexistence inputs and the SLURM array script (Stage 3b)."""
    logger.info("Stage 3b: generating Tm coexistence inputs (GRACE MLIP)")
    from pipeline.lammps_tm import stage_3b

    job_list = stage_3b(comps_df, consts_df)
    logger.info("Stage 3b complete")
    return job_list


def _stage_32() -> None:
    """Patch the real Tm and temperature grid into constants (Stage 3c)."""
    logger.info("Stage 3c: patching constants_all.csv with the real Tm")
    from pipeline.lammps_tm import stage_3c

    stage_3c()
    logger.info("Stage 3c complete")


def _stage_3(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> list[dict]:
    """Generate ADP diffusion inputs (Stage 3)."""
    logger.info("Stage 3: generating ADP diffusion inputs (bcc_vac_adv.in)")
    from pipeline.lammps_diffusion import create_run_directories

    job_list = create_run_directories(comps_df, consts_df)
    logger.info("Stage 3 complete - %d run directories", len(job_list))
    return job_list


def _stage_4(job_list: list[dict]) -> None:
    """Write the diffusion submission script (Stage 4)."""
    logger.info("Stage 4: writing slurm/submit_diffusion.sh")
    slurm_dir = PIPELINE_DIR / "slurm"
    slurm_dir.mkdir(exist_ok=True)
    script = slurm_dir / "submit_diffusion.sh"
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=diffusion_all",
        f"#SBATCH --partition={SLURM_PARTITION}",
        "# Auto-generated - submit from the pipeline root directory",
        "",
    ]
    for job in job_list:
        lines += [f"cd {job['path']}", "sbatch submit.sh", ""]
    script.write_text("\n".join(lines))
    script.chmod(0o755)
    logger.info("Written: %s", script)
    logger.info("Submit:  sbatch slurm/submit_diffusion.sh")
    logger.info("After all jobs finish: python run_pipeline.py --from_stage 5")
    logger.info("Stage 4 complete")


def _stage_5(comps_df: pd.DataFrame) -> None:
    """Parse MSD from LAMMPS logs (Stage 5)."""
    logger.info("Stage 5: parsing MSD from LAMMPS logs")
    from analysis.msd import run_all

    for _, crow in comps_df.iterrows():
        cid = str(crow["comp_id"])
        sim_base, _, _ = _analysis_dirs(cid)
        res_base = config.RESULTS_DIR / cid / "sim_x"
        logger.info("  %s", cid)
        run_all(sim_base, res_base)
    logger.info("Stage 5 complete")


def _stage_6(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> None:
    """Compute the vacancy diffusion coefficient (Stage 6)."""
    logger.info("Stage 6: computing Dv (vacancy diffusion)")
    from analysis.vacancy_diffusion import process

    for _, crow in comps_df.iterrows():
        cid = str(crow["comp_id"])
        sim_base, txt_dir, plot_dir = _analysis_dirs(cid)
        res_base = config.RESULTS_DIR / cid / "sim_x"
        cst = consts_df[consts_df["comp_id"] == cid]
        tm_over = (
            {e: float(cst.iloc[0]["Tm"]) for e in config.ELEMENTS}
            if not cst.empty else None
        )
        logger.info("  %s", cid)
        process(sim_base, res_base, txt_dir, plot_dir, tm_over)
    logger.info("Stage 6 complete")


def _stage_7(comps_df: pd.DataFrame, consts_df: pd.DataFrame) -> None:
    """Compute the equilibrium vacancy concentration (Stage 7)."""
    logger.info("Stage 7: computing Cv (vacancy concentration)")
    from analysis.vacancy_concentration import run

    for _, crow in comps_df.iterrows():
        cid = str(crow["comp_id"])
        cst = consts_df[consts_df["comp_id"] == cid]
        if cst.empty:
            continue
        c = cst.iloc[0]
        t_grid = (
            json.loads(c["T_grid"]) if isinstance(c["T_grid"], str)
            else list(c["T_grid"])
        )
        _, txt_dir, _ = _analysis_dirs(cid)
        logger.info("  %s", cid)
        run(float(c["Ef"]), float(c["C0"]), t_grid, txt_dir, comp_label=cid)
    logger.info("Stage 7 complete")


def _stage_8(comps_df: pd.DataFrame) -> None:
    """Compute the tracer self-diffusion coefficient (Stage 8)."""
    logger.info("Stage 8: computing D* (tracer self-diffusion)")
    from analysis.logparse import LogParseError
    from analysis.tracer_diffusion import run

    for _, crow in comps_df.iterrows():
        cid = str(crow["comp_id"])
        comp = {e: float(crow[f"x_{e}"]) for e in config.ELEMENTS}
        _, txt_dir, plot_dir = _analysis_dirs(cid)
        logger.info("  %s", cid)
        try:
            run(comp, txt_dir, plot_dir, comp_label=cid)
        except LogParseError as exc:
            logger.warning("D2 skipped for %s: %s", cid, exc)
    logger.info("Stage 8 complete")


def _stage_9(comps_df: pd.DataFrame) -> None:
    """Compute Warren-Cowley SRO parameters (Stage 9)."""
    logger.info("Stage 9: computing SRO (Warren-Cowley parameters)")
    from analysis.sro import process

    for _, crow in comps_df.iterrows():
        cid = str(crow["comp_id"])
        sim_base, txt_dir, plot_dir = _analysis_dirs(cid)
        logger.info("  %s", cid)
        process(sim_base, txt_dir, plot_dir, comp_label=cid)
    logger.info("Stage 9 complete")


def _stage_10() -> None:
    """Run post-processing and plots (Stage 10)."""
    logger.info("Stage 10: post-processing and plots")
    from analysis.postprocess import run_postprocess

    run_postprocess()
    logger.info("Stage 10 complete")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parsing and dispatch
# ══════════════════════════════════════════════════════════════════════════════

def _parse_run_stages(args: argparse.Namespace) -> list[int]:
    """Resolve the list of stages to run from the CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list of int
        Stages to execute, in order.
    """
    if args.only_stage is not None:
        return [args.only_stage]
    index = {s: i for i, s in enumerate(_STAGE_ORDER)}
    start = index.get(args.from_stage)
    end = index.get(args.to_stage)
    if start is None:
        logger.error("Unknown --from_stage %s", args.from_stage)
        raise SystemExit(1)
    if end is None:
        logger.error("Unknown --to_stage %s", args.to_stage)
        raise SystemExit(1)
    return _STAGE_ORDER[start: end + 1]


def main() -> None:
    """Command-line entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="WMoNbZrTiTa diffusion pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--from_stage", type=int, default=0, metavar="N",
                        help="Start from stage N (default 0). "
                             "Use 31 for Stage 3b, 32 for Stage 3c.")
    parser.add_argument("--to_stage", type=int, default=10, metavar="N",
                        help="Stop after stage N (default 10).")
    parser.add_argument("--only_stage", type=int, default=None, metavar="N",
                        help="Run exactly one stage.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true",
                      help="Force the fast validation configuration.")
    mode.add_argument("--production", action="store_true",
                      help="Force the full production configuration.")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (default INFO).")
    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.test:
        config.set_test_mode(True)
    elif args.production:
        config.set_test_mode(False)

    run_stages = _parse_run_stages(args)
    logger.info("Mode: %s", "TEST" if config.TEST_MODE else "PRODUCTION")
    logger.info("Running stages: %s", run_stages)

    comp_csv = config.RESULTS_DIR / "compositions.csv"
    const_csv = config.RESULTS_DIR / "constants_all.csv"
    job_csv = config.RESULTS_DIR / "job_list.csv"

    comps_df: pd.DataFrame | None = None
    consts_df: pd.DataFrame | None = None
    job_list: list[dict] | None = None
    t0 = time.time()

    for stage in run_stages:
        if stage == 0:
            _stage_0()
        elif stage == 1:
            comps_df = _stage_1()
        elif stage == 2:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            consts_df = _stage_2(comps_df)
        elif stage == 31:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            if consts_df is None:
                consts_df = _require(const_csv, "constants_all.csv")
            _stage_31(comps_df, consts_df)
        elif stage == 32:
            _stage_32()
            consts_df = None      # force reload of the patched CSV
        elif stage == 3:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            if consts_df is None:
                consts_df = _require(const_csv, "constants_all.csv")
            job_list = _stage_3(comps_df, consts_df)
        elif stage == 4:
            if job_list is None:
                job_list = _require(job_csv, "job_list.csv").to_dict("records")
            _stage_4(job_list)
        elif stage == 5:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            _stage_5(comps_df)
        elif stage == 6:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            if consts_df is None:
                consts_df = _require(const_csv, "constants_all.csv")
            _stage_6(comps_df, consts_df)
        elif stage == 7:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            if consts_df is None:
                consts_df = _require(const_csv, "constants_all.csv")
            _stage_7(comps_df, consts_df)
        elif stage == 8:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            _stage_8(comps_df)
        elif stage == 9:
            if comps_df is None:
                comps_df = _require(comp_csv, "compositions.csv")
            _stage_9(comps_df)
        elif stage == 10:
            _stage_10()
        else:
            logger.warning("Unknown stage %s - skipped", stage)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60.0)


if __name__ == "__main__":
    main()
