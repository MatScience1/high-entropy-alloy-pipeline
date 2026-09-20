# Pipeline Workflow

## Stage diagram

```
                        +---------------------------------------------+
                        |            config.py  (edit first)          |
                        +------------------+--------------------------+
                                           |
          +--------------------------------v----------------------------------+
          |  Stage 0   Validate environment                                   |
          |  Stage 1   Generate compositions  -> results/compositions.csv     |
          |  Stage 2   Compute Ef, Sf, a0, Tm_rom -> results/constants_all.csv|
          +--------------------------------+----------------------------------+
                                           |
          +--------------------------------v----------------------------------+
          |  Stage 3b  Write coexistence.in per composition (GRACE MLIP)      |
          |            Write slurm/submit_Tm_array.sh                         |
          +--------------------------------+----------------------------------+
                                           |
                         [sbatch slurm/submit_Tm_array.sh]
                         [wait - N_comps x N_TGUESS values]
                                           |
          +--------------------------------v----------------------------------+
          |  Stage 3c  Read log.coexistence -> patch real Tm + T_grid         |
          |            -> results/constants_all.csv (Tm column updated)       |
          +--------------------------------+----------------------------------+
                                           |
          +--------------------------------v----------------------------------+
          |  Stage 3   Write bcc_vac_adv.in per (composition, temperature)    |
          |            (ADP potential, 432 atoms, MC+NPT+MD)                  |
          |  Stage 4   Write slurm/submit_diffusion.sh                        |
          +--------------------------------+----------------------------------+
                                           |
                         [sbatch slurm/submit_diffusion.sh]
                         [wait - N_comps x N_TEMPS points]
                                           |
          +--------------------------------v----------------------------------+
          |  Stage 5   Parse MSD from LAMMPS logs                             |
          |  Stage 6   Dv(T) - vacancy diffusion coefficient                  |
          |  Stage 7   Cv(T) - equilibrium vacancy concentration              |
          |  Stage 8   D*(T) - tracer self-diffusion per element              |
          |  Stage 9   SRO   - Warren-Cowley parameters alpha_ij              |
          |  Stage 10  Polynomial fit + comparison plots                      |
          +-------------------------------------------------------------------+
```

---

## Command reference

### First-time setup

```bash
# 1. Clone and install pinned dependencies
git clone https://github.com/akmal523/high-entropy-alloy-pipeline.git
cd high-entropy-alloy-pipeline
make setup

# 2. Edit config.py:
#    - LAMMPS_EXE, LAMMPS_CMD_TMPL (paths to your LAMMPS build)
#    - SLURM_PARTITION, SLURM_NODES, SLURM_TASKS_PER_NODE
#    - GRACE_MODEL_DIR (path to GRACE-2L-OMAT)

# 3. Place the ADP potential file in the project root:
#    WMoNbZrTiTa.nist.adp.txt

# 4. Validate
python run_pipeline.py --only_stage 0
```

### Full production run

```bash
# Stages 0-2: composition generation + constants
make generate

# Stage 3b: generate GRACE coexistence inputs
make tm-inputs
make submit-tm

# [wait for Tm jobs to finish]
sacct -j <JOBID> --format=State

# Stage 3c: patch real Tm into constants
make tm-patch

# Stages 3-4: generate ADP diffusion inputs
make diffusion-inputs
make submit

# [wait for diffusion jobs to finish]

# Stages 5-10: analysis
make analyze
```

### Resuming from a checkpoint

```bash
# After Tm jobs finished:
python run_pipeline.py --only_stage 32

# After diffusion jobs finished:
python run_pipeline.py --from_stage 5

# Re-run a single analysis stage:
python run_pipeline.py --only_stage 9    # SRO only
```

### TEST_MODE

`TEST_MODE` defaults to `False`. Enable the fast validation configuration
either by passing `--test` on the command line or by setting `TEST_MODE = True`
in `config.py`:

```bash
python run_pipeline.py --test --from_stage 0 --to_stage 2
make generate TEST=1
```

| Parameter | TEST | PRODUCTION |
| --- | --- | --- |
| N_COMPOSITIONS | 5 | 100 |
| N_TEMPS | 3 | 10 |
| N_MC | 100 000 | 2 000 000 |
| N_MD | 5 000 000 | 500 000 000 |
| COEX_N_TGUESS | 3 | 5 |
| SLURM_WALLTIME | 2 h | 12 h |
| SLURM_WALLTIME_TM | 4 h | 48 h |

---

## Directory layout (at runtime)

```
high-entropy-alloy-pipeline/
    config.py
    run_pipeline.py
    logging_config.py
    reporting.py
    pyproject.toml
    requirements.txt
    Makefile

    pipeline/                   LAMMPS and phonopy input generation
        lammps_common.py        Shared composition algebra and LAMMPS blocks
        compositions.py         Stage 1
        constants.py            Stage 2  (Ef, Tm_rom, a0, Sf, C0)
        lammps_diffusion.py     Stage 3  (ADP bcc_vac_adv.in)
        lammps_tm.py            Stage 3b/3c  (GRACE coexistence.in)
        compute_sf_phonopy.py   Standalone Sf calculation

    analysis/                   Post-processing
        logparse.py             Shared LAMMPS text-parsing helpers
        arrhenius.py            Weighted Arrhenius fitting
        msd.py                  Stage 5
        vacancy_diffusion.py    Stage 6
        vacancy_concentration.py Stage 7
        tracer_diffusion.py     Stage 8
        sro.py                  Stage 9
        postprocess.py          Stage 10

    docs/
        physics.md              All equations with citations
        logic.md                Stage-by-stage logic
        workflow.md             This file
        testing.md              Cluster validation guide

    slurm/                      Cluster submission scripts
        submit_Tm_array.sh      Generated by Stage 3b
        submit_diffusion.sh     Generated by Stage 4

    results/                    Runtime output (gitignored)
        compositions.csv
        constants_all.csv
        tm_job_list.csv
        job_list.csv
        master_results.csv
        poly_fit_coeffs.csv
        constants/
            <comp_id>.json
        <comp_id>/
            txt/
            plots/
            sim_x/

    runs/                       LAMMPS run directories (gitignored)
        <comp_id>/
            sim_<T>/
                bcc_vac_adv.in
                submit.sh
                dump/
            tm_coexistence/
                T_<guess>/
                    coexistence.in
                    dump/
