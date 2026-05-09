# Pipeline Workflow

## Stage diagram

```
                        ┌─────────────────────────────────────────────┐
                        │            config.py  (edit first)          │
                        └──────────────────┬──────────────────────────┘
                                           │
          ┌────────────────────────────────▼──────────────────────────────────┐
          │  Stage 0   Validate environment                                   │
          │  Stage 1   Generate 100 compositions  → results/compositions.csv  │
          │  Stage 2   Compute Ef, Sf, a0, Tm_rom → results/constants_all.csv │
          └────────────────────────────────┬──────────────────────────────────┘
                                           │
          ┌────────────────────────────────▼──────────────────────────────────┐
          │  Stage 3b  Write coexistence.in per composition (GRACE MLIP)      │
          │            Write slurm/submit_Tm_array.sh                         │
          └────────────────────────────────┬──────────────────────────────────┘
                                           │
                         [sbatch slurm/submit_Tm_array.sh]
                         [wait — 100 comps × 5 T_guess values]
                                           │
          ┌────────────────────────────────▼──────────────────────────────────┐
          │  Stage 3c  Read log.coexistence → patch real Tm + T_grid          │
          │            → results/constants_all.csv (Tm column updated)        │
          └────────────────────────────────┬──────────────────────────────────┘
                                           │
          ┌────────────────────────────────▼──────────────────────────────────┐
          │  Stage 3   Write bcc_vac_adv.in per (composition, temperature)    │
          │            (ADP potential, 432 atoms, MC+NPT+MD)                  │
          │  Stage 4   Write slurm/submit_diffusion.sh                        │
          └────────────────────────────────┬──────────────────────────────────┘
                                           │
                         [sbatch slurm/submit_diffusion.sh]
                         [wait — 100 comps × 10 T points × ~12 h each]
                                           │
          ┌────────────────────────────────▼──────────────────────────────────┐
          │  Stage 5   Parse MSD from LAMMPS logs                             │
          │  Stage 6   Dv(T) — vacancy diffusion coefficient                  │
          │  Stage 7   Cv(T) — equilibrium vacancy concentration              │
          │  Stage 8   D*(T) — tracer self-diffusion per element              │
          │  Stage 9   SRO   — Warren-Cowley parameters αᵢⱼ                  │
          │  Stage 10  Polynomial fit + comparison plots                      │
          └───────────────────────────────────────────────────────────────────┘
```

---

## Command reference

### First-time setup

```bash
# 1. Clone and configure
git clone https://github.com/<you>/WMoNbZrTiTa.git
cd WMoNbZrTiTa
pip install -r requirements.txt

# 2. Edit config.py:
#    - Set LAMMPS_EXE, LAMMPS_CMD_TMPL (paths to your LAMMPS build)
#    - Set SLURM_PARTITION, SLURM_NODES, SLURM_TASKS_PER_NODE
#    - Set GRACE_MODEL_DIR (path to GRACE-2L-OMAT)
#    - Set TEST_MODE = True for a quick validation run

# 3. Place the ADP potential file in the project root:
#    WMoNbZrTiTa.nist.adp.txt

# 4. Validate
python run_pipeline.py --only_stage 0
```

### Full production run

```bash
# Stages 0–2: composition generation + constants
python run_pipeline.py --from_stage 0 --to_stage 2

# Stage 3b: generate GRACE coexistence inputs
python run_pipeline.py --only_stage 31
sbatch slurm/submit_Tm_array.sh

# [wait for Tm jobs to finish]
sacct -j <JOBID> --format=State

# Stage 3c: patch real Tm into constants
python run_pipeline.py --only_stage 32

# Stage 3–4: generate ADP diffusion inputs
python run_pipeline.py --from_stage 3 --to_stage 4
sbatch slurm/submit_diffusion.sh

# [wait for diffusion jobs to finish]

# Stages 5–10: analysis
python run_pipeline.py --from_stage 5
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

Set `TEST_MODE = True` in `config.py` for a rapid validation run:

| Parameter | TEST | PRODUCTION |
|-----------|------|------------|
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
WMoNbZrTiTa/
├── config.py
├── run_pipeline.py
├── requirements.txt
│
├── pipeline/                   LAMMPS input generation
│   ├── compositions.py         Stage 1
│   ├── constants.py            Stage 2  (Ef, Tm_rom, a0, Sf, C0)
│   ├── lammps_diffusion.py     Stage 3  (ADP bcc_vac_adv.in)
│   └── lammps_tm.py            Stage 3b/3c  (GRACE coexistence.in)
│
├── analysis/                   Post-processing
│   ├── msd.py                  Stage 5
│   ├── Dv.py                   Stage 6
│   ├── Cv.py                   Stage 7
│   ├── D2.py                   Stage 8
│   ├── sro.py                  Stage 9
│   └── postprocess.py          Stage 10
│
├── docs/
│   ├── physics.md              All equations with citations
│   └── workflow.md             This file
│
├── slurm/                      Cluster submission scripts
│   ├── submit_Tm_array.sh      Generated by Stage 3b
│   └── submit_diffusion.sh     Generated by Stage 4
│
├── results/                    Runtime output (gitignored)
│   ├── compositions.csv
│   ├── constants_all.csv
│   ├── tm_job_list.csv
│   ├── job_list.csv
│   ├── constants/
│   │   └── comp_<N>.json
│   └── <comp_id>/
│       ├── txt/
│       └── plots/
│
└── runs/                       LAMMPS run directories (gitignored)
    └── <comp_id>/
        ├── sim_<T>/
        │   ├── bcc_vac_adv.in
        │   ├── submit.sh
        │   └── dump/
        └── tm_coexistence/
            └── T_<guess>/
                ├── coexistence.in
                └── dump/
```
