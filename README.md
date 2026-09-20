# WMoNbZrTiTa Diffusion Pipeline

High-throughput molecular-dynamics pipeline for computing vacancy-mediated
tracer diffusivities, short-range order, and melting temperatures across
W-Mo-Nb-Zr-Ti-Ta refractory high-entropy alloy compositions.

---

## Overview

For each alloy composition sampled from the six-element composition space, the
pipeline automates the following calculations:

1. **Pure-element baselines** - ADP-relaxed lattice parameters and vacancy
   formation entropies via phonopy.
2. **Alloy constants** - vacancy formation energy, rule-of-mixtures melting
   temperature, and the temperature grid.
3. **True melting temperature** - phase coexistence via the Modified Z-method
   using the GRACE-2L-OMAT machine-learning potential.
4. **Diffusion and SRO** - Monte Carlo chemical equilibration followed by long
   MD diffusion with a single vacancy, using the fast ADP potential.
5. **Tracer diffusivity** - per-element diffusion rates via the Einstein
   relation and Arrhenius fits.
6. **Chemical ordering** - Warren-Cowley short-range order parameters.
7. **Global predictive model** - a Ridge-regularised composition-property
   polynomial fitted across all compositions.

---

## Architecture

The codebase is split into two strictly separated layers:

| Layer | Directory | Responsibility |
| --- | --- | --- |
| Generation | `pipeline/` | Build LAMMPS and phonopy inputs (Stages 1-4) |
| Post-processing | `analysis/` | Parse outputs and compute physics (Stages 5-10) |

Supporting modules:

| Module | Responsibility |
| --- | --- |
| `config.py` | Single source of truth for constants, paths, and mode |
| `run_pipeline.py` | Orchestrator and unified CLI entry point |
| `logging_config.py` | Central logging configuration |
| `reporting.py` | Aligned plain-text table rendering |

`pipeline/` never reads simulation output, and `analysis/` never writes LAMMPS
input. The only shared state is the file system, described below.

---

## Data flow

```
config.py  (constants, paths, TEST_MODE)
    |
    v
[Stage 1]  pipeline/compositions.py
    |         -> results/compositions.csv
    v
[Stage 2]  pipeline/constants.py
    |         -> results/constants_all.csv
    |         -> results/constants/<comp_id>.json
    v
[Stage 3b] pipeline/lammps_tm.py
    |         -> runs/<comp_id>/tm_coexistence/T_<guess>/coexistence.in
    |         -> slurm/submit_Tm_array.sh
    |   (sbatch; GRACE MLIP)
    v
[Stage 3c] pipeline/lammps_tm.py
    |         -> results/constants_all.csv   (real Tm, rebuilt T_grid)
    v
[Stage 3]  pipeline/lammps_diffusion.py
    |         -> runs/<comp_id>/sim_<T>/bcc_vac_adv.in
    |         -> results/job_list.csv
    v
[Stage 4]  run_pipeline.py
    |         -> slurm/submit_diffusion.sh
    |   (sbatch; ADP potential)
    v
[Stage 5]  analysis/msd.py
    |         -> results/<comp_id>/sim_x/<T>/msd_all.txt
    |         -> results/<comp_id>/sim_x/<T>/msd_solo.txt
    v
[Stage 6]  analysis/vacancy_diffusion.py
    |         -> results/<comp_id>/txt/Dv.txt
    v
[Stage 7]  analysis/vacancy_concentration.py
    |         -> results/<comp_id>/txt/Cv.txt
    v
[Stage 8]  analysis/tracer_diffusion.py
    |         -> results/<comp_id>/txt/D2_components.txt
    v
[Stage 9]  analysis/sro.py
    |         -> results/<comp_id>/txt/sro_vs_temp.csv
    v
[Stage 10] analysis/postprocess.py
              -> results/master_results.csv
              -> results/poly_fit_coeffs.csv
              -> results/plots/*.png
```

---

## Repository layout

```
config.py                  Central configuration - edit before running
run_pipeline.py            Orchestrator and unified CLI entry point
logging_config.py          Central logging configuration
reporting.py               Aligned plain-text table rendering
pyproject.toml             Packaging metadata and pinned dependencies
requirements.txt           Locked dependency list
Makefile                   Workflow shortcuts

pipeline/                  LAMMPS and phonopy input generation
    lammps_common.py       Shared composition algebra and LAMMPS blocks
    compositions.py        Stage 1  - composition sampling
    constants.py           Stage 2  - a0, Tm_rom, Ef, Sf, C0, T_grid
    lammps_diffusion.py    Stage 3  - ADP diffusion inputs
    lammps_tm.py           Stage 3b/3c - GRACE coexistence inputs, Tm patch
    compute_sf_phonopy.py  Standalone phonopy Sf calculation

analysis/                  Post-processing and analytics
    logparse.py            Shared LAMMPS text-parsing helpers
    arrhenius.py           Weighted Arrhenius fitting
    msd.py                 Stage 5  - mean-square displacement
    vacancy_diffusion.py   Stage 6  - Dv(T)
    vacancy_concentration.py Stage 7 - Cv(T)
    tracer_diffusion.py    Stage 8  - D*(T)
    sro.py                 Stage 9  - Warren-Cowley parameters
    postprocess.py         Stage 10 - aggregation, fit, plots

docs/                      Physics, logic, workflow, and testing guides
slurm/                     Cluster submission scripts (generated + static)
tests/                     pytest suite
```

---

## Installation

```bash
git clone https://github.com/akmal523/high-entropy-alloy-pipeline.git
cd high-entropy-alloy-pipeline

# Editable install with pinned dependencies and the test runner
make setup
```

`make setup` runs `pip install -e ".[dev]"`. To install only the runtime
dependencies, use `pip install -r requirements.txt`.

Before running, edit `config.py` to set `LAMMPS_EXE`, `LAMMPS_CMD_TMPL`,
`SLURM_PARTITION`, and `GRACE_MODEL_DIR`, and place
`WMoNbZrTiTa.nist.adp.txt` in the project root.

---

## Quick start

The `Makefile` wraps the common workflow:

```bash
make generate            # Stages 0-2: validate, compositions, constants
make tm-inputs           # Stage 3b: write GRACE coexistence inputs
make submit-tm           # sbatch slurm/submit_Tm_array.sh
# [wait for Tm jobs]
make tm-patch            # Stage 3c: patch real Tm into constants
make diffusion-inputs    # Stages 3-4: write ADP inputs and submit script
make submit              # sbatch slurm/submit_diffusion.sh
# [wait for diffusion jobs]
make analyze             # Stages 5-10: MSD, Dv, Cv, D*, SRO, postprocess
```

The equivalent direct commands are:

```bash
python run_pipeline.py --from_stage 0 --to_stage 2
python run_pipeline.py --only_stage 31
sbatch slurm/submit_Tm_array.sh
python run_pipeline.py --only_stage 32
python run_pipeline.py --from_stage 3 --to_stage 4
sbatch slurm/submit_diffusion.sh
python run_pipeline.py --from_stage 5
```

A fast validation run uses the reduced configuration:

```bash
make generate TEST=1
# or
python run_pipeline.py --test --from_stage 0 --to_stage 2
```

---

## Configuration

All tuneable parameters live in `config.py`.

| Parameter | Default | Description |
| --- | --- | --- |
| `TEST_MODE` | `False` | Full production run; override with `--test` |
| `N_COMPOSITIONS` | 100 | Number of alloy compositions to sample |
| `LAMMPS_EXE` | (set this) | Path to the LAMMPS binary |
| `GRACE_MODEL_DIR` | (set this) | Path to the GRACE-2L-OMAT weights |
| `SLURM_PARTITION` | `compute` | Cluster partition name |
| `T_FRAC_MAX` | 0.80 | Upper temperature bound as a fraction of Tm |

`TEST_MODE` defaults to `False`. It can be overridden at runtime, before any
stage executes, with the `--test` flag (or forced back with `--production`).
The mode switch changes only sampling counts, run lengths, and walltimes;
physical constants are never modified.

---

## Stage reference

| Stage | Module | Output |
| --- | --- | --- |
| 0 | `run_pipeline.py` | Environment validation |
| 1 | `pipeline/compositions.py` | `results/compositions.csv` |
| 2 | `pipeline/constants.py` | `results/constants_all.csv` |
| 3b | `pipeline/lammps_tm.py` | `runs/*/tm_coexistence/`, `slurm/submit_Tm_array.sh` |
| 3c | `pipeline/lammps_tm.py` | Patched `results/constants_all.csv` |
| 3 | `pipeline/lammps_diffusion.py` | `runs/*/sim_*/bcc_vac_adv.in` |
| 4 | `run_pipeline.py` | `slurm/submit_diffusion.sh` |
| 5 | `analysis/msd.py` | `results/*/sim_x/*/msd_*.txt` |
| 6 | `analysis/vacancy_diffusion.py` | `results/*/txt/Dv.txt` |
| 7 | `analysis/vacancy_concentration.py` | `results/*/txt/Cv.txt` |
| 8 | `analysis/tracer_diffusion.py` | `results/*/txt/D2_components.txt` |
| 9 | `analysis/sro.py` | `results/*/txt/sro_vs_temp.csv` |
| 10 | `analysis/postprocess.py` | `results/master_results.csv`, plots |

---

## Output file structure

```
results/
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
            Dv.txt
            Cv.txt
            D2_components.txt
            sro_vs_temp.csv
        plots/
            Dv_total.png
            Dv_elements.png
            Dv_homologous.png
            cv_vs_invT_<comp_id>.png
            D2_vs_invT.png
            Dtotal_vs_invT.png
            sro_vs_temp.png
        sim_x/
            <T>/
                msd_all.txt
                msd_solo.txt
    plots/
        comparison_D_vs_Tm.png
        comparison_sro_vs_T.png
        parity_Dtotal.png
        arrhenius_summary.png

runs/
    <comp_id>/
        sim_<T>/
            bcc_vac_adv.in
            submit.sh
            dump/
        tm_coexistence/
            T_<guess>/
                coexistence.in
                dump/
```

---

## Physics notes

The full derivations, with citations, are in [`docs/physics.md`](docs/physics.md)
and [`docs/logic.md`](docs/logic.md). Key points:

* **Sequential-fraction composition assignment.** LAMMPS
  `set type/fraction` is applied sequentially and overwrites earlier types.
  The conditional fractions are obtained by inverting
  `P(X) = f_X * prod_{Y after X} (1 - f_Y)` in reverse setting order. See
  [`pipeline/lammps_common.py`](pipeline/lammps_common.py).
* **Modified Z-method.** An elongated BCC supercell is split into solid and
  liquid halves; coexistence is detected from the volume and pressure
  trajectory under independent per-axis NPT barostats, followed by stress
  equalization. See [`pipeline/lammps_tm.py`](pipeline/lammps_tm.py).
* **Dynamic SRO cutoff.** The cutoff is the midpoint between the BCC first and
  second neighbour shells, `r_sro = a0 * (sqrt(3)/2 + 1) / 2`, evaluated at
  the composition's lattice parameter. See [`config.py`](config.py).
* **Tracer versus vacancy diffusion.** `Dv` is the vacancy diffusivity;
  `D*_i = f_i * Cv * Dv_i / x_i` is the tracer diffusivity. The BCC
  correlation factor `f = 0.727` is already embedded in the MSD-derived data
  and must not be applied twice. See
  [`analysis/tracer_diffusion.py`](analysis/tracer_diffusion.py).

Key references:

* Starikov et al. *Phys. Rev. Materials* **8**, 043603 (2024) - Ef, Sf, ADP
  validation.
* Karavaev et al. *J. Chem. Phys.* **144**, 194507 (2016) - modified Z-method.
* Cowley, *Phys. Rev.* **77**, 669 (1950) - Warren-Cowley SRO.
* Compaan & Haven, *Trans. Faraday Soc.* **52**, 786 (1956) - BCC correlation
  factor.

---

## Testing

```bash
make test        # python -m pytest
make lint        # byte-compile every module
```

The suite covers the sequential-fraction algebra, weighted Arrhenius fitting,
graceful MSD log parsing, the mode switch, and composition sampling.

---

## Requirements

* Python >= 3.10
* LAMMPS with the `adp` pair style (diffusion runs)
* LAMMPS with the `grace` pair style (Tm runs)
* Phonopy (vacancy formation entropy)
* GRACE-2L-OMAT model weights
* `WMoNbZrTiTa.nist.adp.txt` potential file (project root)
* SLURM cluster environment

Runtime dependencies are pinned in [`pyproject.toml`](pyproject.toml) and
mirrored in [`requirements.txt`](requirements.txt).
