# WMoNbZrTiTa Diffusion Pipeline

High-throughput MD pipeline for computing vacancy-mediated tracer diffusivities,
short-range order, and melting temperatures across 100 W-Mo-Nb-Zr-Ti-Ta
refractory high-entropy alloy compositions.

---

## What this does

For each of 100 alloy compositions uniformly sampled from the 6-element composition space, this pipeline automates the following physics calculations:

1. **Pure Element Baselines:** Computes exact ADP-relaxed lattice parameters ($a_0$) and vacancy formation entropies ($S_f$) via Phonopy.
2. **Alloy Constants:** Computes alloy formation energies ($E_f$) and preliminary Rule-of-Mixtures constants.
3. **True Melting Temperature ($T_m$):** Determines phase coexistence via the Modified Z-method using the high-fidelity **GRACE-2L-OMAT MLIP**.
4. **Diffusion & SRO:** Runs Monte Carlo (MC) chemical equilibration followed by long **MD diffusion** with a single vacancy using the fast **ADP potential** (432-atom BCC supercell).
5. **Tracer Diffusivity ($D^*_i$):** Extracts per-element diffusion rates via the Einstein relation and fits Arrhenius parameters ($D_{0,i}$, $Q_i$).
6. **Chemical Ordering:** Computes Warren-Cowley Short-Range Order (SRO) parameters $\alpha_{ij}(T)$.
7. **Global Predictive Model:** Fits a Ridge-regularized composition-property polynomial across all 100 alloys to predict diffusion across the entire hyperspace.

---

## Quick start

```bash
git clone [https://github.com/akmal523/high-entropy-alloy-pipeline.git](https://github.com/akmal523/high-entropy-alloy-pipeline.git)
cd WMoNbZrTiTa
pip install -r requirements.txt

# Edit config.py — set LAMMPS_EXE, SLURM_PARTITION, GRACE_MODEL_DIR, etc.

python run_pipeline.py --only_stage 0    # validate environment
python run_pipeline.py --only_stage 1    # generate 100 compositions

# Compute pure-element reference data (Sf, a0)
python pipeline/compute_sf_phonopy.py

# Compute alloy constants (Ef, Tm_rom)
python run_pipeline.py --only_stage 2

# Generate and submit Tm jobs (GRACE MLIP)
python run_pipeline.py --only_stage 31
sbatch slurm/submit_Tm_array.sh          # submit to cluster

# [wait for Tm jobs to finish]
python run_pipeline.py --only_stage 32   # patch real Tm into constants

# Generate and submit diffusion jobs (ADP)
python run_pipeline.py --only_stage 3
python run_pipeline.py --only_stage 4
sbatch slurm/submit_diffusion.sh         # submit to cluster

# [wait for diffusion jobs to finish]
python run_pipeline.py --from_stage 5    # parse MSD, extract D*, SRO, and plot

```

See [`docs/testing.md`](https://www.google.com/search?q=docs/testing.md) for a fast cluster validation guide using `TEST_MODE`.

---

## Requirements

* Python ≥ 3.10
* LAMMPS with ADP pair style (`pair_style adp`)
* LAMMPS with GRACE pair style (for Tm runs)
* Phonopy (for vibrational entropy calculations)
* GRACE-2L-OMAT model weights
* `WMoNbZrTiTa.nist.adp.txt` potential file (place in project root)
* SLURM cluster environment

---

## Repository layout

```
config.py              Central configuration — edit before running
run_pipeline.py        Orchestrator — single entry point
pipeline/              LAMMPS & Phonopy input generation (Stages 1–4)
analysis/              Post-processing & Analytics (Stages 5–10)
docs/                  Physics logic, testing guides, and derivations
slurm/                 Cluster submission scripts (generated + static)

```

---

## Physics notes

All pipeline architecture, math, and stage-by-stage workflows are detailed in
[`docs/logic.md`](https://www.google.com/search?q=docs/logic.md).

Key references:

* Starikov et al. *Phys. Rev. Materials* **8**, 043603 (2024) — Ef, Sf, ADP validation
* Karavaev et al. *J. Chem. Phys.* **144**, 194507 (2016) — modified Z-method for Tm
* Zhu et al. *npj Comput. Mater.* **10**, 60 (2024) — MLIP for Tm calculation
* Cowley, *Phys. Rev.* **77**, 669 (1950) — Warren-Cowley SRO parameters

---

## Configuration

All tuneable parameters live in `config.py`. Key settings:

| Parameter | Default | Description |
| --- | --- | --- |
| `TEST_MODE` | `False` | Switch to `True` for a 5-composition quick validation |
| `N_COMPOSITIONS` | 100 | Number of alloy compositions to sample |
| `LAMMPS_EXE` | (set this) | Path to LAMMPS binary |
| `GRACE_MODEL_DIR` | (set this) | Path to GRACE-2L-OMAT weights |
| `SLURM_PARTITION` | `compute` | Cluster partition name |
| `T_FRAC_MAX` | 0.80 | Upper T bound as fraction of Tm (ADP safety margin) |

---

## Citation

If you use this pipeline, please cite:

> Starikov et al. Phys. Rev. Materials **8**, 043603 (2024)
> Karavaev et al. J. Chem. Phys. **144**, 194507 (2016)
> Zhu et al. npj Comput. Mater. **10**, 60 (2024)



