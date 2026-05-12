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
