# WMoNbZrTiTa Diffusion Pipeline

High-throughput MD pipeline for computing vacancy-mediated tracer diffusivities,
short-range order, and melting temperatures across 100 W-Mo-Nb-Zr-Ti-Ta
refractory high-entropy alloy compositions.

---

## What this does

For each of 100 alloy compositions sampled from the 6-element composition space:

1. Computes the **melting temperature** $T_m$ via solid-liquid phase coexistence (GRACE-2L-OMAT MLIP, Karavaev modified Z-method).
2. Runs **MC + NPT equilibration** followed by long **MD diffusion** with a single vacancy (ADP potential, 432-atom BCC supercell).
3. Extracts per-element **tracer diffusivities** $D^*_i(T)$ via the Einstein relation and fits Arrhenius parameters ($D_{0,i}, Q_i$).
4. Computes **Warren-Cowley SRO parameters** $\alpha_{ij}(T)$.
5. Fits a **composition-property polynomial** across all 100 alloys.
6. Produces $D^*_i$ vs. $T/T_m$ plots—homologous temperature normalization.

---

## Quick start

```bash
git clone https://github.com/akmal523/high-entropy-alloy-pipeline.git
cd WMoNbZrTiTa
pip install -r requirements.txt

# Edit config.py — set LAMMPS_EXE, SLURM_PARTITION, GRACE_MODEL_DIR, etc.

python run_pipeline.py --only_stage 0    # validate environment
python run_pipeline.py --from_stage 0 --to_stage 2    # setup + constants

python run_pipeline.py --only_stage 31   # generate Tm inputs
sbatch slurm/submit_Tm_array.sh          # submit to cluster

# [wait for Tm jobs]
python run_pipeline.py --only_stage 32   # patch real Tm
python run_pipeline.py --from_stage 3 --to_stage 4    # diffusion inputs
sbatch slurm/submit_diffusion.sh         # submit to cluster

# [wait for diffusion jobs]
python run_pipeline.py --from_stage 5    # analysis + plots
```

See [`docs/workflow.md`](docs/workflow.md) for the full stage diagram and
`TEST_MODE` instructions.

---

## Requirements

- Python ≥ 3.10
- LAMMPS with ADP pair style (`pair_style adp`)
- LAMMPS with GRACE pair style (for Tm runs)
- GRACE-2L-OMAT model weights
- `WMoNbZrTiTa.nist.adp.txt` potential file (place in project root)
- SLURM cluster

---

## Repository layout

```
config.py              Central configuration — edit before running
run_pipeline.py        Orchestrator — single entry point
pipeline/              LAMMPS input generation (Stages 1–3)
analysis/              Post-processing (Stages 5–10)
docs/                  Physics derivations and workflow diagram
slurm/                 Cluster submission scripts (generated + static)
```

---

## Physics notes

All equations, derivations, and literature references are in
[`docs/physics.md`](docs/physics.md).

Key references:
- Starikov et al. *Phys. Rev. Materials* **8**, 043603 (2024) — Ef, Sf, ADP validation
- Karavaev et al. *J. Chem. Phys.* **144**, 194507 (2016) — modified Z-method for Tm
- Zhu et al. *npj Comput. Mater.* **10**, 60 (2024) — MLIP for Tm calculation
- Cowley, *Phys. Rev.* **77**, 669 (1950) — Warren-Cowley SRO parameters

---

## Configuration

All tuneable parameters live in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEST_MODE` | `False` | Switch to `True` for a 5-composition quick validation |
| `N_COMPOSITIONS` | 100 | Number of alloy compositions |
| `LAMMPS_EXE` | (set this) | Path to LAMMPS binary |
| `GRACE_MODEL_DIR` | (set this) | Path to GRACE-2L-OMAT weights |
| `SLURM_PARTITION` | `compute` | Cluster partition name |
| `T_FRAC_MAX` | 0.80 | Upper T bound as fraction of Tm (ADP safety margin) |

---

## Citation

If you use this pipeline, please cite:

> [Your paper here]  
> Starikov et al. Phys. Rev. Materials **8**, 043603 (2024)  
> Karavaev et al. J. Chem. Phys. **144**, 194507 (2016)
