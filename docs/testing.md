```markdown
# Step-by-step Test Guide

Run this before any production job to verify the full pipeline works on your
cluster. Takes roughly 30–60 minutes total.

---

## 0. Prerequisites

```bash
# Confirm files are in place
ls WMoNbZrTiTa.nist.adp.txt      # must exist in project root
ls pipeline/ analysis/ config.py run_pipeline.py

# Confirm Python packages
pip install -r requirements.txt

# Confirm LAMMPS and Phonopy are in PATH on a compute node (not login node)
srun --partition=compute --pty bash -c "which lmp && which phonopy"

```

---

## 1. Enable TEST_MODE

In `config.py`, set:

```python
TEST_MODE = True

```

This gives: 5 compositions, 3 temperatures each, short MC/MD runs, 3 T_guess
values for coexistence. Total cluster time: ~1–2 h.

---

## 2. Stage 0 — Environment check

```bash
python run_pipeline.py --only_stage 0

```

**Expected output:**

```
[OK] numpy
[OK] pandas
[OK] matplotlib
[OK] scipy
[OK] scikit-learn
[OK] phonopy
[OK] ADP potential: WMoNbZrTiTa.nist.adp.txt
[WARN] 'lmp' not in PATH (expected on compute nodes)   ← OK on login node
Mode: TEST
Stage 0 complete.

```

**If `[FAIL]`:** install the missing package with `pip install <name>`.

**If potential `[FAIL]`:** confirm the file is in the project root.

---

## 3. Stage 1 — Compositions

```bash
python run_pipeline.py --only_stage 1

```

**Expected output:**

```
[compositions] 5 compositions  →  results/compositions.csv
Stage 1 complete — 5 compositions.

```

**Check:**

```bash
cat results/compositions.csv
# Should have 6 columns: comp_id, x_W, x_Mo, x_Nb, x_Zr, x_Ti, x_Ta
# Each row should sum to 1.0
python3 -c "
import pandas as pd
df = pd.read_csv('results/compositions.csv')
frac_cols = [c for c in df.columns if c.startswith('x_')]
print(df[frac_cols].sum(axis=1).round(5))   # all should be 1.0
"

```

---

## 4. Pre-Stage 2 — Pure Element Phonopy ($S_f$ & $a_0$)

Run the standalone script to calculate exact ADP relaxed $a_0$ and vacancy formation entropy ($S_f$) for pure elements. To keep the test short, we test just one element here:

```bash
python pipeline/compute_sf_phonopy.py --elements W

```

**Expected output:**

```
[W] Relaxed ADP a0 = 3.18642 Å (Ref: 3.185 Å)
[W] T0=300 K  a0=3.18642 Å  N=432
  [perfect] 1 displaced structures
  [perfect] S = 1824.6412 kB/supercell at 300 K
  [vacancy] 2586 displaced structures
...
=== Sf_phonopy results ===
  W: X.XXXX kB
Update SF_REF in config.py with these values.
File: results/Sf_phonopy.txt

```

**Check:** Confirm `results/Sf_phonopy.txt` is created. For a full production run, you run without `--elements` and update `config.py` with the final output array.

---

## 5. Stage 2 — Physical constants

```bash
python run_pipeline.py --only_stage 2

```

**Expected output (one block per composition):**

```
[constants] comp_000  active=['W', 'Mo', ...]
  a0: Vegard=3.1850 Å  LAMMPS=3.1864 Å
  Tm=2650 K  Ef=3.1234 eV  C0=7.3891  T_grid=[1325, 1450, 1600]
...
[constants] Saved 5 rows  →  results/constants_all.csv

```

**Check:**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/constants_all.csv')
print(df[['comp_id','a0_vegard','Tm','Ef','Sf','C0']].to_string())
# Ef should be 1–4 eV; Tm should be 1900–3700 K; C0 = exp(Sf) ~ 7–15
"

```

If LAMMPS is not available on the login node, Stage 2 will print:

```
[WARN] a0 LAMMPS failed: ... — using Vegard's law
[WARN] Ef LAMMPS failed: ...
[WARN] Using fallback Ef = X.XXXX eV

```

This is acceptable for a connectivity test. $a_0$ and $E_f$ will be correct on compute nodes.

---

## 6. Stage 3b — Tm coexistence inputs

```bash
python run_pipeline.py --only_stage 31

```

**Expected output:**

```
[lammps_tm] 15 coexistence directories  →  results/tm_job_list.csv
[lammps_tm] Written: slurm/submit_Tm_array.sh  (15 array tasks)

  Next: sbatch slurm/submit_Tm_array.sh

```

**Check:**

```bash
# Verify one coexistence script was generated correctly
head -20 runs/comp_000/tm_coexistence/T_*/coexistence.in
# Should show:  lattice bcc ...  (not fcc)
#               create_box 6 box
#               pair_style grace

# Verify array script
head -15 slurm/submit_Tm_array.sh
# Should show:  #SBATCH --array=0-14%20

```

**Submit:**

```bash
sbatch slurm/submit_Tm_array.sh
# Note the job ID, e.g. 12345

```

**Monitor:**

```bash
watch -n 30 "sacct -j 12345 --format=JobID,State,Elapsed | tail -20"
# Wait until all 15 tasks show COMPLETED

```

---

## 7. Stage 3c — Patch Tm into constants

```bash
python run_pipeline.py --only_stage 32

```

**Expected output:**

```
  comp_000: Tm = 2741 K  (spread 0 K, 1 pts)  T_grid = [1375, ...]
  comp_001: Tm = 2583 K  ...
...
[lammps_tm] Patched 5/5 compositions  →  results/constants_all.csv

```

**If a composition shows `no converged run — keeping Tm_rom`:**
The coexistence job at that T_guess may have crashed or not equilibrated.
Check the log:

```bash
cat runs/comp_000/tm_coexistence/T_<guess>/log.coexistence | grep -E "COEXISTENCE|ERROR"

```

The bracketing covers ±20% of Tm_rom. If the real Tm is outside this range, widen `_BRACKET_FACTORS` in `pipeline/lammps_tm.py`.

**Check that Tm was updated:**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/constants_all.csv')
print(df[['comp_id','Tm','T_grid']])
# Tm values should differ from the Rule-of-Mixtures estimates in Stage 2
"

```

---

## 8. Stage 3 — Diffusion inputs

```bash
python run_pipeline.py --only_stage 3

```

**Expected output:**

```
[diffusion inputs] 15 directories created

```

**Check:**

```bash
# Verify r_sro is composition-specific based on a0
grep 'r_sro equal' runs/comp_000/sim_*/bcc_vac_adv.in
# e.g.  variable  r_sro equal  3.0789

grep 'r_sro equal' runs/comp_004/sim_*/bcc_vac_adv.in
# Should differ from comp_000 if compositions have different a0

```

---

## 9. Stage 4 — Submit diffusion jobs

```bash
python run_pipeline.py --only_stage 4

```

**Expected:**

```
Written: slurm/submit_diffusion.sh

```

```bash
sbatch slurm/submit_diffusion.sh
# Monitor until all tasks COMPLETED

```

---

## 10. Stages 5–10 — Analysis

Once diffusion jobs are done:

```bash
python run_pipeline.py --from_stage 5

```

Each stage prints per-composition progress. No cluster jobs — runs locally.

**Stage 5 check:** look for MSD files:

```bash
ls results/comp_000/sim_x/

```

**Stage 8 check:** verify Arrhenius R² is reasonable:

```bash
grep 'R2\|r2\|rsq' results/comp_000/txt/D2_*.txt | head -10
# R² > 0.95 expected; < 0.90 indicates MSD not converged at low T

```

**Stage 10 check:** plots exist:

```bash
ls results/plots/

```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `LAMMPS exit 1` in Tm jobs | GRACE not loaded | Check `micromamba activate grace` in SLURM script |
| `pair_coeff` error in Tm log | Wrong GRACE_MODEL_DIR | Set correct path in `config.py` |
| `no converged run` for many comps | T_guess bracketing too narrow | Check if Tm_rom is far from real Tm; widen `_BRACKET_FACTORS` |
| MSD not linear (R² < 0.9) | Run too short at low T | Increase `N_MD` or raise `T_FRAC_MIN` in `config.py` |
| Phonopy imaginary frequencies | Asymmetric relaxation | Verify `--tolerance=1e-2` and `--nosym` are active in `compute_sf_phonopy.py` |
| Stage 32 keeps Tm_rom for all | Tm jobs crashed silently | Check `Tm_*.err` files |

---

## Reset and rerun

```bash
# Wipe all runtime output and start fresh
rm -rf results/ runs/ slurm/submit_Tm_array.sh slurm/submit_diffusion.sh
python run_pipeline.py --from_stage 0

```

```

```
