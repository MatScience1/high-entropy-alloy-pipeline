# Pipeline Logic

---

## Stage 1 — Compositions

### Alloy Generation

* **Caller:** `compositions.py`
* **Mechanics:** Generates a 100-point uniform grid across the 6-element (W-Mo-Nb-Zr-Ti-Ta) composition space. First, it defines structured vertices (pure elements, equiatomic binary/ternary/quaternary/quinary alloys). Then, it fills the remaining slots using a symmetric Dirichlet distribution ($\alpha=1$) to randomly but uniformly sample the probability simplex.
* **Why:** To ensure unbiased statistical coverage of the entire High-Entropy Alloy composition space without clustering in the center or edges.

---

## Stage 2 & 3 — Physical Constants

### a0 — Equilibrium lattice parameter

We calculate the equilibrium lattice parameter ($a_0$) to ensure the simulated crystal has zero pressure, preventing artificial stress from skewing diffusion dynamics. The pipeline handles $a_0$ in two distinct locations for two different purposes:

**1. Pure Elements (for Vacancy Entropy $S_f$)**

* **Caller:** `compute_sf_phonopy.py`
* **Mechanics:** Builds a standard $1 \times 1 \times 1$ BCC unit cell of a single pure element. Runs a LAMMPS `box/relax` at $P=0$ followed by `minimize`.
* **Why:** Phonopy is hyper-sensitive to atomic forces. The unit cell must be perfectly relaxed to the specific ADP potential to avoid imaginary phonon frequencies before building the $6 \times 6 \times 6$ supercell.

**2. HEA Compositions (for Main Pipeline)**

* **Caller:** `constants.py`
* **Mechanics:** Builds a $5 \times 5 \times 5$ BCC supercell (250 atoms). Distributes elements randomly matching the exact mole fractions ($x_i$) using a sequential conditional probability assignment. Runs a LAMMPS `box/relax` at $P=0$ followed by `minimize` to relax localized chemical stress.
* **Fallback:** If the MD minimization fails (e.g., highly unstable phase), the system defaults to Vegard's Law (linear atomic volume mixing):

```text
$$a_0 = \sum_i x_i a_{0,i}$$
```

> Vegard, Z. Phys. 5, 17 (1921)

### S_f — Vacancy Formation Entropy

* **Caller:** `compute_sf_phonopy.py`
* **Mechanics:** Computes the vibrational entropy difference between a perfect BCC supercell and one containing a single vacancy using Phonopy.
1. Relaxes a $6\times6\times6$ perfect supercell (432 atoms). Displaces atoms to get $S_{perf}$.
2. Removes one atom (431 atoms). Runs Phonopy with the `--nosym` flag to calculate forces for 2,586 explicitly displaced structures (avoids point-group numerical noise). Extracts $S_{vac}$.
3. Evaluates entropy at $T_0$ ($300$ K for W, Mo, Nb, Ta; $1100$ K for Zr, $1200$ K for Ti to ensure BCC stability).


* **Formula:**
```text
Sf = S_vac_supercell - ((N-1)/N) * S_perf_supercell

```



> Starikov et al., Phys. Rev. Materials 8, 043603 (2024), Sec III.A

### E_f — Vacancy Formation Energy

* **Caller:** `constants.py`
* **Mechanics:** Uses LAMMPS with the ADP potential on a $5\times5\times5$ supercell (250 atoms). Calculates the total potential energy of the perfect mixed alloy crystal $E(N)$. Then, it sequentially removes an atom at `EF_N_SITES` different random locations, minimizing the structure each time to find $E(N-1, \text{site}_i)$.
* **Formula:** Computes the mean energy required to form a vacancy, using the average atom energy $E(N)/N$ as the chemical potential:
```text
⟨Ef⟩ = mean_i [ E(N−1, site_i) ] − ((N−1)/N) * E(N)

```



> Starikov et al., Phys. Rev. Materials 8, 043603 (2024), Eq. 7

### C_0 — Vacancy Concentration Prefactor

* **Caller:** `constants.py` (computed for use in Stage 7 `Cv.py`)
* **Mechanics:** An analytical calculation derived directly from the linear mixture of the vacancy formation entropy computed earlier ($S_f = \sum x_i S_{f,i}$).
* **Formula:** 

```text
C0 = exp(Sf / k_B)
```

> Shewmon, Diffusion in Solids. McGraw-Hill (1963)

### T_m — True Melting Temperature (Phase Coexistence)

* **Caller:** `lammps_tm.py`
* **Mechanics:** Uses the Modified Z-method governed by the highly accurate GRACE Machine Learning Interatomic Potential (MLIP).
1. Builds an elongated $NX \times NY \times NZ$ BCC supercell.
2. Freezes the solid half ($z \in [0, NZ/2]$).
3. Heats the liquid half to $2 \times T_{m,ROM}$ (Rule of Mixtures estimate) until it melts.
4. Releases all atoms and runs NPT at a test temperature ($T_{guess}$) with *independent* x, y, and z barostats (crucial: iso barostats destroy the solid-liquid interface).
5. Performs stress equalization ($P_{xx} \approx P_{yy} \approx P_{zz} \approx 0$).
6. **Diagnosis:** If volume increases, the system is melting ($T_{guess} > T_m$). If volume decreases, it is freezing ($T_{guess} < T_m$). True $T_m$ is found where volume is stable and pressure is near zero.



> Karavaev et al., J. Chem. Phys. 144, 194507 (2016)
> Zhu et al., npj Comput. Mater. 10, 60 (2024)

---

## Stage 4 & 5 — Diffusion & MSD

### MD Diffusion Run

* **Caller:** `lammps_diffusion.py`
* **Mechanics:** Reverts to the fast ADP potential.
1. Creates supercell and assigns elements.
2. MC Equilibration: Runs Monte Carlo atom-swapping under NPT to develop short-range chemical order (SRO).
3. Production MD: Deletes one atom to create a single vacancy and runs long-timescale MD under NPT.



### Mean Square Displacement (MSD)

* **Caller:** `msd.py`
* **Mechanics:** Parses LAMMPS log to extract the unwrapped distance atoms travel over time.
```text
MSD(t) = ⟨|r(t) - r(0)|²⟩

```



---

## Stage 6 to 8 — Tracer Diffusivity (D*)

### D* Calculation

* **Caller:** `D2.py`, `Cv.py`
* **Mechanics:** Extracts the macroscopic diffusion rate. Because MSD tracks the atoms directly, the BCC correlation factor ($f \approx 0.727$) is naturally included.
1. **Vacancy Diffusivity:** 

```text
Dv(T) = lim_{t→∞} MSD_vac(t) / 6t
```

2. **Equilibrium Vacancy Concentration:**
```text
Cv(T) = C0 * exp(−Ef / (k_B * T))

```


3. **Tracer Diffusivity (per element $i$):**
```text
D*_i(T) = Cv * Dv_i / x_i

```


---

## Stage 9 — Short-Range Order (SRO)

### Chemical Ordering

* **Caller:** `sro.py`
* **Mechanics:** Calculates the Warren-Cowley SRO parameter based on the first two neighbor shells (BCC 1NN/2NN midpoint cutoff). Evaluates whether specific elemental pairs attract or repel.
```text
α_ij = 1 − P_ij / x_j

```


Where $P_{ij}$ is the conditional probability of finding $j$ next to $i$.
* $\alpha_{ij} = 0$: Random solid solution
* $\alpha_{ij} > 0$: Elements repel (depletion)
* $\alpha_{ij} < 0$: Elements attract (ordering)



> Cowley, Phys. Rev. 77, 669 (1950)

---

## Stage 10 — Polynomial Post-Processing

### Global Model Fitting (Ridge Regression)
* **Caller:** `postprocess.py`
* **Mechanics:** Aggregates tracer diffusivity ($D^*_{total}$) data across all 100 compositions and simulated temperatures into a single unified dataset. Fits a machine learning regression model to predict diffusion based on alloy makeup and temperature.
  1. **Dimensionality Reduction:** Drops one element (W) from the inputs. Because mole fractions sum to 1 ($\sum x_i = 1$), including all 6 elements causes perfect multicollinearity. 
  2. **Feature Engineering:** Creates a 6-dimensional input vector $\mathbf{x}$ for every data point, combining the 5 independent compositions and a normalized inverse temperature (to keep numerical scaling stable).
  3. **Polynomial Expansion:** Generates higher-order terms (squares, cross-products) up to `POLY_DEGREE` to capture non-linear chemical interactions (e.g., how adding Mo and Ti *together* affects diffusion differently than adding them separately).
  4. **Ridge Regression:** Fits the coefficients ($\beta$) to predict $\ln(D^*_{total})$ using L2 regularization (`POLY_ALPHA`) to penalize overly large coefficients, preventing the model from overfitting the MD noise.

* **Mathematical Formulation:**
  * **Feature Vector:**
    ```text
    x = [ x_Mo, x_Nb, x_Zr, x_Ti, x_Ta, (1/T) / mean(1/T) ]
    ```
  * **Polynomial Prediction ($y$):**
    ```text
    ln(D*_total) = β_0 + Σ_i (β_i * x_i) + Σ_{i ≤ j} (β_{ij} * x_i * x_j) + ...
    ```
  * **Ridge Loss Function (minimized during fit):**
    ```text
    L(β) = Σ_n [ ln(D*_total, n) - predicted_n ]² + α * Σ_k (β_k)²
    ```
    *(Where n is the number of data points, k is the number of polynomial features, and $\alpha$ is the regularization strength).*

* **Why:** By using a joint composition-and-temperature fit with L2 regularization, the pipeline uses the entire statistical weight of the $100 \times N_{temps}$ simulations simultaneously. This allows smooth, stable interpolation of diffusion properties anywhere inside the 6-element High-Entropy Alloy hyperspace.
