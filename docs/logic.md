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
a_0 = \sum_i x_i a_{0,i}
```

> Vegard, Z. Phys. 5, 17 (1921)

### S_f — Vacancy Formation Entropy

* **Caller:** `compute_sf_phonopy.py`
* **Mechanics:** Computes the vibrational entropy difference between a perfect BCC supercell and one containing a single vacancy using Phonopy.
1. Relaxes a $6\times6\times6$ perfect supercell (432 atoms). Displaces atoms to get $S_{perf}$.
2. Removes one atom (431 atoms). Runs Phonopy to calculate forces for 79-520 explicitly displaced structures (avoids point-group numerical noise). Extracts $S_{vac}$. 
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


The exact algorithmic steps—such as building an elongated supercell, freezing the solid half, heating the liquid half to $2 \times T_{m,ROM}$, and diagnosing melting/freezing via volume changes—are the practical molecular dynamics implementations of the solid-liquid coexistence method. These mechanical steps are coded directly in lammps_tm.py to satisfy the physical and thermodynamic requirements established by the quoted literature.


> Karavaev et al., J. Chem. Phys. 144, 194507 (2016)
> Zhu et al., npj Comput. Mater. 10, 60 (2024)

---

## Stage 4 — Diffusion

### MD Diffusion Run

* **Caller:** `lammps_diffusion.py`
* **Mechanics:** Reverts to the fast ADP potential.
1. Creates supercell and assigns elements.
2. MC Equilibration: Runs Monte Carlo atom-swapping under NPT to develop short-range chemical order (SRO).
3. Production MD: Deletes one atom to create a single vacancy and runs long-timescale MD under NPT.

## Stage 5 — Mean Square Displacement (MSD) Extraction

* **Caller:** `msd.py`
* **Data Source:** Parses the `log.lammps` file generated during the Stage 4 MD diffusion run.
* **Mechanics:** LAMMPS computes the displacement during the simulation. This script does not calculate positions; it extracts the raw `compute msd` outputs.
  1. Finds the thermo output header containing `c_msd_all[4]`.
  2. Extracts the simulation time step and converts it: $t = \text{Step} \times 0.0005$ ps.
  3. Extracts the 4th column of the MSD arrays (which represents the total scalar squared displacement $\Delta x^2 + \Delta y^2 + \Delta z^2$).
     * Total MSD: parsed from `c_msd_all[4]`.
     * Per-element MSD: parsed from `c_msd_w[4]`, `c_msd_mo[4]`, etc.
* **Outputs:** `msd_all.txt` and `msd_solo.txt`.

---

## Stage 6 — Vacancy Diffusion Coefficient ($D_v$)

* **Caller:** `Dv.py`
* **Data Source:** Reads `msd_all.txt` and `msd_solo.txt` from Stage 5. Reads atom counts ($N_{total}$, $N_i$) from the MD logs.
* **Mechanics:** Extracts the macroscopic diffusion slope. Because LAMMPS outputs the *average* atomic MSD, and the system contains exactly 1 vacancy, the vacancy's actual MSD is the total atomic MSD multiplied by the number of atoms $N$.
  1. Computes the linear slope of the MSD vs. time curve (discarding $t=0$).
  2. Applies the Einstein relation mapped to the vacancy:
     $$D_v = \frac{\text{slope} \times N}{6} \times 10^{-8}$$
     *(Where $10^{-8}$ converts Å²/ps to m²/s).*
  3. Repeats for individual elements ($D_{v,i}$) using $N_i$.
  4. Fits an Arrhenius curve to extract the vacancy migration barrier ($Q_v$):
     $$\ln D_v = \ln D_{v,0} - \frac{Q_v}{k_B T}$$
* **Outputs:** `Dv.txt` (Contains $T$, $1/T$, $D_{v,all}$, and $D_{v,i}$).

---

## Stage 7 — Equilibrium Vacancy Concentration (<i>C</i><sub>v</sub>)

* **Caller:** `Cv.py`
* **Data Source:** Pulls <i>E</i><sub>f</sub> and <i>C</i><sub>0</sub> from `constants_all.csv` (computed analytically in Stage 2).
* **Mechanics:** Purely analytical calculation. Evaluates the Arrhenius vacancy concentration equation across the simulation temperature grid.
  <br><br>
  <i>C</i><sub>v</sub>(<i>T</i>) = <i>C</i><sub>0</sub> &middot; exp(-<i>E</i><sub>f</sub> / <i>k</i><sub>B</sub><i>T</i>)
  <br><br>
  *(Where <i>C</i><sub>0</sub> = exp(<i>S</i><sub>f</sub> / <i>k</i><sub>B</sub>)).*
* **Outputs:** `Cv.txt` (Contains <i>T</i>, 1/<i>T</i>, and <i>C</i><sub>v</sub>).

---

## Stage 8 — Tracer Self-Diffusion (<i>D</i><sup>*</sup>)

* **Caller:** `D2.py`
* **Data Source:** Merges the macroscopic vacancy diffusion (`Dv.txt` from Stage 6) with the theoretical vacancy concentration (`Cv.txt` from Stage 7).
* **Mechanics:** Calculates the true, experimentally observable tracer diffusivity for each element. Because the original <i>D</i><sub>v</sub> was derived directly from the physical atomic displacement in MD, the BCC correlation factor (<i>f</i> &approx; 0.727) is natively embedded in the data.
  1. Computes per-element tracer diffusivity by dividing by the species mole fraction (<i>x</i><sub>i</sub>):
     <br><br>
     <i>D</i><sup>*</sup><sub>i</sub>(<i>T</i>) = [ <i>C</i><sub>v</sub>(<i>T</i>) &middot; <i>D</i><sub>v,i</sub> ] / <i>x</i><sub>i</sub>
     <br><br>
  2. Computes the total alloy tracer diffusivity:
     <br><br>
     <i>D</i><sup>*</sup><sub>total</sub> = <i>C</i><sub>v</sub>(<i>T</i>) &middot; &sum;<sub>i</sub> <i>D</i><sub>v,i</sub>
     <br><br>
  3. Fits the final Arrhenius parameters (Total activation energy <i>Q</i> and pre-exponential <i>D</i><sub>0</sub>) for the actual tracer diffusion.
* **Outputs:** `D2_components.txt` and Arrhenius plots (`Dtotal_vs_invT.png`).

---

## Stage 9 — Short-Range Order (SRO)

* **Caller:** `sro.py`
* **Data Source:** Parses `MD_T_<T>K.txt` generated by LAMMPS.
* **Mechanics:** This Python script **does not calculate** the SRO. LAMMPS calculates the Warren-Cowley parameters natively during the MD run using the `fix ave/time` command. This script acts purely as a data aggregator.
  1. Reads the time-averaged LAMMPS file.
  2. Extracts the final equilibrated row.
  3. Parses the actual temperature (`c_T_c`) and the Warren-Cowley parameters (`v_alphaWW`, `v_alphaWMo`, etc.).
* **Formula used by LAMMPS internally:**
  $$\alpha_{ij} = 1 - \frac{P_{ij}}{x_j}$$
* **Outputs:** `sro_vs_temp.csv` (Aggregates $\alpha_{ij}$ values for all active element pairs across all temperatures).

---

## Stage 10 — Polynomial Post-Processing

### Global Model Fitting (Ridge Regression)
* **Caller:** `postprocess.py`
* **Mechanics:** Aggregates tracer diffusivity ($D^*_{total}$) data across all 100 compositions and simulated temperatures into a single unified dataset. Fits a machine learning (Linear Regression with a safety net (Regularization)) regression model to predict diffusion based on alloy makeup and temperature.
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

---

### 1. The Physics Foundation: Why these variables?

Before applying Machine Learning, the code transforms the physical parameters into a mathematically linear format. The foundation of this is the **Arrhenius Equation** for diffusion:


$$D^* = D_0 \exp\left(-\frac{Q}{k_B T}\right)$$

Taking the natural logarithm of both sides linearizes the relationship with respect to temperature:


$$\ln(D^*) = \ln(D_0) - \left(\frac{Q}{k_B}\right) \frac{1}{T}$$

In a pure metal, $\ln(D_0)$ and $Q$ are constants. However, in a High-Entropy Alloy (HEA), the prefactor ($D_0$) and the activation energy ($Q$) are highly complex, unknown functions of the composition fractions ($x_i$).

Instead of fitting separate Arrhenius lines for every single composition, Stage 10 fits a **single, global polynomial model** that captures how composition *modifies* both the intercept and the slope of that line.

### 2. Exact Code Breakdown (`analysis/postprocess.py`)

#### A. Dimensionality Reduction (The Dummy Variable Trap)

```python
comp_feat_cols = [f"x_{e}" for e in ELEMENTS_ALL if e != "W"]

```

**What is happening:** The code drops Tungsten (`W`) from the feature list.
**The Theory:** Because the mole fractions must sum to 100% ($\sum x_i = 1$), the 6 variables are linearly dependent ($x_W = 1 - x_{Mo} - x_{Nb} - x_{Zr} - x_{Ti} - x_{Ta}$). In statistical modeling, feeding perfectly correlated variables into a regression causes the matrix inversion to fail (multicollinearity, or the "Dummy Variable Trap"). Dropping one element projects the data onto a set of 5 truly independent coordinates.

#### B. Feature Engineering (Normalization)

```python
inv_T = 1.0 / sub["T"].values
inv_T_mean = inv_T.mean()
sub["inv_T_norm"] = inv_T / inv_T_mean

```

**What is happening:** The inverse temperature ($1/T$) is divided by the mean of all inverse temperatures in the dataset.
**The Theory:** Machine learning models (especially those with L2 regularization) are highly sensitive to feature scaling. Compositions range from `0.0` to `1.0`, but $1/T$ values are on the order of `0.0003`. If left unscaled, the polynomial cross-terms would be infinitesimally small, causing the Ridge regression to improperly penalize the temperature weights. Normalizing centers the temperature feature around `1.0`.

#### C. The ML Pipeline (Polynomial Expansion & Ridge)

```python
X = sub[feat_cols].values
y = np.log(sub[target].values)

model = make_pipeline(
    PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
    Ridge(alpha=POLY_ALPHA),
)
model.fit(X, y)

```

**What is happening:** The code feeds the 6D array ($x_{Mo}, x_{Nb}, x_{Zr}, x_{Ti}, x_{Ta}, 1/T_{norm}$) into a standard `scikit-learn` pipeline.
**The Theory:** 1. **`PolynomialFeatures`**: If `POLY_DEGREE = 2`, this step automatically generates all squares ($x_{Mo}^2$) and cross-products ($x_{Mo} \cdot x_{Ti}$, $x_{Nb} \cdot 1/T_{norm}$).

* *Physical interpretation:* A cross-term like $x_{Mo} \cdot x_{Ti}$ represents the **binary chemical interaction** between Molybdenum and Titanium and how it alters the diffusion energy landscape. A term like $x_{Mo} \cdot 1/T_{norm}$ represents how adding Molybdenum changes the activation energy slope $Q$.

2. **`Ridge`**: Fits the equation by minimizing the Mean Squared Error *plus* a penalty term ($\alpha \sum \beta^2$).
* *Physical interpretation:* A degree-3 polynomial of 6 variables creates over 80 features. With only 300 data points (100 compositions $\times$ 3 temperatures), standard Ordinary Least Squares (OLS) would wildly overfit the MD noise. Ridge regularization mathematically forces the coefficients of less important chemical interactions close to zero, ensuring the model remains smooth and physically predictive between the sampled data points.



---

### 3. Origins of the Theory & Literature Citations

#### 1. Polynomials for Mixture Thermodynamics (CALPHAD)

The logic of using polynomial expansions to describe the complex properties of multicomponent mixtures comes from the **CALPHAD (CALculation of PHAse Diagrams)** methodology. Specifically, it mirrors the **Redlich-Kister polynomial** approach.

* **Theory:** In CALPHAD, the excess Gibbs free energy of mixing is modeled as a polynomial sum of pure components, binary interactions, and ternary interactions. The pipeline's use of `PolynomialFeatures` is a machine-learning equivalent of building a multi-dimensional Redlich-Kister expansion for the activation free energy of diffusion.
* **Citation:** Redlich, O., & Kister, A. T. (1948). *Algebraic representation of thermodynamic properties and the classification of solutions*. Industrial & Engineering Chemistry, 40(2), 345-348.

#### 2. Arrhenius Behavior in High-Entropy Alloys

The premise that tracer diffusivity in complex concentrated alloys still fundamentally obeys a macroscopic Arrhenius relationship (allowing us to fit $\ln(D)$ vs $1/T$) is established in the literature provided in your context.

* **Theory:** While local atomic jumps in an HEA have wildly varying energy barriers, the *macroscopic* tracer diffusion averages out into a predictable Arrhenius slope at high temperatures.
* **Citation:** Starikov, S., Grigorev, P., Drautz, R., & Divinski, S. V. (2024). *Large-scale atomistic simulation of diffusion in refractory metals and alloys*. Physical Review Materials, 8, 043603. (As seen in the provided PDF, defining the transition of diffusion behavior in CCAs).

#### 3. Ridge Regularization (Tikhonov Regularization)

The exact loss function applied by `Ridge(alpha=POLY_ALPHA)` is a foundational mathematical technique to solve ill-posed problems (like fitting 80 variables to 300 noisy MD data points).

* **Theory:** By adding the $L_2$ norm penalty ($\alpha ||\beta||_2^2$), the matrix $(X^T X + \alpha I)$ becomes strictly positive-definite and invertible, curing multicollinearity and overfitting.
* **Citation:** Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55-67.


<h3>Example: Global Model Prediction</h3>
<p>
  To predict diffusion for a specific alloy, the pipeline performs the following transformation:
</p>
<ul>
  <li><b>Feature Construction:</b> For a 50/50 W-Mo alloy at 2500 K, the input vector uses <i>x</i><sub>Mo</sub>=0.5 and a normalized inverse temperature (e.g., <i>T</i><sub>norm</sub>=1.0).</li>
  <li><b>Chemical Interaction:</b> The polynomial expansion generates a cross-term <i>x</i><sub>Mo</sub> &middot; 1/<i>T</i><sub>norm</sub>. Physically, this represents how Mo concentration modifies the activation energy <i>Q</i>.</li>
  <li><b>Regularization:</b> Ridge regression applies an <i>L</i><sub>2</sub> penalty (&alpha;) that shrinks the coefficients of high-order interactions. This prevents MD "noise" from creating physically impossible spikes in the predicted diffusion landscape.</li>
  <li><b>Final Result:</b> The model outputs ln(<i>D</i><sup>*</sup>), which is exponentiated to provide the final tracer diffusivity in m<sup>2</sup>/s.</li>
</ul>
