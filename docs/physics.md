# Physics Reference

Mathematical derivations, physical models, and literature sources for every
quantity computed in the WMoNbZrTiTa diffusion pipeline.

---

## 1. Composition space

### 1.1 Sampling strategy

The alloy composition space is a 5-dimensional simplex (6 elements, constraint $\sum_i x_i = 1$). We cover it with:

- **Structured vertices**: all equal-molar sub-alloys (unary through senary) — 63 deterministic compositions providing boundary and edge coverage.
- **Random interior**: symmetric Dirichlet distribution with concentration parameter $\alpha = 1$:

$$f(x_1, \ldots, x_6) = \frac{\Gamma(6)}{\Gamma(1)^6} \prod_{i=1}^{6} x_i^{\alpha - 1} = \text{const}$$

$\alpha = 1$ gives the **uniform distribution on the simplex**: no region is preferentially sampled. Equivalent to normalising 6 i.i.d. Uniform(0,1) draws (the stick-breaking construction).

---

## 2. Lattice parameter — Vegard's law

$$a_0^{\text{alloy}} = \sum_i x_i \, a_{0,i}$$

Reference values $a_{0,i}$ (BCC, 0 K, ADP potential):

| Element | $a_0$ (Å) | Source |
|---------|-----------|--------|
| W  | 3.185 | Starikov 2024, Table S1 |
| Mo | 3.168 | Starikov 2024, Table S1 |
| Nb | 3.320 | Starikov 2024, Table S1 |
| Zr | 3.580 | Starikov 2024, Table S1 |
| Ti | 3.260 | Starikov 2024, Table S1 |
| Ta | 3.315 | Starikov 2024, Table S1 |

> Vegard, L. Z. Phys. **5**, 17–26 (1921).  
> Starikov et al. Phys. Rev. Materials **8**, 043603 (2024).

---

## 3. Melting temperature — phase coexistence

### 3.1 Preliminary estimate (Rule of Mixtures)

Used only as the initial bracket centre for coexistence runs:

$$T_m^{\text{rom}} = \sum_i x_i \, T_{m,i}$$

### 3.2 Real $T_m$ — Modified Z-method (Karavaev 2016)

An elongated BCC supercell ($N_x \times N_y \times N_z = 10 \times 10 \times 40 = 8000$ atoms) is partitioned into solid ($z \in [0, N_z/2]$) and liquid ($z \in [N_z/2, N_z]$) halves.

**Protocol:**
1. Equilibrate full solid at $T_{\text{solid}} = 0.5 \, T_m^{\text{rom}}$ (NVT).
2. Freeze solid half ($\texttt{setforce 0 0 0}$); heat liquid half to $T_{\text{liquid}} = 2 \, T_m^{\text{rom}}$ (NVT, 50 000 steps) until disordered.
3. Release all atoms. Run NPT at $T_{\text{guess}}$ with **independent** $x$, $y$, $z$ barostats:

$$\text{fix npt temp } T_g \; T_g \; \tau_T \;\; x \; 0 \; 0 \; \tau_P \;\; y \; 0 \; 0 \; \tau_P \;\; z \; 0 \; 0 \; \tau_P$$

   Using a single isotropic barostat (`iso`) is incorrect: it prevents differential relaxation of solid and liquid sublattices.

4. **Stress equalization** (Karavaev mandate): tighten barostat damping ($\tau_P^{\text{eq}} = 0.05$ ps) until $P_{xx} \approx P_{yy} \approx P_{zz} \approx 0$. Without this step, normal stress anisotropy biases $T_m$.

**Detection:** from the equilibrated pressure trajectory,

- $\langle \text{vol} \rangle$ increasing → melting ($T_{\text{guess}} > T_m$)
- $\langle \text{vol} \rangle$ decreasing → freezing ($T_{\text{guess}} < T_m$)
- $|\langle P \rangle| < 500$ bar and $\langle \text{vol} \rangle$ stable → coexistence → $T_{\text{guess}} \approx T_m$

Five bracketing values $T_{\text{guess}} \in \{0.80, 0.90, 1.00, 1.10, 1.20\} \times T_m^{\text{rom}}$ are run in parallel (SLURM array). Converged runs are averaged.

> Karavaev et al. J. Chem. Phys. **144**, 194507 (2016).  
> Zhu et al. npj Comput. Mater. **10**, 60 (2024).

---

## 4. Vacancy formation energy

### 4.1 Formula (Starikov 2024, Eq. 7)

For a disordered alloy, the average vacancy formation energy is:

$$\langle E_f \rangle = \frac{1}{N} \sum_{i=1}^{N} \left[ E(N-1, \text{site } i) - \frac{N-1}{N} E(N) \right]$$

$$= \langle E^{\text{vac}} \rangle - \frac{N-1}{N} E_0$$

where:
- $E(N)$ = DFT/MLIP energy of the perfect $N$-atom supercell
- $E(N-1, i)$ = energy after removing atom $i$ and re-minimising
- $\frac{E(N)}{N}$ = chemical potential of the removed atom (alloy reference)

Averaged over $N_{\text{sites}} = 50$ randomly selected sites to reduce statistical noise from compositional disorder.

**Implementation:** 5×5×5 BCC supercell (250 atoms), ADP potential, conjugate-gradient minimisation to $10^{-12}$ eV/Å.

> Starikov et al. Phys. Rev. Materials **8**, 043603 (2024), Eq. 7.

### 4.2 Reference values (DFT, for fallback)

| Element | $E_f$ (eV) | Source |
|---------|-----------|--------|
| W  | 3.56 | Domain et al., Acta Mater. 2004 |
| Mo | 2.88 | Ehrhart et al. |
| Nb | 2.91 | Ehrhart et al. |
| Zr | 2.39 | Estimation |
| Ti | 1.84 | Estimation |
| Ta | 3.03 | Estimation |

---

## 5. Vacancy concentration

Arrhenius model (Shewmon 1963):

$$C_v(T) = C_0 \exp\!\left(-\frac{E_f}{k_B T}\right)$$

$$C_0 = \exp\!\left(\frac{S_f}{k_B}\right)$$

The vacancy formation entropy $S_f$ (in units of $k_B$) is composition-weighted:

$$S_f = \sum_i x_i \, S_{f,i}$$

Reference values from Starikov 2024: $S_{f,W} = 2.7\,k_B$, $S_{f,\text{others}} \approx 2.0\,k_B$.

> Shewmon, P. G. *Diffusion in Solids*. McGraw-Hill, 1963.  
> Starikov et al. Phys. Rev. Materials **8**, 043603 (2024).

---

## 6. Alloy composition in LAMMPS — sequential-fraction formula

LAMMPS `set group all type/fraction N f seed` independently assigns type $N$ with probability $f$ to **all** atoms, including those already typed by earlier commands.

Setting order: Mo → Nb → Zr → Ti → Ta (W is the base type, never explicitly set).

The probability of ending as element $X$ after all set operations:

$$P(\text{final} = X) = f_X \prod_{Y \text{ set after } X} (1 - f_Y)$$

Inverting in reverse order (Ta last → not overwritten):

$$f_{\text{Ta}} = c_{\text{Ta}}, \quad f_{\text{Ti}} = \frac{c_{\text{Ti}}}{1 - c_{\text{Ta}}}, \quad f_{\text{Zr}} = \frac{c_{\text{Zr}}}{1 - c_{\text{Ta}} - c_{\text{Ti}}}, \quad \ldots$$

The **incorrect** formula in the original `bcc_vac_adv.in`:

$$f_{\text{Mo}}^{\text{wrong}} = \frac{c_{\text{Mo}}}{(1 - c_{\text{Nb}})(1 - c_{\text{Zr}})(1 - c_{\text{Ti}})(1 - c_{\text{Ta}})}$$

This expands to $1 - c_{\text{Nb}} - c_{\text{Zr}} - c_{\text{Ti}} - c_{\text{Ta}} + O(c^2)$, overestimating the denominator by cross-terms of order $c^2 \sim 0.04$ for equiatomic alloys (6–12% error in set fractions).

---

## 7. SRO cutoff — composition-dependent

The BCC first nearest-neighbour distance: $r_{\text{1NN}} = a_0 \sqrt{3}/2$.  
The BCC second nearest-neighbour distance: $r_{\text{2NN}} = a_0$.

Safe cutoff (captures all 1NN, excludes all 2NN):

$$r_{\text{SRO}} = \frac{r_{\text{1NN}} + r_{\text{2NN}}}{2} = a_0 \cdot \frac{\sqrt{3}/2 + 1}{2} \approx 0.933 \, a_0$$

For the full composition space ($a_0^{\text{Vegard}} \in [3.19, 3.52]$ Å), the gap between $r_{\text{SRO}}$ and $r_{\text{2NN}}$ is $\geq 210$ pm.

**Why a fixed cutoff (3.0 Å) fails:** pure Zr has $a_0 = 3.58$ Å and $r_{\text{1NN}} = 3.10$ Å. A fixed 3.0 Å cutoff misses the entire Zr first shell, making all Zr-centred Warren-Cowley parameters incorrect.

---

## 8. Warren-Cowley short-range order

The Warren-Cowley SRO parameter (Cowley 1950):

$$\alpha_{ij} = 1 - \frac{P_{ij}}{x_j}$$

where $P_{ij}$ is the conditional probability of finding element $j$ as a nearest neighbour of $i$:

$$P_{ij} = \frac{\bar{n}_{ij}}{z_i}, \quad z_i = \sum_j \bar{n}_{ij}$$

$\bar{n}_{ij}$ is the time- and ensemble-averaged number of $j$ neighbours within $r_{\text{SRO}}$ of each $i$ atom.

**In LAMMPS:** computed via `compute coord/atom` + `reduce ave` per element pair, time-averaged with `fix ave/time`.

Interpretation:
- $\alpha_{ij} = 0$: random solid solution
- $\alpha_{ij} > 0$: $i$–$j$ pairs depleted (like-atom clustering / phase separation tendency)
- $\alpha_{ij} < 0$: $i$–$j$ pairs enriched (chemical ordering tendency)

> Cowley, J. M. Phys. Rev. **77**, 669 (1950).

---

## 9. Tracer self-diffusion

### 9.1 Einstein relation

$$D^*_i(T) = \lim_{t \to \infty} \frac{\langle | \mathbf{r}_i(t) - \mathbf{r}_i(0) |^2 \rangle}{6t}$$

where $\mathbf{r}_i(t)$ is the **unwrapped** position of species-$i$ atoms (LAMMPS `compute msd` handles image-flag unwrapping automatically).

Units: $[D^*] = $ Å² ps⁻¹. Convert to m² s⁻¹: multiply by $10^{-8}$ (= $10^{-20}$ m² / $10^{-12}$ s).

### 9.2 Arrhenius fit

$$\ln D^*_i = \ln D_{0,i} - \frac{Q_i}{k_B T}$$

Linear regression in $1/T$ space. The slope gives $-Q_i/k_B$ (migration barrier) and intercept gives $\ln D_{0,i}$ (attempt frequency prefactor).

### 9.3 Note on correlation factor

$D^*_i$ measured from MSD is the **tracer** diffusivity, which already accounts for the correlation factor $f$ (geometric probability that successive vacancy jumps are uncorrelated). For BCC: $f \approx 0.727$. The uncorrelated jump diffusivity would be $D_J = D^* / f$, but $D^*$ is the experimentally comparable quantity.

### 9.4 Homologous temperature normalisation

The primary plot is $D^*_i$ vs $T / T_m$ where $T_m$ is the **alloy's** melting temperature from the GRACE coexistence calculation (Stage 3b/3c). This collapses data from 100 compositions with different absolute $T_m$ onto a single comparable axis.

---

## 10. Composition–property polynomial model

A Ridge-regularised polynomial in the 5 independent composition coordinates $\xi = (\xi_1, \ldots, \xi_5)$ (simplex projection, removing the linear constraint):

$$y(\mathbf{x}) = \beta_0 + \sum_j \beta_j \xi_j + \sum_{j \leq k} \beta_{jk} \xi_j \xi_k + \ldots$$

Fitted separately for each target property $y \in \{\ln D_{0,i}, Q_i, \alpha_{ij}, E_f\}$.

Regularisation: Ridge (L2), penalty $\lambda \|\boldsymbol{\beta}\|^2$, $\lambda = 10^{-3}$.

Degree 2 used in production ($R^2 > 0.98$ reported for $D^*$); degree 1 in TEST_MODE.

---

## References

| # | Citation |
|---|---------|
| 1 | Starikov, S. et al. *Phys. Rev. Materials* **8**, 043603 (2024) |
| 2 | Karavaev, A. V. et al. *J. Chem. Phys.* **144**, 194507 (2016) |
| 3 | Zhu, J. et al. *npj Comput. Mater.* **10**, 60 (2024) |
| 4 | Cowley, J. M. *Phys. Rev.* **77**, 669 (1950) |
| 5 | Vegard, L. *Z. Phys.* **5**, 17 (1921) |
| 6 | Shewmon, P. G. *Diffusion in Solids*. McGraw-Hill (1963) |
| 7 | Gutierrez, G. et al. *J. Appl. Phys.* **123**, 045108 (2018) |
| 8 | Gunawardana, K. G. S. H. et al. *Phys. Rev. E* **90**, 023301 (2014) |
