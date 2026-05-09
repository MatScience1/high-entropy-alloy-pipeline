"""
analysis/postprocess.py
=======================
Stage 10 — Aggregate results, polynomial fit, comparison plots.

Interface (called by run_pipeline.py)
--------------------------------------
    run_postprocess() -> None

What this module does
---------------------
1. Reads results/<comp_id>/txt/D2_*_vs_T.txt for all compositions.
2. Builds a feature matrix X ∈ R^{N×5} from mole fractions (6 elements
   on the composition simplex → 5 independent coordinates).
3. Fits a Ridge-regularised polynomial of degree POLY_DEGREE to each
   Arrhenius parameter (ln D0_i, Q_i) for each element i:

       y = β₀ + Σⱼ βⱼ xⱼ + Σⱼ≤ₖ βⱼₖ xⱼ xₖ + …

   Regularisation strength: POLY_ALPHA (Ridge, from config).

4. Plots:
     - D*(T/Tm) vs T/Tm for all compositions (homologous temperature axis)
     - Q_i composition map (ternary projections)
     - Arrhenius comparison vs literature / Starikov 2024

Output
------
    results/summary.csv
    results/plots/D_vs_homologous_T.png
    results/plots/Q_composition_map.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from config import (
    ELEMENTS, POLY_ALPHA, POLY_DEGREE, RESULTS_DIR, RUNS_DIR,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ELEMENTS_ALL = ELEMENTS
DIFF_COLS    = [f"D_{e}" for e in ELEMENTS_ALL] + ["D_total",
                f"Dv_all"] + [f"Dv_{e}" for e in ELEMENTS_ALL]


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Arrhenius summary for a diffusion array
# ══════════════════════════════════════════════════════════════════════════════

def arrhenius_params(T_arr: np.ndarray, D_arr: np.ndarray
                     ) -> dict[str, float]:
    mask = (D_arr > 0) & ~np.isnan(D_arr) & (T_arr > 0)
    if mask.sum() < 3:
        return {"D0": float("nan"), "Q_eV": float("nan"), "R2": float("nan")}
    res = linregress(1.0 / T_arr[mask], np.log(D_arr[mask]))
    return {
        "D0":   float(np.exp(res.intercept)),
        "Q_eV": float(-res.slope * 8.617333e-5),
        "R2":   float(res.rvalue**2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1-3: collect data
# ══════════════════════════════════════════════════════════════════════════════

def collect_results(compositions_df: pd.DataFrame,
                    constants_df: pd.DataFrame) -> pd.DataFrame:
    """Build master_results DataFrame from per-composition result files."""
    master_rows: list[dict] = []

    for _, crow in compositions_df.iterrows():
        comp_id = crow["comp_id"]
        comp = {e: float(crow[f"x_{e}"]) for e in ELEMENTS_ALL}

        # constants
        const_row_df = constants_df[constants_df["comp_id"] == comp_id]
        if const_row_df.empty:
            print(f"  [WARN] no constants for {comp_id}")
            continue
        cst = const_row_df.iloc[0]
        Ef = float(cst["Ef"])
        Sf = float(cst["Sf"])
        C0 = float(cst["C0"])
        Tm = float(cst["Tm"])

        # T grid
        t_grid_raw = cst["T_grid"]
        try:
            T_grid = json.loads(t_grid_raw) if isinstance(t_grid_raw, str) else list(t_grid_raw)
        except Exception:
            continue

        # Per-composition results directories
        comp_res_dir  = RESULTS_DIR / comp_id
        comp_txt_dir  = comp_res_dir / "txt"
        comp_plot_dir = comp_res_dir / "plots"

        # --- D2_components.txt ---
        d2_path = comp_txt_dir / "D2_components.txt"
        d2_data: dict[str, np.ndarray] = {}
        T_d2: np.ndarray | None = None
        if d2_path.exists():
            with open(d2_path) as fh:
                hdr = fh.readline().lstrip("#").strip()
            col_names = hdr.split()
            raw = np.genfromtxt(d2_path, skip_header=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            T_d2 = raw[:, 0]
            for j, cn in enumerate(col_names):
                d2_data[cn] = raw[:, j]

        # --- Dv.txt ---
        dv_path = comp_txt_dir / "Dv.txt"
        dv_data: dict[str, np.ndarray] = {}
        T_dv: np.ndarray | None = None
        if dv_path.exists():
            with open(dv_path) as fh:
                hdr = fh.readline().lstrip("#").strip()
            col_names_dv = hdr.split()
            raw_dv = np.genfromtxt(dv_path, skip_header=1)
            if raw_dv.ndim == 1:
                raw_dv = raw_dv.reshape(1, -1)
            T_dv = raw_dv[:, 0]
            for j, cn in enumerate(col_names_dv):
                dv_data[cn] = raw_dv[:, j]

        # --- Cv.txt ---
        cv_path = comp_txt_dir / "Cv.txt"
        cv_vals: dict[float, float] = {}
        if cv_path.exists():
            raw_cv = np.genfromtxt(cv_path, skip_header=2)
            if raw_cv.ndim == 1:
                raw_cv = raw_cv.reshape(1, -1)
            for row in raw_cv:
                cv_vals[float(row[0])] = float(row[2])

        # --- sro_vs_temp.csv ---
        sro_path = comp_txt_dir / "sro_vs_temp.csv"
        sro_rows: dict[float, dict[str, float]] = {}
        if sro_path.exists():
            # np.savetxt writes the header as a comment line: "# T_actual alphaWW ..."
            # Read column names from the first comment line, then parse data rows.
            sro_cols = []
            with open(sro_path) as _f:
                for _line in _f:
                    stripped = _line.strip()
                    if stripped.startswith("#"):
                        sro_cols = stripped.lstrip("# ").split()
                        break
            if sro_cols:
                sro_raw = pd.read_csv(sro_path, comment="#", sep=r"\s+",
                                      header=None, names=sro_cols)
                if "T_actual" in sro_raw.columns:
                    for _, sr in sro_raw.iterrows():
                        T_s = float(sr["T_actual"])
                        sro_rows[T_s] = {c: float(sr[c]) for c in sro_raw.columns
                                         if c != "T_actual"}

        # Build one row per temperature
        ref_T = T_d2 if T_d2 is not None else np.array(T_grid, dtype=float)
        for T_val in ref_T:
            row: dict = {
                "comp_id": comp_id,
                "T": T_val,
                "Ef": Ef,
                "Sf": Sf,
                "C0": C0,
                "Tm": Tm,
            }
            for e in ELEMENTS_ALL:
                row[f"x_{e}"] = comp.get(e, 0.0)

            # D* columns
            for e in ELEMENTS_ALL:
                col = f"D_{e}(m2/s)"
                if col in d2_data:
                    idx = np.argmin(np.abs(T_d2 - T_val))
                    row[f"D_{e}"] = float(d2_data[col][idx])
                else:
                    row[f"D_{e}"] = float("nan")
            if "D_total(m2/s)" in d2_data:
                idx = np.argmin(np.abs(T_d2 - T_val))
                row["D_total"] = float(d2_data["D_total(m2/s)"][idx])
            else:
                row["D_total"] = float("nan")

            # Dv columns
            for col_key, out_key in [("Dv_all(m2/s)", "Dv_all")] + \
                    [(f"Dv_{e.lower()}(m2/s)", f"Dv_{e}") for e in ELEMENTS_ALL]:
                if col_key in dv_data and T_dv is not None:
                    idx = np.argmin(np.abs(T_dv - T_val))
                    row[out_key] = float(dv_data[col_key][idx])
                else:
                    row[out_key] = float("nan")

            # Cv
            if cv_vals:
                closest = min(cv_vals.keys(), key=lambda t: abs(t - T_val))
                row["Cv"] = cv_vals[closest]
            else:
                row["Cv"] = float("nan")

            # SRO
            if sro_rows:
                closest_sro = min(sro_rows.keys(), key=lambda t: abs(t - T_val))
                row.update(sro_rows[closest_sro])

            master_rows.append(row)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(RESULTS_DIR / "master_results.csv", index=False)
    print(f"[postprocess] Saved master_results.csv  ({len(master_df)} rows)")
    return master_df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: polynomial fit
# ══════════════════════════════════════════════════════════════════════════════

def fit_polynomial(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a Ridge-regularised polynomial to log(D*_total) as a function of
    composition AND temperature together.

    Why joint fit instead of per-T slices:
    In test mode (5 compositions, each at 3 different temperatures), each
    temperature slice contains only 1 composition row — too few for a fit.
    By including normalised temperature 1/T as a 6th feature alongside the
    5 independent composition variables, all N_comp * N_temp rows are used.

    Features: x_Mo, x_Nb, x_Zr, x_Ti, x_Ta, inv_T_norm
        inv_T_norm = (1/T) / mean(1/T)   [dimensionless, centred ~1]

    The fit is stored once (not per-T) with a "T" column set to NaN to
    signal the joint model.  Coefficients are written to poly_fit_coeffs.csv.

    Returns a non-empty DataFrame so that plot_parity can proceed.
    """
    comp_feat_cols = [f"x_{e}" for e in ELEMENTS_ALL if e != "W"]
    feat_cols      = comp_feat_cols + ["inv_T_norm"]
    target = "D_total"

    sub = master_df.dropna(subset=[target])
    sub = sub[sub[target] > 0].copy()

    if len(sub) < len(feat_cols) + 1:
        print(f"  [poly] Only {len(sub)} valid rows — need at least {len(feat_cols)+1}. "
              "Skipping polynomial fit.")
        empty = pd.DataFrame(columns=["T","feature","coefficient","intercept"])
        empty.to_csv(RESULTS_DIR / "poly_fit_coeffs.csv", index=False)
        return empty

    inv_T = 1.0 / sub["T"].values
    inv_T_mean = inv_T.mean()
    sub = sub.copy()
    sub["inv_T_norm"] = inv_T / inv_T_mean   # normalise to ~1 for numerical stability

    X = sub[feat_cols].values
    y = np.log(sub[target].values)

    model = make_pipeline(
        PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
        Ridge(alpha=POLY_ALPHA),
    )
    model.fit(X, y)

    poly: PolynomialFeatures = model.named_steps["polynomialfeatures"]
    ridge: Ridge              = model.named_steps["ridge"]
    feature_names = list(poly.get_feature_names_out(feat_cols))

    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    print(f"  [poly] Joint fit  degree={POLY_DEGREE}  n={len(sub)}  R²={r2:.4f}")

    rows = []
    for fname, coef in zip(feature_names, ridge.coef_):
        rows.append({"T": float("nan"), "feature": fname,
                     "coefficient": coef, "intercept": ridge.intercept_,
                     "inv_T_mean": inv_T_mean})

    coeff_df = pd.DataFrame(rows)
    coeff_df.to_csv(RESULTS_DIR / "poly_fit_coeffs.csv", index=False)
    print(f"[postprocess] Saved poly_fit_coeffs.csv")
    return coeff_df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: comparison plots
# ══════════════════════════════════════════════════════════════════════════════

def _comp_short_label(comp_row: "pd.Series") -> str:
    """Build a short label like 'W' or 'WMoNbTa' from a compositions row."""
    return "".join(e for e in ELEMENTS_ALL if comp_row.get(f"x_{e}", 0) > 1e-4)


def _n_active(comp_row: "pd.Series") -> int:
    return sum(1 for e in ELEMENTS_ALL if comp_row.get(f"x_{e}", 0) > 1e-4)


def plot_D_vs_Tm(master_df: pd.DataFrame) -> None:
    """
    D*_total evaluated at T = 0.65 Tm vs alloy Tm.

    Reproduces Fig. 8 of Starikov 2024 (Phys. Rev. Mater. 8, 043603).
    Empirical trend: C_v^melt ∝ exp(Tm / 610 K), scaled by
    representative Dv ~ 1e-9 m²/s at Tm (Starikov 2024 §IVA).
    Marker shape encodes number of active elements.
    """
    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df  = pd.read_csv(comps_csv) if comps_csv.exists() else None

    # Empirical trend from Starikov 2024 Fig. 8
    Tm_range = np.linspace(1800, 4200, 300)
    D_emp = 4e-6 * np.exp(Tm_range / 610.0) * 1e-9
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(Tm_range, D_emp, "k--", lw=1.8,
        label=r"Empirical $C_v^{\rm melt} \propto \exp(T_m/610)$" + "\n(Starikov 2024 Fig. 8)")

    comp_ids = master_df["comp_id"].unique()
    cmap     = cm.get_cmap("tab10", max(len(comp_ids), 10))
    markers  = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v", 6: "*"}

    for i, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid]
        Tm  = sub["Tm"].iloc[0]
        sub_valid = sub[sub["D_total"] > 0].copy()
        if sub_valid.empty:
            continue

        # Point closest to T = 0.65 Tm
        idx  = (sub_valid["T"] - 0.65 * Tm).abs().idxmin()
        D_pt = sub_valid.loc[idx, "D_total"]

        # Also show Arrhenius fit line vs Tm
        if len(sub_valid) >= 2:
            from scipy.stats import linregress as _lr
            res = _lr(1.0 / sub_valid["T"].values, np.log(sub_valid["D_total"].values))
            T_fit = np.linspace(sub_valid["T"].min() * 0.9, sub_valid["T"].max() * 1.05, 80)
            ax.semilogy(np.full_like(T_fit, Tm), np.exp(res.intercept + res.slope / T_fit), color=cmap(i), lw=0.8, alpha=0.25)

        n_act = 6  # default
        label = cid
        if comps_df is not None:
            crow = comps_df[comps_df["comp_id"] == cid].iloc[0]
            n_act = _n_active(crow)
            label = f"{_comp_short_label(crow)}  ({n_act}-el)"

        ax.scatter(Tm, D_pt, color=cmap(i), marker=markers.get(n_act, "o"),
                   s=120, zorder=5, edgecolors="black", linewidths=0.6,
                   label=label)

    ax.set_xlabel(r"Alloy $T_m$  (K)", fontsize=13)
    ax.set_ylabel(r"$D^*_{\rm total}$ (m$^2$/s)", fontsize=12)
    ax.set_title(r"$T_m$–diffusivity correlation  (Starikov 2024 Fig. 8 trend)", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, ncol=2,
                  bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "comparison_D_vs_Tm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[postprocess] Saved {out}")


def plot_sro_comparison(master_df: pd.DataFrame) -> None:
    """
    Overlay α_ij vs T for all compositions, plotting ONLY pairs where
    BOTH elements i and j are active (concentration > threshold).

    Background: when element j has concentration ≈ 0, LAMMPS computes
    P_ij / c_j with c_j ≈ 0, causing α → 1.0 (numerical artefact).
    These stuck-at-1.0 lines dominate the plot for pure metals and
    binary alloys and must be excluded.

    Visual encoding:
      colour   → central element i  (fixed element-colour map)
      linestyle → composition       (each comp gets its own style)
    """
    sro_keys = [c for c in master_df.columns if c.startswith("alpha")]
    if not sro_keys:
        return

    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df  = pd.read_csv(comps_csv) if comps_csv.exists() else None

    ELEM_COLORS = {
        "W": "#1f77b4", "Mo": "#ff7f0e", "Nb": "#2ca02c",
        "Zr": "#d62728", "Ti": "#9467bd", "Ta": "#8c564b",
    }
    LINESTYLES = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), ":"]
    comp_ids = list(master_df["comp_id"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    any_plotted = False

    for ci, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid].sort_values("T")
        T   = sub["T"].values
        ls  = LINESTYLES[ci % len(LINESTYLES)]

        # Determine which elements are active for this composition
        if comps_df is not None:
            crow = comps_df[comps_df["comp_id"] == cid].iloc[0]
            active = [e for e in ELEMENTS_ALL if crow.get(f"x_{e}", 0) > 1e-4]
        else:
            # Fallback: infer from non-stuck SRO values
            active = ELEMENTS_ALL

        for a in active:
            for b in active:
                col = f"alpha{a}{b}"
                if col not in sub.columns:
                    continue
                vals = sub[col].values
                # Skip stuck-at-1.0 (absent-element artefact):
                # if all values within 0.01 of 1.0, the denominator c_j ≈ 0
                if np.all(np.abs(vals - 1.0) < 0.02):
                    continue
                color = ELEM_COLORS.get(a, "black")
                ax.plot(T, vals, color=color, ls=ls, lw=1.4, alpha=0.8)
                any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No valid SRO data\n(check sro_vs_temp.csv)",
                transform=ax.transAxes, ha="center", fontsize=12)

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5,
               label=r"$\alpha=0$  (random solid solution)")

    # Legend: colour → element, linestyle → composition
    from matplotlib.lines import Line2D
    elem_h = [
        Line2D([0], [0], color=ELEM_COLORS[e], lw=2.5, label=f"{e} (central)")
        for e in ELEMENTS_ALL
    ]
    comp_h = [
        Line2D([0], [0], color="grey", ls=LINESTYLES[i % len(LINESTYLES)], lw=2,
               label=comp_ids[i])
        for i in range(len(comp_ids))
    ]
    ax.legend(handles=elem_h + comp_h, fontsize=7.5, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)

    ax.set_xlabel("T (K)", fontsize=13)
    ax.set_ylabel(r"Warren-Cowley  $\alpha_{ij}$", fontsize=13)
    ax.set_title("SRO (active pairs only) vs temperature — all compositions", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "comparison_sro_vs_T.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[postprocess] Saved {out}")


def plot_parity(master_df: pd.DataFrame, coeff_df: pd.DataFrame) -> None:
    """
    Parity plot: polynomial-predicted log D*_total vs MD-computed log D*_total.
    Uses the joint composition+temperature model from fit_polynomial().
    """
    comp_feat_cols = [f"x_{e}" for e in ELEMENTS_ALL if e != "W"]
    feat_cols      = comp_feat_cols + ["inv_T_norm"]
    target = "D_total"

    # Guard: empty or no coefficient rows
    if coeff_df is None or coeff_df.empty or "feature" not in coeff_df.columns:
        print("[postprocess] No polynomial coefficients — skipping parity plot")
        return

    sub = master_df.dropna(subset=[target])
    sub = sub[sub[target] > 0].copy()
    if len(sub) < 3:
        print("[postprocess] Too few valid D_total rows — skipping parity plot")
        return

    # Recover inv_T_mean used during fitting
    inv_T_mean_vals = coeff_df["inv_T_mean"].dropna() if "inv_T_mean" in coeff_df.columns else None
    if inv_T_mean_vals is None or inv_T_mean_vals.empty:
        inv_T_mean = (1.0 / sub["T"].values).mean()
    else:
        inv_T_mean = float(inv_T_mean_vals.iloc[0])

    sub["inv_T_norm"] = (1.0 / sub["T"].values) / inv_T_mean
    X = sub[feat_cols].values
    y = np.log(sub[target].values)

    # Re-fit the model (sklearn models are not serialised to CSV; refit on same data)
    model = make_pipeline(
        PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
        Ridge(alpha=POLY_ALPHA),
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, s=20, alpha=0.7, color="steelblue")
    lims = [min(y.min(), y_pred.min()) - 0.5, max(y.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(r"MD  $\ln D^*_\mathrm{total}$", fontsize=11)
    ax.set_ylabel(r"Polynomial  $\ln D^*_\mathrm{total}$", fontsize=11)
    ax.set_title(f"Parity plot — degree-{POLY_DEGREE} polynomial (joint comp+T)",
                 fontsize=11)
    residuals = y_pred - y
    r2 = 1 - np.var(residuals) / np.var(y)
    ax.text(0.05, 0.95, f"$R^2$={r2:.4f}  n={len(y)}",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlim(lims); ax.set_ylim(lims)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "parity_Dtotal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[postprocess] Saved {out}")


def plot_arrhenius_summary(master_df: pd.DataFrame,
                           constants_df: pd.DataFrame) -> None:
    """
    Two-panel figure:
      Left:  Arrhenius lines D*_total vs 1/T for all compositions.
      Right: Summary table of Q, D0, R² per composition.

    Activation energy Q and pre-exponential D0 are extracted from
    linear regression of ln(D*_total) vs 1/T.
    """
    from scipy.stats import linregress
    kB = 8.617333e-5

    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df  = pd.read_csv(comps_csv) if comps_csv.exists() else None

    comp_ids = list(master_df["comp_id"].unique())
    cmap     = cm.get_cmap("tab10", max(len(comp_ids), 10))
    summary  = []

    fig, (ax_arr, ax_tbl) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    for i, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid].sort_values("T")
        T   = sub["T"].values
        D   = sub["D_total"].values
        valid = (D > 0) & ~np.isnan(D)
        if valid.sum() < 2:
            continue

        inv_T = 1.0 / T[valid]
        lnD   = np.log(D[valid])
        res   = linregress(inv_T, lnD)
        Q_eV  = -res.slope * kB
        D0    = np.exp(res.intercept)

        # Short label
        label = cid
        Tm    = constants_df[constants_df["comp_id"] == cid]["Tm"].values[0]
        Ef    = constants_df[constants_df["comp_id"] == cid]["Ef"].values[0]
        if comps_df is not None:
            crow  = comps_df[comps_df["comp_id"] == cid].iloc[0]
            label = _comp_short_label(crow)

        color = cmap(i)
        ax_arr.semilogy(inv_T, D[valid], "o", color=color, ms=7, zorder=5)
        T_fit = np.linspace(T[valid].min() * 0.95, T[valid].max() * 1.05, 100)
        ax_arr.semilogy(1.0 / T_fit,
                        np.exp(res.intercept + res.slope / T_fit),
                        "-", color=color, lw=2.0, label=label)

        summary.append([label, f"{Tm:.0f}", f"{Ef:.3f}",
                        f"{Q_eV:.3f}", f"{D0:.2e}", f"{res.rvalue**2:.4f}"])

    ax_arr.set_xlabel("1/T  (K⁻¹)", fontsize=13)
    ax_arr.set_ylabel(r"$D^*_{\rm total}$ (m$^2$/s)", fontsize=12)
    ax_arr.set_title("Arrhenius: tracer diffusion vs 1/T", fontsize=12)
    ax_arr.legend(fontsize=8, frameon=True)
    ax_arr.grid(True, which="both", ls=":", alpha=0.4)

    # Table
    ax_tbl.axis("off")
    col_labels = ["Composition", "Tm (K)", "Ef (eV)", "Q (eV)", "D₀ (m²/s)", "R²"]
    if summary:
        tbl = ax_tbl.table(cellText=summary, colLabels=col_labels,
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.8)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#2c3e50")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        for row_i in range(1, len(summary) + 1):
            bg = "#eaf2f8" if row_i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                tbl[(row_i, j)].set_facecolor(bg)
    ax_tbl.set_title("Arrhenius parameters", fontsize=12, pad=14)

    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "arrhenius_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[postprocess] Saved {out}")


def run_postprocess() -> None:
    comps   = pd.read_csv(RESULTS_DIR / "compositions.csv")
    consts  = pd.read_csv(RESULTS_DIR / "constants_all.csv")

    master  = collect_results(comps, consts)
    coeffs  = fit_polynomial(master)
    plot_D_vs_Tm(master)
    plot_sro_comparison(master)
    plot_parity(master, coeffs)
    plot_arrhenius_summary(master, consts)


if __name__ == "__main__":
    run_postprocess()

