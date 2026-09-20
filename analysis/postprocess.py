"""Stage 10 - aggregate results, polynomial fit, and comparison plots.

Interface (called by ``run_pipeline.py``)
-----------------------------------------
``run_postprocess()``

What this module does
---------------------
1. Read ``results/<comp_id>/txt/D2_components.txt`` for all compositions.
2. Build a feature matrix from the mole fractions (six elements on the
   composition simplex give five independent coordinates).
3. Fit a Ridge-regularised polynomial of degree ``POLY_DEGREE`` to
   ``ln(D*_total)`` as a joint function of composition and inverse
   temperature.
4. Produce comparison plots: D* versus Tm, SRO versus temperature, a parity
   plot, and an Arrhenius summary.

Output
------
``results/master_results.csv``
``results/poly_fit_coeffs.csv``
``results/plots/comparison_D_vs_Tm.png``
``results/plots/comparison_sro_vs_T.png``
``results/plots/parity_Dtotal.png``
``results/plots/arrhenius_summary.png``
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from analysis.arrhenius import fit_arrhenius_wls
from config import ELEMENTS, POLY_ALPHA, POLY_DEGREE, RESULTS_DIR
from logging_config import configure_logging, get_logger
from reporting import log_dataframe

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ELEMENTS_ALL: list[str] = ELEMENTS


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: Arrhenius summary for a diffusion array
# ══════════════════════════════════════════════════════════════════════════════

def arrhenius_params(T_arr: np.ndarray, D_arr: np.ndarray) -> dict[str, float]:
    """Return ``{D0, Q_eV, R2}`` from a linear fit of ``ln(D)`` versus ``1/T``.

    Parameters
    ----------
    T_arr : numpy.ndarray
        Temperatures [K].
    D_arr : numpy.ndarray
        Diffusivities.

    Returns
    -------
    dict of str to float
        Pre-exponential factor, activation energy [eV], and R^2.  Values are
        ``NaN`` when fewer than three valid points are available.
    """
    mask = (D_arr > 0) & np.isfinite(D_arr) & (T_arr > 0)
    if mask.sum() < 3:
        return {"D0": float("nan"), "Q_eV": float("nan"), "R2": float("nan")}
    fit = fit_arrhenius_wls(1.0 / T_arr[mask], np.log(D_arr[mask]))
    if fit is None:
        return {"D0": float("nan"), "Q_eV": float("nan"), "R2": float("nan")}
    return {"D0": fit.D0, "Q_eV": fit.Q_eV, "R2": fit.r2}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1-3: collect data
# ══════════════════════════════════════════════════════════════════════════════

def collect_results(
    compositions_df: pd.DataFrame,
    constants_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the master results table from per-composition result files.

    Parameters
    ----------
    compositions_df : pandas.DataFrame
        Compositions table.
    constants_df : pandas.DataFrame
        Constants table.

    Returns
    -------
    pandas.DataFrame
        One row per (composition, temperature).  Written to
        ``results/master_results.csv``.
    """
    master_rows: list[dict[str, object]] = []

    for _, crow in compositions_df.iterrows():
        comp_id = str(crow["comp_id"])
        comp = {e: float(crow[f"x_{e}"]) for e in ELEMENTS_ALL}

        const_row_df = constants_df[constants_df["comp_id"] == comp_id]
        if const_row_df.empty:
            logger.warning("no constants for %s", comp_id)
            continue
        cst = const_row_df.iloc[0]
        ef = float(cst["Ef"])
        sf = float(cst["Sf"])
        c0 = float(cst["C0"])
        tm = float(cst["Tm"])

        t_grid_raw = cst["T_grid"]
        try:
            t_grid = (
                json.loads(t_grid_raw)
                if isinstance(t_grid_raw, str)
                else list(t_grid_raw)
            )
        except (TypeError, ValueError):
            continue

        comp_txt_dir = RESULTS_DIR / comp_id / "txt"

        # --- D2_components.txt ---
        d2_path = comp_txt_dir / "D2_components.txt"
        d2_data: dict[str, np.ndarray] = {}
        t_d2: np.ndarray | None = None
        if d2_path.exists():
            with open(d2_path) as fh:
                hdr = fh.readline().lstrip("#").strip()
            col_names = hdr.split()
            raw = np.genfromtxt(d2_path, skip_header=1)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            t_d2 = raw[:, 0]
            for j, cn in enumerate(col_names):
                d2_data[cn] = raw[:, j]

        # --- Dv.txt ---
        dv_path = comp_txt_dir / "Dv.txt"
        dv_data: dict[str, np.ndarray] = {}
        t_dv: np.ndarray | None = None
        if dv_path.exists():
            with open(dv_path) as fh:
                hdr = fh.readline().lstrip("#").strip()
            col_names_dv = hdr.split()
            raw_dv = np.genfromtxt(dv_path, skip_header=1)
            if raw_dv.ndim == 1:
                raw_dv = raw_dv.reshape(1, -1)
            t_dv = raw_dv[:, 0]
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
            sro_cols: list[str] = []
            with open(sro_path) as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        sro_cols = stripped.lstrip("# ").split()
                        break
            if sro_cols:
                sro_raw = pd.read_csv(sro_path, comment="#", sep=r"\s+",
                                      header=None, names=sro_cols)
                if "T_actual" in sro_raw.columns:
                    for _, sr in sro_raw.iterrows():
                        t_s = float(sr["T_actual"])
                        sro_rows[t_s] = {
                            c: float(sr[c])
                            for c in sro_raw.columns if c != "T_actual"
                        }

        ref_t = t_d2 if t_d2 is not None else np.array(t_grid, dtype=float)
        for t_val in ref_t:
            row: dict[str, object] = {
                "comp_id": comp_id,
                "T": t_val,
                "Ef": ef,
                "Sf": sf,
                "C0": c0,
                "Tm": tm,
            }
            for element in ELEMENTS_ALL:
                row[f"x_{element}"] = comp.get(element, 0.0)

            for element in ELEMENTS_ALL:
                col = f"D_{element}(m2/s)"
                if col in d2_data and t_d2 is not None:
                    idx = int(np.argmin(np.abs(t_d2 - t_val)))
                    row[f"D_{element}"] = float(d2_data[col][idx])
                else:
                    row[f"D_{element}"] = float("nan")
            if "D_total(m2/s)" in d2_data and t_d2 is not None:
                idx = int(np.argmin(np.abs(t_d2 - t_val)))
                row["D_total"] = float(d2_data["D_total(m2/s)"][idx])
            else:
                row["D_total"] = float("nan")

            for col_key, out_key in [("Dv_all(m2/s)", "Dv_all")] + [
                (f"Dv_{e.lower()}(m2/s)", f"Dv_{e}") for e in ELEMENTS_ALL
            ]:
                if col_key in dv_data and t_dv is not None:
                    idx = int(np.argmin(np.abs(t_dv - t_val)))
                    row[out_key] = float(dv_data[col_key][idx])
                else:
                    row[out_key] = float("nan")

            if cv_vals:
                closest = min(cv_vals.keys(), key=lambda t: abs(t - t_val))
                row["Cv"] = cv_vals[closest]
            else:
                row["Cv"] = float("nan")

            if sro_rows:
                closest_sro = min(sro_rows.keys(), key=lambda t: abs(t - t_val))
                row.update(sro_rows[closest_sro])

            master_rows.append(row)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(RESULTS_DIR / "master_results.csv", index=False)
    logger.info("Saved master_results.csv (%d rows)", len(master_df))
    return master_df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: polynomial fit
# ══════════════════════════════════════════════════════════════════════════════

def fit_polynomial(master_df: pd.DataFrame) -> pd.DataFrame:
    """Fit a Ridge-regularised polynomial to ``ln(D*_total)``.

    The model is a joint function of composition and temperature.  Including
    the normalised inverse temperature ``1/T`` as a sixth feature alongside
    the five independent composition variables lets every
    ``N_comp * N_temp`` row contribute to a single fit, which is essential in
    test mode where each temperature slice holds too few compositions.

    Features: ``x_Mo, x_Nb, x_Zr, x_Ti, x_Ta, inv_T_norm`` with
    ``inv_T_norm = (1/T) / mean(1/T)``.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of :func:`collect_results`.

    Returns
    -------
    pandas.DataFrame
        Coefficient table written to ``results/poly_fit_coeffs.csv``.
    """
    comp_feat_cols = [f"x_{e}" for e in ELEMENTS_ALL if e != "W"]
    feat_cols = comp_feat_cols + ["inv_T_norm"]
    target = "D_total"

    sub = master_df.dropna(subset=[target])
    sub = sub[sub[target] > 0].copy()

    if len(sub) < len(feat_cols) + 1:
        logger.warning(
            "Only %d valid rows - need at least %d. Skipping polynomial fit.",
            len(sub), len(feat_cols) + 1,
        )
        empty = pd.DataFrame(columns=["T", "feature", "coefficient", "intercept"])
        empty.to_csv(RESULTS_DIR / "poly_fit_coeffs.csv", index=False)
        return empty

    inv_t = 1.0 / sub["T"].values
    inv_t_mean = inv_t.mean()
    sub["inv_T_norm"] = inv_t / inv_t_mean   # normalise to ~1 for stability

    x = sub[feat_cols].values
    y = np.log(sub[target].values)

    model = make_pipeline(
        PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
        Ridge(alpha=POLY_ALPHA),
    )
    model.fit(x, y)

    poly: PolynomialFeatures = model.named_steps["polynomialfeatures"]
    ridge: Ridge = model.named_steps["ridge"]
    feature_names = list(poly.get_feature_names_out(feat_cols))

    y_pred = model.predict(x)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    logger.info("Joint fit degree=%d n=%d R2=%.4f", POLY_DEGREE, len(sub), r2)

    rows = [
        {
            "T": float("nan"),
            "feature": fname,
            "coefficient": coef,
            "intercept": ridge.intercept_,
            "inv_T_mean": inv_t_mean,
        }
        for fname, coef in zip(feature_names, ridge.coef_)
    ]

    coeff_df = pd.DataFrame(rows)
    coeff_df.to_csv(RESULTS_DIR / "poly_fit_coeffs.csv", index=False)
    logger.info("Saved poly_fit_coeffs.csv")
    return coeff_df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: comparison plots
# ══════════════════════════════════════════════════════════════════════════════

def _comp_short_label(comp_row: pd.Series) -> str:
    """Return a short label such as ``W`` or ``WMoNbTa`` from a row."""
    return "".join(e for e in ELEMENTS_ALL if comp_row.get(f"x_{e}", 0) > 1e-4)


def _n_active(comp_row: pd.Series) -> int:
    """Return the number of elements with non-negligible concentration."""
    return sum(1 for e in ELEMENTS_ALL if comp_row.get(f"x_{e}", 0) > 1e-4)


def plot_D_vs_Tm(master_df: pd.DataFrame) -> None:
    """Plot D*_total at ``T = 0.65 Tm`` against the alloy Tm.

    Reproduces Fig. 8 of Starikov 2024 (Phys. Rev. Mater. 8, 043603).  The
    empirical trend is ``C_v^melt ~ exp(Tm / 610 K)`` scaled by a
    representative ``Dv ~ 1e-9 m^2/s`` at Tm.  Marker shape encodes the number
    of active elements.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of :func:`collect_results`.
    """
    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df = pd.read_csv(comps_csv) if comps_csv.exists() else None

    tm_range = np.linspace(1800, 4200, 300)
    d_emp = 4e-6 * np.exp(tm_range / 610.0) * 1e-9
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(tm_range, d_emp, "k--", lw=1.8,
                label=r"Empirical $C_v^{\rm melt} \propto \exp(T_m/610)$"
                      + "\n(Starikov 2024 Fig. 8)")

    comp_ids = master_df["comp_id"].unique()
    cmap = plt.get_cmap("tab10", max(len(comp_ids), 10))
    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v", 6: "*"}

    for i, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid]
        tm = sub["Tm"].iloc[0]
        sub_valid = sub[sub["D_total"] > 0].copy()
        if sub_valid.empty:
            continue

        idx = (sub_valid["T"] - 0.65 * tm).abs().idxmin()
        d_pt = sub_valid.loc[idx, "D_total"]

        if len(sub_valid) >= 2:
            fit = fit_arrhenius_wls(
                1.0 / sub_valid["T"].values,
                np.log(sub_valid["D_total"].values),
            )
            if fit is not None:
                t_fit = np.linspace(
                    sub_valid["T"].min() * 0.9, sub_valid["T"].max() * 1.05, 80
                )
                ax.semilogy(
                    np.full_like(t_fit, tm),
                    np.exp(fit.ln_D0 + fit.slope / t_fit),
                    color=cmap(i), lw=0.8, alpha=0.25,
                )

        n_act = 6
        label = cid
        if comps_df is not None:
            crow = comps_df[comps_df["comp_id"] == cid].iloc[0]
            n_act = _n_active(crow)
            label = f"{_comp_short_label(crow)}  ({n_act}-el)"

        ax.scatter(tm, d_pt, color=cmap(i), marker=markers.get(n_act, "o"),
                   s=120, zorder=5, edgecolors="black", linewidths=0.6,
                   label=label)

    ax.set_xlabel(r"Alloy $T_m$  (K)", fontsize=13)
    ax.set_ylabel(r"$D^*_{\rm total}$ (m$^2$/s)", fontsize=12)
    ax.set_title(r"$T_m$-diffusivity correlation  (Starikov 2024 Fig. 8 trend)",
                 fontsize=12)
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
    logger.info("Saved %s", out)


def plot_sro_comparison(master_df: pd.DataFrame) -> None:
    """Overlay ``alpha_ij`` versus T for all compositions.

    Only pairs where both elements are active are plotted.  When element
    ``j`` has concentration near zero, LAMMPS evaluates ``P_ij / c_j`` with
    ``c_j ~ 0``, driving ``alpha`` to 1.0; these numerical artefacts are
    excluded.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of :func:`collect_results`.
    """
    sro_keys = [c for c in master_df.columns if c.startswith("alpha")]
    if not sro_keys:
        return

    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df = pd.read_csv(comps_csv) if comps_csv.exists() else None

    elem_colors = {
        "W": "#1f77b4", "Mo": "#ff7f0e", "Nb": "#2ca02c",
        "Zr": "#d62728", "Ti": "#9467bd", "Ta": "#8c564b",
    }
    linestyles = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (5, 2)), ":"]
    comp_ids = list(master_df["comp_id"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    any_plotted = False

    for ci, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid].sort_values("T")
        t_vals = sub["T"].values
        ls = linestyles[ci % len(linestyles)]

        if comps_df is not None:
            crow = comps_df[comps_df["comp_id"] == cid].iloc[0]
            active = [e for e in ELEMENTS_ALL if crow.get(f"x_{e}", 0) > 1e-4]
        else:
            active = ELEMENTS_ALL

        for a in active:
            for b in active:
                col = f"alpha{a}{b}"
                if col not in sub.columns:
                    continue
                vals = sub[col].values
                if np.all(np.abs(vals - 1.0) < 0.02):
                    continue
                ax.plot(t_vals, vals, color=elem_colors.get(a, "black"),
                        ls=ls, lw=1.4, alpha=0.8)
                any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No valid SRO data\n(check sro_vs_temp.csv)",
                transform=ax.transAxes, ha="center", fontsize=12)

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5,
               label=r"$\alpha=0$  (random solid solution)")

    from matplotlib.lines import Line2D
    elem_h = [
        Line2D([0], [0], color=elem_colors[e], lw=2.5, label=f"{e} (central)")
        for e in ELEMENTS_ALL
    ]
    comp_h = [
        Line2D([0], [0], color="grey", ls=linestyles[i % len(linestyles)], lw=2,
               label=comp_ids[i])
        for i in range(len(comp_ids))
    ]
    ax.legend(handles=elem_h + comp_h, fontsize=7.5, ncol=2,
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)

    ax.set_xlabel("T (K)", fontsize=13)
    ax.set_ylabel(r"Warren-Cowley  $\alpha_{ij}$", fontsize=13)
    ax.set_title("SRO (active pairs only) vs temperature - all compositions",
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "comparison_sro_vs_T.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_parity(master_df: pd.DataFrame, coeff_df: pd.DataFrame) -> None:
    """Plot polynomial-predicted versus MD-computed ``ln D*_total``.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of :func:`collect_results`.
    coeff_df : pandas.DataFrame
        Output of :func:`fit_polynomial`.
    """
    comp_feat_cols = [f"x_{e}" for e in ELEMENTS_ALL if e != "W"]
    feat_cols = comp_feat_cols + ["inv_T_norm"]
    target = "D_total"

    if coeff_df is None or coeff_df.empty or "feature" not in coeff_df.columns:
        logger.warning("no polynomial coefficients - skipping parity plot")
        return

    sub = master_df.dropna(subset=[target])
    sub = sub[sub[target] > 0].copy()
    if len(sub) < 3:
        logger.warning("too few valid D_total rows - skipping parity plot")
        return

    inv_t_mean_vals = (
        coeff_df["inv_T_mean"].dropna()
        if "inv_T_mean" in coeff_df.columns else None
    )
    if inv_t_mean_vals is None or inv_t_mean_vals.empty:
        inv_t_mean = (1.0 / sub["T"].values).mean()
    else:
        inv_t_mean = float(inv_t_mean_vals.iloc[0])

    sub["inv_T_norm"] = (1.0 / sub["T"].values) / inv_t_mean
    x = sub[feat_cols].values
    y = np.log(sub[target].values)

    model = make_pipeline(
        PolynomialFeatures(degree=POLY_DEGREE, include_bias=False),
        Ridge(alpha=POLY_ALPHA),
    )
    model.fit(x, y)
    y_pred = model.predict(x)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, s=20, alpha=0.7, color="steelblue")
    lims = [min(y.min(), y_pred.min()) - 0.5, max(y.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(r"MD  $\ln D^*_\mathrm{total}$", fontsize=11)
    ax.set_ylabel(r"Polynomial  $\ln D^*_\mathrm{total}$", fontsize=11)
    ax.set_title(f"Parity plot - degree-{POLY_DEGREE} polynomial (joint comp+T)",
                 fontsize=11)
    residuals = y_pred - y
    r2 = 1 - np.var(residuals) / np.var(y)
    ax.text(0.05, 0.95, f"$R^2$={r2:.4f}  n={len(y)}",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    out = RESULTS_DIR / "plots" / "parity_Dtotal.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_arrhenius_summary(
    master_df: pd.DataFrame,
    constants_df: pd.DataFrame,
) -> None:
    """Produce a two-panel Arrhenius figure with a parameter table.

    The left panel shows ``D*_total`` versus ``1/T`` for all compositions; the
    right panel tabulates Q, D0, and R^2 per composition.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Output of :func:`collect_results`.
    constants_df : pandas.DataFrame
        Constants table.
    """
    comps_csv = RESULTS_DIR / "compositions.csv"
    comps_df = pd.read_csv(comps_csv) if comps_csv.exists() else None

    comp_ids = list(master_df["comp_id"].unique())
    cmap = plt.get_cmap("tab10", max(len(comp_ids), 10))
    summary: list[list[str]] = []

    fig, (ax_arr, ax_tbl) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]}
    )

    for i, cid in enumerate(comp_ids):
        sub = master_df[master_df["comp_id"] == cid].sort_values("T")
        t_vals = sub["T"].values
        d_vals = sub["D_total"].values
        valid = (d_vals > 0) & np.isfinite(d_vals)
        if valid.sum() < 2:
            continue

        fit = fit_arrhenius_wls(1.0 / t_vals[valid], np.log(d_vals[valid]))
        if fit is None:
            continue

        label = cid
        tm = constants_df[constants_df["comp_id"] == cid]["Tm"].values[0]
        ef = constants_df[constants_df["comp_id"] == cid]["Ef"].values[0]
        if comps_df is not None:
            crow = comps_df[comps_df["comp_id"] == cid].iloc[0]
            label = _comp_short_label(crow)

        color = cmap(i)
        ax_arr.semilogy(1.0 / t_vals[valid], d_vals[valid], "o",
                        color=color, ms=7, zorder=5)
        t_fit = np.linspace(t_vals[valid].min() * 0.95,
                            t_vals[valid].max() * 1.05, 100)
        ax_arr.semilogy(1.0 / t_fit,
                        np.exp(fit.ln_D0 + fit.slope / t_fit),
                        "-", color=color, lw=2.0, label=label)

        summary.append([label, f"{tm:.0f}", f"{ef:.3f}",
                        f"{fit.Q_eV:.3f}", f"{fit.D0:.2e}", f"{fit.r2:.4f}"])

    ax_arr.set_xlabel("1/T  (K^-1)", fontsize=13)
    ax_arr.set_ylabel(r"$D^*_{\rm total}$ (m$^2$/s)", fontsize=12)
    ax_arr.set_title("Arrhenius: tracer diffusion vs 1/T", fontsize=12)
    ax_arr.legend(fontsize=8, frameon=True)
    ax_arr.grid(True, which="both", ls=":", alpha=0.4)

    ax_tbl.axis("off")
    col_labels = ["Composition", "Tm (K)", "Ef (eV)", "Q (eV)", "D0 (m2/s)", "R2"]
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
    logger.info("Saved %s", out)


def run_postprocess() -> None:
    """Run the full Stage 10 post-processing chain."""
    comps = pd.read_csv(RESULTS_DIR / "compositions.csv")
    consts = pd.read_csv(RESULTS_DIR / "constants_all.csv")

    master = collect_results(comps, consts)
    coeffs = fit_polynomial(master)
    plot_D_vs_Tm(master)
    plot_sro_comparison(master)
    plot_parity(master, coeffs)
    plot_arrhenius_summary(master, consts)


def main() -> None:
    """Command-line entry point for post-processing."""
    configure_logging()
    run_postprocess()


if __name__ == "__main__":
    main()
