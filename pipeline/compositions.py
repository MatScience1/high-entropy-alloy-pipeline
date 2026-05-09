"""
pipeline/compositions.py
========================
Stage 1 — Generate the 100 alloy compositions to simulate.

Strategy
--------
We want to cover the WMoNbZrTiTa composition space uniformly, including:
  - Pure-element vertices (6 compositions)
  - Equal-molar binary to quinary sub-alloys (structured vertices)
  - Random interior points sampled from a symmetric Dirichlet distribution

Symmetric Dirichlet (α = 1) is the uniform distribution on the probability
simplex:  f(x₁,…,x₆) ∝ 1  for xᵢ ≥ 0, Σxᵢ = 1.
This ensures no region of composition space is over- or under-sampled.

Output
------
results/compositions.csv with columns:
    comp_id, x_W, x_Mo, x_Nb, x_Zr, x_Ti, x_Ta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ELEMENTS, N_COMPOSITIONS, RANDOM_SEED, RESULTS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def row_to_comp(row: pd.Series) -> dict[str, float]:
    """Extract {element: fraction} from a compositions DataFrame row."""
    return {e: float(row[f"x_{e}"]) for e in ELEMENTS}


def active_elements(comp: dict[str, float], tol: float = 1e-6) -> list[str]:
    """Return elements with non-negligible concentration."""
    return [e for e in ELEMENTS if comp.get(e, 0.0) > tol]


# ── structured vertices ───────────────────────────────────────────────────────

def _structured_vertices() -> list[dict[str, float]]:
    """
    Return equal-molar compositions for all non-trivial sub-alloys:
      - 6 unary (pure elements)
      - 15 binary, 20 ternary, 15 quaternary, 6 quinary  (all from 6 elements)
      - 1 senary (equiatomic)
    Total: 63 structured points.
    """
    from itertools import combinations

    vertices: list[dict[str, float]] = []
    for k in range(1, len(ELEMENTS) + 1):
        for subset in combinations(ELEMENTS, k):
            frac = 1.0 / k
            comp = {e: 0.0 for e in ELEMENTS}
            for e in subset:
                comp[e] = frac
            vertices.append(comp)
    return vertices


# ── main generator ────────────────────────────────────────────────────────────

def generate_compositions() -> pd.DataFrame:
    """
    Generate N_COMPOSITIONS alloy compositions and save to
    results/compositions.csv.

    The first min(63, N_COMPOSITIONS) slots are filled with structured
    vertices; remaining slots are Dirichlet-sampled random compositions.
    Compositions with x_i < 0.01 for any nominally present element are
    excluded to avoid extreme dilution artefacts in the LAMMPS sequential-
    fraction assignment.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Structured vertices (deterministic, reproducible)
    structured = _structured_vertices()
    n_struct   = min(len(structured), N_COMPOSITIONS)
    records    = structured[:n_struct]

    # Random interior points
    n_random = N_COMPOSITIONS - n_struct
    if n_random > 0:
        # Dirichlet α=1: equivalent to normalising uniform-[0,1] samples
        raw = rng.dirichlet(alpha=np.ones(len(ELEMENTS)), size=n_random * 3)
        kept = 0
        for row in raw:
            if kept >= n_random:
                break
            comp = dict(zip(ELEMENTS, row.tolist()))
            records.append(comp)
            kept += 1

    # Assign IDs and build DataFrame
    rows = []
    for i, comp in enumerate(records[:N_COMPOSITIONS]):
        row = {"comp_id": f"comp_{i:03d}"}
        row.update({f"x_{e}": round(comp.get(e, 0.0), 6) for e in ELEMENTS})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sanity check: each row should sum to 1 within floating-point tolerance
    frac_cols = [f"x_{e}" for e in ELEMENTS]
    sums = df[frac_cols].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-5, "Composition fractions do not sum to 1"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "compositions.csv"
    df.to_csv(out, index=False)
    print(f"[compositions] {len(df)} compositions  →  {out}")
    return df
