"""Stage 1 - generate the alloy compositions to simulate.

Strategy
--------
The W-Mo-Nb-Zr-Ti-Ta composition space is a 5-dimensional simplex.  It is
covered with:

* Structured vertices: every equal-molar sub-alloy (unary through senary),
  63 deterministic compositions that pin the boundaries and edges.
* Random interior points: a symmetric Dirichlet distribution with
  concentration parameter :math:`\\alpha = 1`, which is the uniform
  distribution on the probability simplex,

  .. math::

      f(x_1, \\ldots, x_6) \\propto \\prod_i x_i^{\\alpha - 1} = \\text{const},
      \\qquad x_i \\ge 0, \\; \\sum_i x_i = 1.

  This guarantees that no region of composition space is over- or
  under-sampled.

Output
------
``results/compositions.csv`` with columns
``comp_id, x_W, x_Mo, x_Nb, x_Zr, x_Ti, x_Ta``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import config
from config import ELEMENTS, RANDOM_SEED, RESULTS_DIR
from logging_config import get_logger

logger = get_logger(__name__)


def row_to_comp(row: pd.Series) -> dict[str, float]:
    """Extract ``{element: fraction}`` from a compositions DataFrame row.

    Parameters
    ----------
    row : pandas.Series
        A row of ``compositions.csv`` containing ``x_<element>`` columns.

    Returns
    -------
    dict of str to float
        Mole fractions keyed by element symbol.
    """
    return {e: float(row[f"x_{e}"]) for e in ELEMENTS}


def active_elements(comp: dict[str, float], tol: float = 1e-6) -> list[str]:
    """Return the elements with non-negligible concentration.

    Parameters
    ----------
    comp : dict of str to float
        Mole fractions keyed by element symbol.
    tol : float, optional
        Concentration threshold below which an element is considered absent.

    Returns
    -------
    list of str
        Element symbols with concentration greater than ``tol``.
    """
    return [e for e in ELEMENTS if comp.get(e, 0.0) > tol]


def _structured_vertices() -> list[dict[str, float]]:
    """Return equal-molar compositions for all non-trivial sub-alloys.

    Returns
    -------
    list of dict
        Six unary, 15 binary, 20 ternary, 15 quaternary, six quinary, and one
        senary composition: 63 structured points in total.
    """
    vertices: list[dict[str, float]] = []
    for k in range(1, len(ELEMENTS) + 1):
        for subset in combinations(ELEMENTS, k):
            fraction = 1.0 / k
            comp = {e: 0.0 for e in ELEMENTS}
            for e in subset:
                comp[e] = fraction
            vertices.append(comp)
    return vertices


def generate_compositions() -> pd.DataFrame:
    """Generate the alloy compositions and write ``results/compositions.csv``.

    The first ``min(63, N_COMPOSITIONS)`` slots are filled with structured
    vertices; the remaining slots are Dirichlet-sampled interior points.

    Returns
    -------
    pandas.DataFrame
        One row per composition with columns ``comp_id`` and ``x_<element>``.

    Raises
    ------
    ValueError
        If any composition's mole fractions do not sum to unity within
        floating-point tolerance.
    """
    n_compositions = int(config.N_COMPOSITIONS)
    rng = np.random.default_rng(RANDOM_SEED)

    structured = _structured_vertices()
    n_struct = min(len(structured), n_compositions)
    records: list[dict[str, float]] = structured[:n_struct]

    n_random = n_compositions - n_struct
    if n_random > 0:
        # Dirichlet(alpha=1) is the uniform distribution on the simplex.
        raw = rng.dirichlet(alpha=np.ones(len(ELEMENTS)), size=n_random)
        for row in raw:
            records.append(dict(zip(ELEMENTS, row.tolist())))

    rows: list[dict[str, object]] = []
    for i, comp in enumerate(records[:n_compositions]):
        row: dict[str, object] = {"comp_id": f"comp_{i:03d}"}
        row.update({f"x_{e}": round(comp.get(e, 0.0), 6) for e in ELEMENTS})
        rows.append(row)

    df = pd.DataFrame(rows)

    frac_cols = [f"x_{e}" for e in ELEMENTS]
    sums = df[frac_cols].sum(axis=1)
    if (sums - 1.0).abs().max() >= 1e-5:
        raise ValueError("Composition fractions do not sum to 1")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "compositions.csv"
    df.to_csv(out, index=False)
    logger.info("Generated %d compositions -> %s", len(df), out)
    return df
