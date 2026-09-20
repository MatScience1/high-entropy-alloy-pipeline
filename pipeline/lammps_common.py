"""Shared LAMMPS input-generation helpers.

This module centralises the blocks that are common to the ADP diffusion
generator (:mod:`pipeline.lammps_diffusion`) and the GRACE coexistence
generator (:mod:`pipeline.lammps_tm`): the sequential-fraction composition
algebra, the ``mass`` block, and the ``set type/fraction`` block.  Keeping
them in one place guarantees that every generator assigns an identical
composition for a given alloy.
"""

from __future__ import annotations

from config import (
    ELEM_TYPE,
    ELEMENTS,
    MASSES,
    SET_FRACTION_ORDER,
    SET_FRACTION_SEEDS,
)

__all__ = ["mass_block", "sequential_fracs", "set_fraction_block"]


def sequential_fracs(comp: dict[str, float]) -> dict[str, float]:
    """Convert target mole fractions into LAMMPS ``set type/fraction`` values.

    LAMMPS applies ``set group all type/fraction`` sequentially, and each
    command overwrites the type of a random subset of *all* atoms, including
    atoms already assigned by an earlier command.  With the setting order
    ``Mo -> Nb -> Zr -> Ti -> Ta`` (W is the base type and is never set), the
    probability that an atom ends as element :math:`X` is

    .. math::

        P(X) = f_X \\prod_{Y \\text{ set after } X} (1 - f_Y).

    Inverting this relation in reverse setting order gives the conditional
    fractions that must be passed to LAMMPS:

    .. math::

        f_{Ta} = c_{Ta}, \\quad
        f_{Ti} = \\frac{c_{Ti}}{1 - c_{Ta}}, \\quad
        f_{Zr} = \\frac{c_{Zr}}{1 - c_{Ta} - c_{Ti}}, \\quad \\ldots

    where :math:`c_X` is the target mole fraction of :math:`X`.

    Notes
    -----
    The naive product form
    :math:`f_{Mo} = c_{Mo} / \\prod_Y (1 - c_Y)` treats the fractions as
    independent and overestimates the denominator by cross-terms of order
    :math:`c^2`, producing a 6-12% concentration error for equiatomic alloys.

    Parameters
    ----------
    comp : dict of str to float
        Target mole fractions keyed by element symbol.

    Returns
    -------
    dict of str to float
        Conditional set fractions keyed by element symbol, in reverse setting
        order (Ta first, Mo last).
    """
    fracs: dict[str, float] = {}
    claimed = 0.0
    for element in reversed(SET_FRACTION_ORDER):
        concentration = comp.get(element, 0.0)
        denominator = 1.0 - claimed
        fracs[element] = concentration / denominator if denominator > 1e-9 else 0.0
        claimed += concentration
    return fracs


def mass_block() -> str:
    """Return the LAMMPS ``mass`` commands for all six element types.

    Returns
    -------
    str
        Newline-separated ``mass`` commands, one per element.
    """
    return "\n".join(
        f"mass  {ELEM_TYPE[element]}  {MASSES[element]}   # {element}"
        for element in ELEMENTS
    )


def set_fraction_block(comp: dict[str, float]) -> str:
    """Return the sequential ``set type/fraction`` block for a composition.

    The commands are emitted in :data:`config.SET_FRACTION_ORDER` (Mo first,
    Ta last) with the fixed seeds from :data:`config.SET_FRACTION_SEEDS`, so
    the assignment is deterministic and reproducible.

    Parameters
    ----------
    comp : dict of str to float
        Target mole fractions keyed by element symbol.

    Returns
    -------
    str
        Newline-separated ``set`` commands, or a comment for pure W.
    """
    fracs = sequential_fracs(comp)
    lines: list[str] = []
    for element in SET_FRACTION_ORDER:
        fraction = fracs.get(element, 0.0)
        if fraction > 1e-9:
            lines.append(
                f"set group all type/fraction {ELEM_TYPE[element]} "
                f"{fraction:.8f} {SET_FRACTION_SEEDS[element]}   "
                f"# {element}: target {comp.get(element, 0.0):.4f}"
            )
    return "\n".join(lines) if lines else "# (pure W)"