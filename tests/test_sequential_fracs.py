"""Tests for the sequential-fraction composition algebra."""

from __future__ import annotations

from config import ELEMENTS, SET_FRACTION_ORDER
from pipeline.lammps_common import mass_block, sequential_fracs, set_fraction_block


def _reconstruct(fracs: dict[str, float]) -> dict[str, float]:
    """Reconstruct final mole fractions from LAMMPS set fractions.

    Simulates the sequential ``set type/fraction`` process: each command
    assigns the element to a fraction ``f`` of *all* atoms, overwriting
    previously assigned types.
    """
    result = {e: 0.0 for e in ELEMENTS}
    result["W"] = 1.0   # W is the base type and is never set
    for element in SET_FRACTION_ORDER:
        f = fracs.get(element, 0.0)
        for other in ELEMENTS:
            if other == element:
                result[other] = result[other] * (1.0 - f) + f
            else:
                result[other] *= (1.0 - f)
    return result


def test_equiatomic_round_trip() -> None:
    """The set fractions must reconstruct an equiatomic composition."""
    comp = {e: 1.0 / len(ELEMENTS) for e in ELEMENTS}
    fracs = sequential_fracs(comp)
    reconstructed = _reconstruct(fracs)
    for element in ELEMENTS:
        assert abs(reconstructed[element] - comp[element]) < 1e-9


def test_arbitrary_composition_round_trip() -> None:
    """The set fractions must reconstruct an arbitrary composition."""
    comp = {"W": 0.30, "Mo": 0.20, "Nb": 0.15, "Zr": 0.10, "Ti": 0.10, "Ta": 0.15}
    fracs = sequential_fracs(comp)
    reconstructed = _reconstruct(fracs)
    for element in ELEMENTS:
        assert abs(reconstructed[element] - comp[element]) < 1e-9


def test_pure_tungsten_has_no_set_commands() -> None:
    """A pure-W composition must produce no set commands."""
    comp = {e: 0.0 for e in ELEMENTS}
    comp["W"] = 1.0
    assert set_fraction_block(comp) == "# (pure W)"


def test_mass_block_lists_all_elements() -> None:
    """The mass block must contain one command per element."""
    block = mass_block()
    for element in ELEMENTS:
        assert f"# {element}" in block
    assert block.count("mass") == len(ELEMENTS)