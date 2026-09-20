"""Tests for composition sampling helpers."""

from __future__ import annotations

from config import ELEMENTS
from pipeline.compositions import _structured_vertices, active_elements


def test_structured_vertices_count() -> None:
    """All equal-molar sub-alloys of six elements number 63."""
    assert len(_structured_vertices()) == 63


def test_structured_vertices_sum_to_one() -> None:
    """Every structured vertex must be a valid point on the simplex."""
    for comp in _structured_vertices():
        assert abs(sum(comp.values()) - 1.0) < 1e-12


def test_active_elements_filters_dilute_species() -> None:
    """Elements below the tolerance must be excluded."""
    comp = {e: 0.0 for e in ELEMENTS}
    comp["W"] = 0.5
    comp["Mo"] = 0.5
    assert active_elements(comp) == ["W", "Mo"]