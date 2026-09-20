"""Tests for the runtime test/production mode switch."""

from __future__ import annotations

import config


def test_production_mode_values() -> None:
    """Production mode must use the full sampling and run lengths."""
    config.set_test_mode(False)
    assert config.TEST_MODE is False
    assert config.N_COMPOSITIONS == 100
    assert config.N_TEMPS == 10
    assert config.POLY_DEGREE == 2
    assert config.COEX_N_TGUESS == 5


def test_test_mode_values() -> None:
    """Test mode must use the reduced sampling and run lengths."""
    config.set_test_mode(True)
    assert config.TEST_MODE is True
    assert config.N_COMPOSITIONS == 5
    assert config.N_TEMPS == 3
    assert config.POLY_DEGREE == 1
    assert config.COEX_N_TGUESS == 3
    config.set_test_mode(False)


def test_physical_constants_are_mode_independent() -> None:
    """Physical constants must not change with the mode."""
    config.set_test_mode(False)
    kb_prod = config.KB_EV
    elements_prod = list(config.ELEMENTS)
    config.set_test_mode(True)
    assert config.KB_EV == kb_prod
    assert config.ELEMENTS == elements_prod
    config.set_test_mode(False)