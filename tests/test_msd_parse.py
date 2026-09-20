"""Tests for graceful LAMMPS MSD log parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.logparse import LogParseError
from analysis.msd import parse_log

_HEADER = "Step Temp c_msd_all[1] c_msd_all[4] c_msd_w[1] c_msd_w[4]\n"


def _write_log(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "log.lammps"
    path.write_text(body)
    return path


def test_parse_valid_log(tmp_path: Path) -> None:
    """A well-formed log must parse into the expected arrays."""
    body = (
        "LAMMPS preamble\n"
        + _HEADER
        + "0 300 0 0 0 0\n"
        + "100 300 1 2 0.5 1.0\n"
        + "Loop time of 1.0\n"
    )
    data = parse_log(_write_log(tmp_path, body))
    assert data["Step"].tolist() == [0, 100]
    assert data["c_msd_all[4]"].tolist() == [0.0, 2.0]


def test_parse_skips_truncated_row(tmp_path: Path) -> None:
    """A truncated row must be skipped without aborting the parse."""
    body = (
        _HEADER
        + "0 300 0 0 0 0\n"
        + "100 300 1 2\n"          # truncated: too few columns
        + "200 300 2 4 1 2\n"
    )
    data = parse_log(_write_log(tmp_path, body))
    assert data["Step"].tolist() == [0, 200]


def test_missing_header_raises(tmp_path: Path) -> None:
    """A log without an MSD header must raise ``LogParseError``."""
    with pytest.raises(LogParseError):
        parse_log(_write_log(tmp_path, "no header here\n"))


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing log file must raise ``LogParseError``."""
    with pytest.raises(LogParseError):
        parse_log(tmp_path / "does_not_exist.lammps")


def test_header_without_data_raises(tmp_path: Path) -> None:
    """A header with no data rows must raise ``LogParseError``."""
    with pytest.raises(LogParseError):
        parse_log(_write_log(tmp_path, _HEADER + "Loop time of 1.0\n"))