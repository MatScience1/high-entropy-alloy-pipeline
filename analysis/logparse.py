"""Shared helpers for parsing LAMMPS text output.

LAMMPS logs and ``fix ave/time`` files are plain text and may be truncated by
a walltime kill, a crashed run, or an interrupted write.  The helpers here
provide a single exception type and a safe line reader so that every parser in
:mod:`analysis` fails gracefully instead of aborting the pipeline.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["LogParseError", "read_lines"]


class LogParseError(RuntimeError):
    """Raised when a LAMMPS output file cannot be parsed into usable data."""


def read_lines(path: Path) -> list[str]:
    """Read a text file into a list of lines.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    list of str
        File contents split into lines.

    Raises
    ------
    LogParseError
        If the file does not exist or cannot be read.
    """
    if not path.exists():
        raise LogParseError(f"file not found: {path}")
    try:
        with open(path) as fh:
            return fh.readlines()
    except OSError as exc:
        raise LogParseError(f"cannot read {path}: {exc}") from exc