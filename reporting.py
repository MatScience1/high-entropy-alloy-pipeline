#!/usr/bin/env python3
"""Tabular and numeric reporting helpers.

The pipeline reports numerical results (composition tables, diffusion
coefficients, SRO matrices, Arrhenius parameters) as aligned plain-text
tables.  Rendering is delegated to :mod:`pandas` so that no additional
runtime dependency is required.

All helpers return strings; callers decide whether to log them or write them
to a file.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "format_dataframe",
    "format_float",
    "format_mapping",
    "log_dataframe",
    "log_mapping",
]


def format_float(
    value: float,
    precision: int = 4,
    *,
    scientific: bool = False,
) -> str:
    """Format a scalar for tabular display.

    Parameters
    ----------
    value : float
        Value to format.  ``NaN`` and infinities are rendered as ``"n/a"``.
    precision : int, optional
        Number of significant digits (scientific) or decimal places (fixed).
    scientific : bool, optional
        Use scientific notation instead of fixed point.

    Returns
    -------
    str
        Human-readable representation.
    """
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(fvalue):
        return "n/a"
    if scientific:
        return f"{fvalue:.{precision}e}"
    return f"{fvalue:.{precision}f}"


def format_dataframe(
    frame: pd.DataFrame,
    *,
    float_format: str = "%.6g",
    index: bool = False,
) -> str:
    """Render a DataFrame as an aligned plain-text table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to render.
    float_format : str, optional
        printf-style format applied to floating-point columns.
    index : bool, optional
        Include the DataFrame index in the output.

    Returns
    -------
    str
        Aligned table, or ``"(no rows)"`` for an empty frame.
    """
    if frame is None or frame.empty:
        return "(no rows)"
    return frame.to_string(index=index, float_format=float_format)


def format_mapping(
    mapping: Mapping[str, float],
    *,
    key_header: str = "key",
    value_header: str = "value",
    precision: int = 6,
) -> str:
    """Render a scalar mapping as a two-column aligned table.

    Parameters
    ----------
    mapping : Mapping[str, float]
        Mapping of labels to scalar values.
    key_header, value_header : str, optional
        Column headers for the rendered table.
    precision : int, optional
        Decimal places used for the value column.

    Returns
    -------
    str
        Aligned two-column table, or ``"(no entries)"`` when empty.
    """
    if not mapping:
        return "(no entries)"
    frame = pd.DataFrame(
        {
            key_header: list(mapping.keys()),
            value_header: [format_float(v, precision) for v in mapping.values()],
        }
    )
    return format_dataframe(frame)


def log_dataframe(
    logger: logging.Logger,
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    level: int = logging.INFO,
    float_format: str = "%.6g",
) -> None:
    """Log a DataFrame as an aligned table at the given level.

    Parameters
    ----------
    logger : logging.Logger
        Destination logger.
    frame : pandas.DataFrame
        Table to render.
    title : str, optional
        Optional heading logged before the table.
    level : int, optional
        Logging level for every emitted line.
    float_format : str, optional
        printf-style format applied to floating-point columns.
    """
    if title:
        logger.log(level, title)
    for line in format_dataframe(frame, float_format=float_format).splitlines():
        logger.log(level, line)


def log_mapping(
    logger: logging.Logger,
    mapping: Mapping[str, float],
    *,
    title: str | None = None,
    level: int = logging.INFO,
    precision: int = 6,
) -> None:
    """Log a scalar mapping as an aligned two-column table.

    Parameters
    ----------
    logger : logging.Logger
        Destination logger.
    mapping : Mapping[str, float]
        Mapping of labels to scalar values.
    title : str, optional
        Optional heading logged before the table.
    level : int, optional
        Logging level for every emitted line.
    precision : int, optional
        Decimal places used for the value column.
    """
    if title:
        logger.log(level, title)
    for line in format_mapping(mapping, precision=precision).splitlines():
        logger.log(level, line)