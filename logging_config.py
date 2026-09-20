#!/usr/bin/env python3
"""Central logging configuration for the WMoNbZrTiTa diffusion pipeline.

Every module acquires its logger through :func:`get_logger`.  The pipeline
root logger is configured exactly once by :func:`configure_logging`, which is
called by the orchestrator and by each standalone module CLI.

Log-level policy
----------------
``INFO``
    Normal progress: stage banners, per-composition progress, written files.
``WARNING``
    Recoverable problems: a truncated LAMMPS log, a skipped run, or a fallback
    value substituted for a failed calculation.
``ERROR``
    Unrecoverable failure of a single unit of work (one composition, one
    temperature) that does not abort the whole pipeline.
``DEBUG``
    Per-file parser diagnostics; disabled by default.

The record format is terse and greppable::

    2026-09-20 16:55:39 | INFO     | hea_pipeline.pipeline.constants | message
"""

from __future__ import annotations

import logging
import sys

__all__ = ["ROOT_LOGGER_NAME", "configure_logging", "get_logger"]

ROOT_LOGGER_NAME: str = "hea_pipeline"
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configure the pipeline root logger and return it.

    Parameters
    ----------
    level : int or str, optional
        Logging threshold.  Accepts a :mod:`logging` level constant or a
        case-insensitive level name such as ``"INFO"`` or ``"DEBUG"``.
    force : bool, optional
        Reconfigure the logger even if it was already configured.  Used by
        tests and by CLIs that change the level at runtime.

    Returns
    -------
    logging.Logger
        The configured pipeline root logger.
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        level = resolved if isinstance(resolved, int) else logging.INFO

    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the pipeline root logger.

    Parameters
    ----------
    name : str
        Logger name, conventionally the ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A logger whose records propagate to the pipeline root handler.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")