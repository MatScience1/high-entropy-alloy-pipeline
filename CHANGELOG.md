# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-20

Production-readiness refactor for open-source publication. The architecture,
pipeline stages, and core physics are preserved; this release focuses on
documentation, robustness, CLI/UX, and project infrastructure.

### Added

- `logging_config.py`: central logging configuration with `configure_logging()`
  and `get_logger()`. Progress uses `INFO`, skipped runs use `WARNING`, and
  failures use `ERROR`.
- `reporting.py`: aligned plain-text table rendering (`format_dataframe()`,
  `format_mapping()`, `log_dataframe()`, `log_mapping()`, `format_float()`)
  built on pandas, requiring no additional runtime dependency.
- `pipeline/lammps_common.py`: shared LAMMPS input helpers. Centralizes
  `sequential_fracs()`, `mass_block()`, and `set_fraction_block()` so the ADP
  and GRACE generators assign identical compositions.
- `analysis/logparse.py`: shared LAMMPS text-parsing helpers, including the
  `LogParseError` exception and a safe `read_lines()` reader.
- `analysis/arrhenius.py`: weighted Arrhenius fitting. Provides
  `fit_arrhenius_wls()` (weighted least squares) and `fit_arrhenius_irls()`
  (iteratively reweighted least squares), plus the `ArrheniusFit` result type.
- `config.set_test_mode()`: runtime switch between the test and production
  configurations, recomputing all mode-dependent parameters.
- `config.SET_FRACTION_ORDER` and `config.SET_FRACTION_SEEDS`: centralized,
  deterministic ordering and RNG seeds for `set type/fraction`.
- `config.__version__`: programmatic access to the package version.
- `run_pipeline.py` CLI flags: `--test`, `--production`, and `--log-level`.
- `pyproject.toml`: packaging metadata and strictly pinned dependencies, with
  a `hea-pipeline` console entry point and pytest configuration.
- `Makefile`: workflow shortcuts (`setup`, `generate`, `tm-inputs`,
  `submit-tm`, `tm-patch`, `diffusion-inputs`, `submit`, `analyze`, `test`,
  `lint`, `clean`).
- `tests/`: pytest suite covering the sequential-fraction algebra, weighted
  Arrhenius fitting, graceful MSD log parsing, the mode switch, and
  composition sampling.
- `CHANGELOG.md`: this file.

### Changed

- Renamed analysis modules to descriptive names:
  - `analysis/D2.py` -> `analysis/tracer_diffusion.py`
  - `analysis/Dv.py` -> `analysis/vacancy_diffusion.py`
  - `analysis/Cv.py` -> `analysis/vacancy_concentration.py`
- `config.TEST_MODE` now defaults to `False` (full production run). It can be
  overridden at runtime with `--test` or `set_test_mode(True)`.
- Arrhenius fitting in the tracer-diffusion stage upgraded from ordinary least
  squares to weighted least squares to account for heteroscedastic MSD noise at
  the temperature extremes.
- Log parsing in `analysis/msd.py` and `analysis/sro.py` now skips malformed or
  truncated rows with a warning and raises `LogParseError` instead of aborting
  the pipeline.
- Replaced all raw `print()` statements with the `logging` module across the
  pipeline, analysis, and orchestrator modules.
- Numerical outputs (composition tables, constants, diffusion coefficients,
  SRO matrices, Arrhenius parameters) are now rendered as aligned text tables.
- Enforced strict type hints across all modules.
- `requirements.txt` now pins exact versions and mirrors `pyproject.toml`.
- `run_pipeline.py` imports the renamed analysis modules and reads
  mode-dependent values through the `config` module so runtime overrides take
  effect.
- `pipeline/compositions.py` uses `raise ValueError` instead of `assert` for
  the simplex sanity check, so the check survives optimized (`-O`) runs.
- `pipeline/compute_sf_phonopy.py` de-duplicated imports and added a
  `main()` entry point with logging.
- `analysis/postprocess.py` uses `plt.get_cmap()` instead of the deprecated
  `matplotlib.cm.get_cmap()`.

### Fixed

- Corrected the `set type/fraction` emission order in `pipeline/constants.py`.
  The previous code emitted the commands in reverse order (Ta first),
  contradicting the documented sequential-fraction algebra and producing
  incorrect compositions in the `a0` and `Ef` calculations. The commands are
  now emitted in `SET_FRACTION_ORDER` (Mo first, Ta last).
- Replaced the non-deterministic `abs(hash(element))` seeds in
  `pipeline/constants.py` with the fixed `SET_FRACTION_SEEDS`, making the
  composition assignment reproducible across runs and `PYTHONHASHSEED` values.
- Guarded against missing simulation directories and unreadable logs in the
  analysis stages, which previously raised uncaught exceptions.

### Removed

- `analysis/D2.py`, `analysis/Dv.py`, `analysis/Cv.py` (superseded by the
  renamed modules).
- The empty `import` file at the repository root.
- `pipeline/test_a0_parse.py` (a non-pytest script that executed on import);
  its coverage is subsumed by the `tests/` suite.
- `config.R_SRO`, the legacy fixed SRO cutoff constant. All callers use
  `config.compute_r_sro(a0)`.
- Unused imports and dead code across the pipeline and analysis modules.

### Documentation

- Rewrote `README.md` with an architecture overview, a full data-flow map, the
  repository layout, the runtime output file structure, installation and quick
  start via the Makefile, a stage reference table, and physics notes with
  citations.
- Added rigorous NumPy/SciPy-style docstrings documenting the physical
  rationale for the sequential-fraction algebra, the Modified Z-method
  coexistence setup, the dynamic SRO cutoff, and the tracer-versus-vacancy
  diffusion distinction including the BCC correlation factor.
- Updated `docs/logic.md`, `docs/workflow.md`, and `docs/testing.md` for the
  renamed modules, the `--test` flag, and the Makefile workflow.

### Infrastructure

- Expanded `.gitignore` to cover SLURM logs, LAMMPS dumps, restart and data
  files, local environment configuration, virtual environments, and IDE
  artifacts.
- Added `pyproject.toml` as the dependency source of truth and locked
  `requirements.txt`.

### Validation

- All modules byte-compile and import successfully.
- The unit test suite passes.
- Stage 0 and Stage 1 smoke tests run with clean, tabular logging.

[1.1.0]: https://github.com/akmal523/high-entropy-alloy-pipeline/releases/tag/v1.1.0