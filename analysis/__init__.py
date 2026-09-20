"""Post-processing and analytics for the WMoNbZrTiTa diffusion pipeline.

Modules
-------
logparse
    Shared LAMMPS text-parsing helpers and the :class:`LogParseError` type.
arrhenius
    Weighted and iteratively reweighted Arrhenius fitting.
msd
    Stage 5 - mean-square displacement extraction.
vacancy_diffusion
    Stage 6 - vacancy diffusion coefficient Dv(T).
vacancy_concentration
    Stage 7 - equilibrium vacancy concentration Cv(T).
tracer_diffusion
    Stage 8 - tracer self-diffusion coefficient D*(T).
sro
    Stage 9 - Warren-Cowley short-range order parameters.
postprocess
    Stage 10 - aggregation, polynomial fit, and comparison plots.
"""
