"""
analysis/
=========
Post-processing modules for the WMoNbZrTiTa diffusion pipeline.

Modules
-------
msd.py         Stage 5  — Parse MSD from LAMMPS thermo logs
Dv.py          Stage 6  — Vacancy diffusion coefficients Dv(T)
Cv.py          Stage 7  — Equilibrium vacancy concentration Cv(T)
D2.py          Stage 8  — Tracer self-diffusion D*(T)
sro.py         Stage 9  — Warren-Cowley short-range order parameters
postprocess.py Stage 10 — Polynomial fitting, comparison plots
"""
