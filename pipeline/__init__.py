"""LAMMPS and phonopy input generation for the WMoNbZrTiTa pipeline.

Modules
-------
lammps_common
    Shared composition algebra and LAMMPS block builders.
compositions
    Stage 1 - composition sampling.
constants
    Stage 2 - a0, Tm_rom, Ef, Sf, C0, and the temperature grid.
lammps_diffusion
    Stage 3 - ADP diffusion inputs.
lammps_tm
    Stages 3b/3c - GRACE coexistence inputs and Tm patching.
compute_sf_phonopy
    Standalone phonopy calculation of the vacancy formation entropy.
"""