Program complex for solving Boltzmann kinetic equation with the S-model collision integral in arbitrary spacial domain.

A folder must contain the following files:

- `mesh* folder` -- contains mesh.pickle and 4 mesh files

- `read_starcd.py` -- preprocesses spacial mesh in starcd format

- `solver.py / solver_tt.py` -- solving code (explicit or implicit)

- `run_*.py` -- where parameters are set (CFL, velocities mesh)

