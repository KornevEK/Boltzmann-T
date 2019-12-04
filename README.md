## Program

Program complex for solving Boltzmann kinetic equation with the S-model collision integral in arbitrary spacial domain using TT-format.

A folder must contain the following files:

- `mesh* folder` : contains mesh.pickle (a pickled preprocessed mesh) and 4 mesh files, defining vertices, faces, cells and boundary faces

- `read_starcd.py` : processes (reads from file into memory) spacial mesh in starcd format for further usage in the program; also contains function which writes a tecplot file with macroparameters

- `solver.py / solver_tt.py` : solver code without and with TT-decomposition (explicit or implicit) and all secondary functions

- `run_*.py` : where parameters are set (CFL, velocities mesh, etc.); run this file to perform the solver

## Reference 

*link to an article*
