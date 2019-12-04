## Program

Program complex for solving Boltzmann kinetic equation with the S-model collision integral in arbitrary spacial domain using TT-format.

A folder must contain the following files:

- `mesh-* folder` : contains mesh.pickle (a pickled preprocessed mesh) and 4 mesh files, defining vertices, faces, cells and boundary faces

- `read_starcd.py` : processes (reads from file into memory) spacial mesh in starcd format for further usage in the program; also contains function which writes a tecplot file with macroparameters

- `solver.py / solver_tt.py` : solver code without and with TT-decomposition (explicit or implicit) and all secondary functions

- `run_*.py` : where parameters are set (CFL, velocities mesh, etc.); run this script to perform the solver

This files are placed in `code` and `mesh` folders.

To solve a problem you should place this files into one folder and run the `run_*.py` script.

The result would be:

- `` : numpy array which could be used as a restart file

- `norm_iter.png` : a graph depicting evolution of RHS

- `tec.dat` : tecplot file with macroparameters

- `macroparameters_data.txt` : text file with macroparameters info in each cell

- `log.txt` : log file

## Tests

This repository has tests for 1D shock wave structure problem and flow past cylinder, accordant meshes are in `mesh` folder.

## Reference 

*link to an article*
