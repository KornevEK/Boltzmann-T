from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import time
from read_starcd import write_tecplot

def f_maxwell(vx, vy, vz, T, n, ux, uy, uz, Rg):
    """Compute maxwell distribution function on cartesian velocity mesh
    
    vx, vy, vz - 3d numpy arrays with x, y, z components of velocity mesh
    in each node
    T - float, temperature in K
    n - float, numerical density
    ux, uy, uz - floats, x,y,z components of equilibrium velocity
    Rg - gas constant for specific gas
    """
    return n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * (np.exp(-((vx - ux)**2 + (vy - uy)**2 + (vz - uz)**2) / (2. * Rg * T)))

class GasParams:
    Na = 6.02214129e+23 # Avogadro constant
    kB = 1.381e-23 # Boltzmann constant, J / K
    Ru = 8.3144598 # Universal gas constant

    def __init__(self, Mol = 40e-3, Pr = 2. / 3., g = 5. / 3., d = 3418e-13):
        self.Mol = Mol
        self.Rg = self.Ru  / self.Mol  # J / (kg * K) 
        self.m = self.Mol / self.Na # kg
    
        self.Pr = Pr
        
        self.C = 144.4
        self.T_0 = 273.11
        self.mu_0 = 2.125e-05
        self.mu_suth = lambda T: self.mu_0 * ((self.T_0 + self.C) / (T + self.C)) * ((T / self.T_0) ** (3. / 2.))
        self.mu = lambda T: self.mu_suth(200.) * (T/200.)**0.734
        self.g = g # specific heat ratio
        self.d = d # diameter of molecule
        
class Problem:
    def __init__(self, bc_type_list = None, bc_data = None, f_init = None):
        # list of boundary conditions' types
        # acording to order in starcd '.bnd' file
        # list of strings
        self.bc_type_list = bc_type_list
        # data for b.c.: wall temperature, inlet n, u, T and so on.
        # list of lists
        self.bc_data = bc_data
        # Function to set initial condition
        self.f_init = f_init

def set_bc(gas_params, bc_type, bc_data, f, vx, vy, vz, vn):
    """Set boundary condition
    """
    if (bc_type == 'sym-x'): # symmetry in x
        return f[::-1, :, :]
    elif (bc_type == 'sym-y'): # symmetry in y
        return f[:, ::-1, :]
    elif (bc_type == 'sym-z'): # symmetry in z
        return f[:, :, ::-1]
    elif (bc_type == 'sym'): # zero derivative
        return f[:, :, :]
    elif (bc_type == 'in'): # inlet
        # unpack bc_data
        n =  bc_data[0]
        ux = bc_data[1]
        uy = bc_data[2]
        uz = bc_data[3]
        T =  bc_data[4]
        return f_maxwell(vx, vy, vz, T, n, ux, uy, uz, gas_params.Rg)
    elif (bc_type == 'out'): # outlet
        # unpack bc_data
        n =  bc_data[0]
        ux = bc_data[1]
        uy = bc_data[2]
        uz = bc_data[3]
        T =  bc_data[4]
        return f_maxwell(vx, vy, vz, T, n, ux, uy, uz, gas_params.Rg)
    elif (bc_type == 'wall'): # wall
        # unpack bc_data
        T_w = bc_data[0]
        hv = vx[1, 0, 0] - vx[0, 0, 0]
        fmax = f_maxwell(vx, vy, vz, T_w, 1., 0., 0., 0., gas_params.Rg)
        Ni = (hv**3) * np.sum(f * np.where(vn > 0, vn, 0.))
        Nr = (hv**3) * np.sum(fmax * np.where(vn < 0, vn, 0.))
        # TODO: replace np.sqrt(2 * np.pi / (gas_params.Rg * T_w))
        # with discrete quarature, as in the dissertation
        n_wall = - Ni/ Nr
#        n_wall = 2e+23 # temprorary
        return n_wall * fmax
            
def comp_macro_param_and_j(f, vx, vy, vz, gas_params):
    Rg = gas_params.Rg
    hv = vx[1, 0, 0] - vx[0, 0, 0]
    n = (hv ** 3) * np.sum(f)

    ux = (1. / n) * (hv ** 3) * np.sum(vx * f)
    uy = (1. / n) * (hv ** 3) * np.sum(vy * f)
    uz = (1. / n) * (hv ** 3) * np.sum(vz * f)
    
    v2 = vx*vx + vy*vy + vz*vz
    u2 = ux*ux + uy*uy + uz*uz
    
    T = (1. / (3. * n * Rg)) * ((hv ** 3) * np.sum(v2 * f) - n * u2)

    Vx = vx - ux
    Vy = vy - uy
    Vz = vz - uz

    rho = gas_params.m * n

    p = rho * Rg * T

    cx = Vx / ((2. * Rg * T) ** (1. / 2.))
    cy = Vy / ((2. * Rg * T) ** (1. / 2.))
    cz = Vz / ((2. * Rg * T) ** (1. / 2.))
    
    c2 = cx*cx + cy*cy + cz*cz

    Sx = (1. / n) * (hv ** 3) * np.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * np.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * np.sum(cz * c2 * f)

    mu = gas_params.mu(T)

    f_plus = f_maxwell(vx, vy, vz, T, n, ux, uy, uz, gas_params.Rg) * (1. + (4. / 5.) * (1. - gas_params.Pr) * (cx*Sx + cy*Sy + cz*Sz) * (c2 - (5. / 2.)))

    J = (f_plus - f) * (p / mu)
    
    nu = p / mu
    
    return J, n, ux, uy, uz, T, nu, rho, p

    
def solver(gas_params, problem, mesh, nt, vmax, nv, CFL, filename, init = '0'):
    """Solve Boltzmann equation with model collision integral 
    
    gas_params -- object of class GasParams, contains gas parameters and viscosity law
    
    problem -- object of class Problem, contains list of boundary conditions,
    data for b.c., and function for initial condition
    
    mesh - object of class Mesh
    
    nt -- number of time steps
    
    vmax -- maximum velocity in each direction in velocity mesh
    
    nv -- number of nodes in velocity mesh
    
    CFL -- courant number
    
    filename -- name of output file for f
    
    init - name of restart file
    """
        
    h = np.min(mesh.cell_diam)
    tau = h * CFL / (vmax * (3.**0.5))
    
    hv = 2. * vmax / nv
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes
    
    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')
    
    # set initial condition 
    f = np.zeros((mesh.nc, nv, nv, nv))
    if (init == '0'):
        for i in range(mesh.nc):
            x = mesh.cell_center_coo[i, 0]
            y = mesh.cell_center_coo[i, 1]
            z = mesh.cell_center_coo[i, 2]
            f[i, :, :, :] = problem.f_init(x, y, z, vx, vy, vz) 
    else:
#        restart from distribution function
#        f = np.reshape(np.load(init), (mesh.nc, nv, nv, nv))
#        restart form macroparameters array
        init_data = np.loadtxt(init)
        for ic in range(mesh.nc):
            f[ic, :, :, :] = f_maxwell(vx, vy, vz, init_data[ic, 5], init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], gas_params.Rg)
    
    # TODO: may be join f_plus and f_minus in one array
    f_plus = np.zeros((mesh.nf, nv, nv, nv)) # Reconstructed values on the right
    f_minus = np.zeros((mesh.nf, nv, nv, nv)) # reconstructed values on the left
    flux = np.zeros((mesh.nf, nv, nv, nv)) # Flux values
    rhs = np.zeros((mesh.nc, nv, nv, nv))
    df = np.zeros((mesh.nc, nv, nv, nv)) # Array for increments \Delta f    
    vn = np.zeros((mesh.nf, nv, nv, nv))
    for jf in range(mesh.nf):
        vn[jf, :, :, :] = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz

    diag = np.zeros((mesh.nc, nv, nv, nv)) # part of diagonal coefficient in implicit scheme
    # precompute diag
    for ic in range(mesh.nc):
        for j in range(6):
            jf = mesh.cell_face_list[ic, j]
            vnp = np.where(mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :] > 0,
                                    mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :], 0.)
            diag[ic, :, :, :] += (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vnp
    # Arrays for macroparameters
    n = np.zeros(mesh.nc)
    rho = np.zeros(mesh.nc)
    ux = np.zeros(mesh.nc)
    uy = np.zeros(mesh.nc)
    uz = np.zeros(mesh.nc)
    p =  np.zeros(mesh.nc)
    T = np.zeros(mesh.nc)
    nu = np.zeros(mesh.nc)
    data = np.zeros((mesh.nc, 7))
    
    frob_norm_rhs = np.zeros(mesh.nc)
    frob_norm_iter = np.array([])

    it = 0
    while(it < nt):
        it += 1
        # reconstruction for inner faces
        # 1st order
        for ic in range(mesh.nc):
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                # TODO: think how do this without 'if'
                if (mesh.cell_face_normal_direction[ic, j] == 1):
                    f_minus[jf, :, :, :] = f[ic, :, :, :]
                else:
                    f_plus[jf, :, :, :] = f[ic, :, :, :]
                      
        # boundary condition
        # loop over all boundary faces
        for j in range(mesh.nbf):
            jf = mesh.bound_face_info[j, 0] # global face index
            bc_num = mesh.bound_face_info[j, 1]
            bc_type = problem.bc_type_list[bc_num]
            bc_data = problem.bc_data[bc_num]
            if (mesh.bound_face_info[j, 2] == 1):
                # TODO: normal velocities vn can be pre-computed one time
                # then we can pass to function p.bc only vn
                f_plus[jf, :, :, :] =  set_bc(gas_params, bc_type, bc_data, f_minus[jf, :, :, :], vx, vy, vz, vn[jf, :, :, :])
            else:
                f_minus[jf, :, :, :] = set_bc(gas_params, bc_type, bc_data, f_plus[jf, :, :, :], vx, vy, vz, -vn[jf, :, :, :])

        # riemann solver - compute fluxes
        for jf in range(mesh.nf):
            # TODO: Pre-compute array of vn[:] ???
            flux[jf, :, :, :] = mesh.face_areas[jf] * vn[jf, :, :, :] * \
            np.where((vn[jf, :, :, :] < 0), f_plus[jf, :, :, :], f_minus[jf, :, :, :])
#            flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((vn * (f_plus[jf, :, :, :] + f_minus[jf, :, :, :])) - (vn_abs * (f_plus[jf, :, :, :] - f_minus[jf, :, :, :])))
                
        # computation of the right-hand side
        rhs[:] = 0.
        for ic in range(mesh.nc):
            # sum up fluxes from all faces of this cell
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                rhs[ic, :, :, :] += - (mesh.cell_face_normal_direction[ic, j]) * (1. / mesh.cell_volumes[ic]) * flux[jf, :, :, :]
            # Compute macroparameters and collision integral
            J, n[ic], ux[ic], uy[ic], uz[ic], T[ic], nu[ic], rho[ic], p[ic] = comp_macro_param_and_j(f[ic, :, :, :], vx, vy, vz, gas_params)
            rhs[ic, :, :, :] += J
        
        frob_norm_iter = np.append(frob_norm_iter, np.linalg.norm(rhs))
        #     
        # update values - explicit scheme
        #
        f += tau * rhs
        
        if ((it % 50) == 0):     
            fig, ax = plt.subplots(figsize = (20,10))
            line, = ax.semilogy(frob_norm_iter/frob_norm_iter[0])
            ax.set(title='$Steps =$' + str(it))
            plt.grid(True)
            plt.savefig('norm_iter.png')
            plt.close()
                            
            data[:, 0] = n[:]
            data[:, 1] = ux[:]
            data[:, 2] = uy[:]
            data[:, 3] = uz[:]
            data[:, 4] = p[:]
            data[:, 5] = T[:]
            data[:, 6] = np.zeros(mesh.nc)
            
            write_tecplot(mesh, data, 'cyl.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))
            np.save(filename, np.ravel(f))
        
    for ic in range(mesh.nc):
        frob_norm_rhs[ic] = np.linalg.norm(rhs[ic])
    
    Return = namedtuple('Return', ['f', 'n', 'ux', 'uy', 'uz', 'T', 'p', 'frob_norm_iter', 'frob_norm_rhs'])
    
    S = Return(f, n, ux, uy, uz, T, p, frob_norm_iter, frob_norm_rhs)
  
    return S
