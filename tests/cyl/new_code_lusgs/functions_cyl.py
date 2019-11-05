from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import time
from read_starcd import write_tecplot
import tt

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

    t = time.time()

    log = open('log.txt', 'w') #log file

    cpu_time = np.zeros(10)
    cpu_time_name = ['reconstruction', 'boundary', 'fluxes', 'rhs', 'update', 'rhs_j']
        
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
        f = np.reshape(np.load(init), (mesh.nc, nv, nv, nv))
#        restart form macroparameters array
#        init_data = np.loadtxt(init)
#        for ic in range(mesh.nc):
#            f[ic, :, :, :] = f_maxwell(vx, vy, vz, init_data[ic, 5], init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], gas_params.Rg)
    
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

    t_ = time.time() - t
#    print "initialization", t_
    log.write("initialization " + str(t_) + "\n")
    
    t1 = time.time()
    it = 0
    while(it < nt):
        it += 1
        # reconstruction for inner faces
        # 1st order
        t = time.time()
        for ic in range(mesh.nc):
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                # TODO: think how do this without 'if'
                if (mesh.cell_face_normal_direction[ic, j] == 1):
                    f_minus[jf, :, :, :] = f[ic, :, :, :]
                else:
                    f_plus[jf, :, :, :] = f[ic, :, :, :]
        t_ = time.time() - t
        cpu_time[0] += t_
#        print "reconstruction", it, t_
        log.write("reconstruction " + str(it) + " " + str(t_) + "\n")
                      
        # boundary condition
        # loop over all boundary faces
        t = time.time()
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
        t_ = time.time() - t
        cpu_time[1] += t_
#        print "boundary", it, t_
        log.write("boundary " + str(it) + " " + str(t_) + "\n")

        # riemann solver - compute fluxes
        t = time.time()
        for jf in range(mesh.nf):
            # TODO: Pre-compute array of vn[:] ???
            flux[jf, :, :, :] = mesh.face_areas[jf] * vn[jf, :, :, :] * \
            np.where((vn[jf, :, :, :] < 0), f_plus[jf, :, :, :], f_minus[jf, :, :, :])
#            flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((vn * (f_plus[jf, :, :, :] + f_minus[jf, :, :, :])) - (vn_abs * (f_plus[jf, :, :, :] - f_minus[jf, :, :, :])))
        t_ = time.time() - t
        cpu_time[2] += t_
#        print "fluxes", it, t_ 
        log.write("fluxes " + str(it) + " " + str(t_) + "\n")
                
        # computation of the right-hand side
        t = time.time()
        rhs[:] = 0.
        for ic in range(mesh.nc):
            # sum up fluxes from all faces of this cell
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                rhs[ic, :, :, :] += - (mesh.cell_face_normal_direction[ic, j]) * (1. / mesh.cell_volumes[ic]) * flux[jf, :, :, :]
            # Compute macroparameters and collision integral
            jt = time.time()
            J, n[ic], ux[ic], uy[ic], uz[ic], T[ic], nu[ic], rho[ic], p[ic] = comp_macro_param_and_j(f[ic, :, :, :], vx, vy, vz, gas_params)
            rhs[ic, :, :, :] += J
            cpu_time[5] += (time.time() - jt)
        t_ = time.time() - t
        cpu_time[3] += t_
#        print "rhs", it, t_
        log.write("rhs " + str(it) + " " + str(t_) + "\n")
        
        frob_norm_iter = np.append(frob_norm_iter, np.linalg.norm(rhs))
        #     
        # update values - explicit scheme
        #
        t = time.time()
#        f += tau * rhs
        '''
        LU-SGS iteration
        '''
        #
        # Backward sweep 
        #
        for ic in range(mesh.nc - 1, -1, -1):
            df[ic, :, :, :] = rhs[ic, :, :, :]
            # loop over neighbors of cell ic
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                icn = mesh.cell_neighbors_list[ic, j] # index of neighbor
                vnm = np.where(mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :] < 0,
                                    mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :], 0.)
                if (icn >= 0 ) and (icn > ic):
                    df[ic, :, :, :] += -(mesh.face_areas[jf] / mesh.cell_volumes[ic]) \
                    * vnm * df[icn, : , :, :] 
            # divide by diagonal coefficient
            df[ic, :, :, :] = df[ic, :, :, :] / (np.ones((nv, nv, nv)) * (1/tau + nu[ic]) + diag[ic])
        #
        # Forward sweep
        # 
        for ic in range(mesh.nc):
            # loop over neighbors of cell ic
            incr = np.zeros((nv, nv, nv))
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                icn = mesh.cell_neighbors_list[ic, j] # index of neighbor
                vnm = np.where(mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :] < 0,
                                    mesh.cell_face_normal_direction[ic, j] * vn[jf, :, :, :], 0.)
                if (icn >= 0 ) and (icn < ic):
                    incr+= -(mesh.face_areas[jf] /  mesh.cell_volumes[ic]) \
                    * vnm * df[icn, : , :, :] 
            # divide by diagonal coefficient
            df[ic, :, :, :] += incr / (np.ones((nv, nv, nv)) * (1./tau + nu[ic]) + diag[ic])
        f += df
        '''
        end of LU-SGS iteration
        '''
        t_ = time.time() - t
        cpu_time[4] += t_
#        print "update", it, t_
        log.write("update " + str(it) + " " + str(t_) + "\n")
        
        if ((it % 50) == 0):     
            fig, ax = plt.subplots(figsize = (20,10))
            line, = ax.semilogy(frob_norm_iter/frob_norm_iter[0])
            ax.set(title='$Steps =$' + str(it))
            plt.grid(True)
            plt.savefig('norm_iter.png')
            plt.close()
                            
            data[:, 0] = n[:] # now n instead of rho
            data[:, 1] = ux[:]
            data[:, 2] = uy[:]
            data[:, 3] = uz[:]
            data[:, 4] = p[:]
            data[:, 5] = T[:]
            data[:, 6] = np.zeros(mesh.nc)
            
            write_tecplot(mesh, data, 'cyl.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))
            np.save(filename, np.ravel(f))
        
    t2 = time.time() - t1

    t2 = int(round(t2))
    
    
#    print "time =", t2 / 3600, "h", (t2 % 3600) / 60, "m", t2 % 60, "s"
    log.write("time = " + str(t2 / 3600) + " h " + str((t2 % 3600) / 60) + " m " + str(t2 % 60) + " s " + "\n")
        
    for ic in range(mesh.nc):
        frob_norm_rhs[ic] = np.linalg.norm(rhs[ic])

#    l = 1. / ((2 ** .5) * np.pi * n_l * p.d * p.d)
#    delta = l / (n_r - n_l) * np.max(Dens[1:] - Dens[:-1]) / (2 * h
    
    Return = namedtuple('Return', ['f', 'n', 'ux', 'uy', 'uz', 'T', 'p', 'frob_norm_iter', 'frob_norm_rhs'])
    
    S = Return(f, n, ux, uy, uz, T, p, frob_norm_iter, frob_norm_rhs)

    for i in range(len(cpu_time_name)):
#        print cpu_time[i], cpu_time_name[i]
        log.write(str(cpu_time[i]) + " " + cpu_time_name[i] + "\n")

    log.close()    
    return S

def set_bc_tt(gas_params, bc_type, bc_data, f, vx, vy, vz, vn, vnp, vnm, tol):
    """Set boundary condition
    """
    if (bc_type == 'sym-x'): # symmetry in x
        l = f.to_list(f)
        l[0] = l[0][:,::-1,:]
        return f.from_list(l)
    elif (bc_type == 'sym-y'): # symmetry in y
        l = f.to_list(f)
        l[1] = l[1][:,::-1,:]
        return f.from_list(l)
    elif (bc_type == 'sym-z'): # symmetry in z
        l = f.to_list(f)
        l[2] = l[2][:,::-1,:]
        return f.from_list(l)
    elif (bc_type == 'sym'): # zero derivative
        return f.copy()
    elif (bc_type == 'in'): # inlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'out'): # outlet
        # unpack bc_data
        return bc_data[0]
    elif (bc_type == 'wall'): # wall
        # unpack bc_data
        fmax = bc_data[0]
        hv = vx[1, 0, 0] - vx[0, 0, 0]
        Ni = (hv**3) * tt.sum((f * vnp).round(tol))
        Nr = (hv**3) * tt.sum((fmax * vnm).round(tol))
        n_wall = - Ni/ Nr
        return n_wall * fmax
            
def comp_macro_param_and_j_tt(f, vx_, vx, vy, vz, vx_tt, vy_tt, vz_tt, v2, gas_params, tol, F, ones_tt):
    # takes "F" with (1, 1, 1, 1) ranks to perform to_list function
    # takes precomputed "ones_tt" tensor
    # "ranks" array for mean ranks
    # much less rounding
    Rg = gas_params.Rg
    hv = vx_[1] - vx_[0]
    n = (hv ** 3) * tt.sum(f)

    ux = (1. / n) * (hv ** 3) * tt.sum(vx_tt * f)
    uy = (1. / n) * (hv ** 3) * tt.sum(vy_tt * f)
    uz = (1. / n) * (hv ** 3) * tt.sum(vz_tt * f)
    
    u2 = ux*ux + uy*uy + uz*uz
    
    T = (1. / (3. * n * Rg)) * ((hv ** 3) * tt.sum((v2 * f)) - n * u2)

    Vx = vx - ux
    Vy = vy - uy
    Vz = vz - uz

    Vx_tt = (vx_tt - ux * ones_tt).round(tol)
    Vy_tt = (vy_tt - uy * ones_tt).round(tol)
    Vz_tt = (vz_tt - uz * ones_tt).round(tol)

    rho = gas_params.m * n

    p = rho * Rg * T

    cx = Vx_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    cy = Vy_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    cz = Vz_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    # TODO: do rounding after each operation
    c2 = ((cx*cx) + (cy*cy) + (cz*cz)).round(tol)
    
    Sx = (1. / n) * (hv ** 3) * tt.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * tt.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * tt.sum(cz * c2 * f)

    mu = gas_params.mu(T)

    fmax = F
    fmax_list = F.to_list(F)
    fmax_list[0][0, :, 0] = np.exp(-((vx_ - ux) ** 2) / (2. * Rg * T)) # vx_ instead of Vx[0, :, :]
    fmax_list[1][0, :, 0] = np.exp(-((vx_ - uy) ** 2) / (2. * Rg * T))
    fmax_list[2][0, :, 0] = np.exp(-((vx_ - uz) ** 2) / (2. * Rg * T))
    fmax = fmax.from_list(fmax_list)
    fmax = n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * fmax
    fmax = fmax.round(tol)
	
	# TODO: replace with tt.from_list
    #fmax = tt.tensor(f_maxwell(vx, vy, vz, T, n, ux, uy, uz, gas_params.Rg))
	# TODO: do rounding after each operation
    f_plus = fmax * (ones_tt + ((4. / 5.) * (1. - gas_params.Pr) * (cx*Sx + cy*Sy + cz*Sz) * ((c2 - (5. / 2.) * ones_tt))))
    f_plus = f_plus.round(tol)
    J = (f_plus - f) * (p / mu)
    J = J.round(tol)
    nu = p / mu
    return J, n, ux, uy, uz, T, nu, rho, p

def save_tt(filename, f, L, N):
    
    m = max(f[i].core.size for i in range(L))

    F = np.zeros((m+4, L))
    
    for i in range(L):
        F[:4, i] = f[i].r.ravel()
        F[4:f[i].core.size+4, i] = f[i].core.ravel()
    
    np.save(filename, F)#, fmt='%s')
    
def load_tt(filename, L, N):
    
    F = np.load(filename)
    
    f = list()
    
    for i in range(L):
        
        f.append(tt.rand([N, N, N], 3, F[:4, i]))
        f[i].core = F[4:f[i].core.size+4, i]
        
    return f

def solver_tt(gas_params, problem, mesh, nt, nv, vx_, vx, vy, vz, \
              CFL, tol, filename, init = '0'):
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
#    CPU_time = namedtuple('CPU_time', ['reconstruction', 'boundary', 'fluxes', 'rhs', 'update'])
#    CPU = CPU_time(0., 0., 0., 0., 0.)
    # FUnction for LU-SGS
    mydivide = lambda x: x[:, 0] / x[:, 1]
    zero_tt = 0. * tt.ones((nv,nv,nv))
    ones_tt = tt.ones((nv,nv,nv))
    #
    # Initialize main arrays and lists
    #
    vn = [None] * mesh.nf # list of tensors of normal velocities at each mesh face  
    vn_tmp = np.zeros((nv, nv, nv))
    vnm = [None] * mesh.nf # negative part of vn: 0.5 * (vn - |vn|)
    vnp = [None] * mesh.nf # positive part of vn: 0.5 * (vn + |vn|) 
    vn_abs = [None] * mesh.nf # approximations of |vn|
    
    # TODO: create this rank-1 tensors explicitly using tensor.from_list()   
    vx_tt = tt.tensor(vx).round(tol)
    vy_tt = tt.tensor(vy).round(tol)
    vz_tt = tt.tensor(vz).round(tol)
    
    vn_error = 0.
    for jf in range(mesh.nf):
        vn_tmp = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz
        vn[jf] = mesh.face_normals[jf, 0] * vx_tt + mesh.face_normals[jf, 1] * vy_tt + mesh.face_normals[jf, 2] * vz_tt
        vnp[jf] = tt.tensor(np.where(vn_tmp > 0, vn_tmp, 0.), eps = tol)
        vnm[jf] = tt.tensor(np.where(vn_tmp < 0, vn_tmp, 0.), eps = tol)
        vn_abs[jf] = tt.tensor(np.abs(vn_tmp), rmax = 4)
        vn_error = max(vn_error, np.linalg.norm(vn_abs[jf].full() - np.abs(vn_tmp))/
                       np.linalg.norm(np.abs(vn_tmp)))
    print('max||vn_abs_tt - vn_abs||_F/max||vn_abs||_F = ', vn_error)

    h = np.min(mesh.cell_diam)
    tau = h * CFL / (np.max(vx_) * (3.**0.5))
    v2 = (vx_tt*vx_tt + vy_tt*vy_tt + vz_tt*vz_tt).round(tol)

    diag = [None] * mesh.nc # part of diagonal coefficient in implicit scheme
    # precompute diag
    diag_full = np.zeros((mesh.nc, nv, nv, nv)) # part of diagonal coefficient in implicit scheme
    # precompute diag
    for ic in range(mesh.nc):
        for j in range(6):
            jf = mesh.cell_face_list[ic, j]
            vn_tmp = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz
            vnp_full = np.where(mesh.cell_face_normal_direction[ic, j] * vn_tmp[:, :, :] > 0,
                                    mesh.cell_face_normal_direction[ic, j] * vn_tmp[:, :, :], 0.)
            diag_full[ic, :, :, :] += (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vnp_full
    
    for ic in range(mesh.nc):
        diag_temp = np.zeros((nv, nv, nv))
        for j in range(6):
            jf = mesh.cell_face_list[ic, j]
            vn_full = (mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy \
                       + mesh.face_normals[jf, 2] * vz) * mesh.cell_face_normal_direction[ic, j]
            vnp_full = np.where(vn_full > 0, vn_full, 0.)
            vn_abs_full = np.abs(vn_full)
            diag_temp += (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vnp_full
#            diag_temp += 0.5 * (mesh.face_areas[jf] / mesh.cell_volumes[ic]) * vn_abs_full
        diag[ic] = tt.tensor(diag_temp)
    # Compute mean rank of diag
    diag_rank = 0.
    for ic in range(mesh.nc):
        diag_rank += 0.5*(diag[ic].r[1] + diag[ic].r[2])
    diag_rank = diag_rank / ic
    print('diag_rank = ', diag_rank)
    #
    diag_scal = np.zeros(mesh.nc)
    for ic in range(mesh.nc):
        for j in range(6):
            jf = mesh.cell_face_list[ic, j]
            diag_scal[ic] += 0.5 * (mesh.face_areas[jf] / mesh.cell_volumes[ic])
    diag_scal *= np.max(np.abs(vx_)) * 3**0.5
    # set initial condition 
    f = [None] * mesh.nc
    if (init == '0'):
        for i in range(mesh.nc):
            x = mesh.cell_center_coo[i, 0]
            y = mesh.cell_center_coo[i, 1]
            z = mesh.cell_center_coo[i, 2]
            f[i] = problem.f_init(x, y, z, vx, vy, vz)
    else:
#        restart from distribution function
#        f = load_tt(init, mesh.nc, nv)
#        restart form macroparameters array
        init_data = np.loadtxt(init)
        for ic in range(mesh.nc):
            f[ic] = tt.tensor(f_maxwell(vx, vy, vz, init_data[ic, 5], \
             init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], gas_params.Rg), tol)
        
    # TODO: may be join f_plus and f_minus in one array
    f_plus = [None] * mesh.nf # Reconstructed values on the right
    f_minus = [None] * mesh.nf # reconstructed values on the left
    flux = [None] * mesh.nf # Flux values
    rhs = [None] * mesh.nc
    df = [None] * mesh.nc
    
    # Arrays for macroparameters
    n = np.zeros(mesh.nc)
    rho = np.zeros(mesh.nc)
    ux = np.zeros(mesh.nc)
    uy = np.zeros(mesh.nc)
    uz = np.zeros(mesh.nc)
    p = np.zeros(mesh.nc)
    T = np.zeros(mesh.nc)
    nu = np.zeros(mesh.nc)
    rank = np.zeros(mesh.nc)
    data = np.zeros((mesh.nc, 7))

    F = tt.tensor(f_maxwell(vx, vy, vz, 200., 2e+23, 100., 0., 0., gas_params.Rg)).round(tol)
    
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
                    f_minus[jf] = f[ic].copy()
                else:
                    f_plus[jf] = f[ic].copy()
 
        # boundary condition
        # loop over all boundary faces
        for j in range(mesh.nbf):
            jf = mesh.bound_face_info[j, 0] # global face index
            bc_num = mesh.bound_face_info[j, 1]
            bc_type = problem.bc_type_list[bc_num]
            bc_data = problem.bc_data[bc_num]
            if (mesh.bound_face_info[j, 2] == 1):
                # TODO: normal velocities vn can be pretime = 0 h 5 m 47 s-computed one time
                # then we can pass to function p.bc only vn
                f_plus[jf] =  set_bc_tt(gas_params, bc_type, bc_data, f_minus[jf], vx, vy, vz, vn[jf], vnp[jf], vnm[jf], tol)
            else:
                f_minus[jf] = set_bc_tt(gas_params, bc_type, bc_data, f_plus[jf], vx, vy, vz, -vn[jf], -vnm[jf], -vnp[jf], tol)

        # riemann solver - compute fluxes
        for jf in range(mesh.nf):
            flux[jf] = 0.5 * mesh.face_areas[jf] *\
            ((f_plus[jf] + f_minus[jf]) * vn[jf]  - (f_plus[jf] - f_minus[jf]) * vn_abs[jf])
            flux[jf] = flux[jf].round(tol)
# TODO: move round to separate string, measure time, store ranks before rounding

        # computation of the right-hand side
        for ic in range(mesh.nc):
            rhs[ic] = zero_tt.copy()
            # sum up fluxes from all faces of this cell
            for j in range(6):
				# TODO: do rounding after each binary operation
                jf = mesh.cell_face_list[ic, j]
                rhs[ic] += -(mesh.cell_face_normal_direction[ic, j]) * (1. / mesh.cell_volumes[ic]) * flux[jf]
                rhs[ic] = rhs[ic].round(tol)
            # Compute macroparameters and collision integral
            J, n[ic], ux[ic], uy[ic], uz[ic], T[ic], nu[ic], rho[ic], p[ic] = comp_macro_param_and_j_tt(f[ic], vx_, vx, vy, vz, vx_tt, vy_tt, vz_tt, v2, gas_params, tol, F, ones_tt)
            rhs[ic] += J
            rhs[ic] = rhs[ic].round(tol)

        frob_norm_iter = np.append(frob_norm_iter, np.sqrt(sum([(rhs[ic].norm())**2 for ic in range(mesh.nc)])))
        #
        # update values, expclicit scheme
        #
#        for ic in range(mesh.nc):
#            f[ic] = (f[ic] + tau * rhs[ic]).round(tol)
        '''
        LU-SGS iteration
        '''
        #
        # Backward sweep 
        #
        for ic in range(mesh.nc - 1, -1, -1):
            df[ic] = rhs[ic].copy()
        for ic in range(mesh.nc - 1, -1, -1):
            # loop over neighbors of cell ic
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                icn = mesh.cell_neighbors_list[ic, j] # index of neighbor
                if mesh.cell_face_normal_direction[ic, j] == 1:
                    vnm_loc = vnm[jf]
                else:
                    vnm_loc = -vnp[jf]
                if (icn >= 0 ) and (icn > ic):
#                    df[ic] += -(0.5 * mesh.face_areas[jf] / mesh.cell_volumes[ic]) \
#                        * (mesh.cell_face_normal_direction[ic, j] * vn[jf] * df[icn] + vn_abs[jf] * df[icn]) 
                    df[ic] += -(mesh.face_areas[jf] / mesh.cell_volumes[ic]) \
                    * vnm_loc * df[icn] 
                    df[ic] = df[ic].round(tol)
            # divide by diagonal coefficient
            diag_temp = ones_tt * (1/tau + nu[ic]) + diag[ic]
            df[ic] = tt.multifuncrs([df[ic], diag_temp], mydivide, eps = tol, verb = 0)
#            df[ic] = df[ic] * (1./(diag_scal[ic] + 1/tau + nu[ic] ))
        #
        # Forward sweep
        # 
        for ic in range(mesh.nc):
            # loop over neighbors of cell ic
            incr = zero_tt.copy()
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                icn = mesh.cell_neighbors_list[ic, j] # index of neighbor
                if mesh.cell_face_normal_direction[ic, j] == 1:
                    vnm_loc = vnm[jf]
                else:
                    vnm_loc = -vnp[jf]
                if (icn >= 0 ) and (icn < ic):
#                    incr+= -(0.5 * mesh.face_areas[jf] /  mesh.cell_volumes[ic]) \
#                    * (mesh.cell_face_normal_direction[ic, j] * vn[jf] + vn_abs[jf]) * df[icn] 
                    incr+= -(mesh.face_areas[jf] /  mesh.cell_volumes[ic]) \
                    * vnm_loc * df[icn] 
                    incr = incr.round(tol)
            # divide by diagonal coefficient
            diag_temp = ones_tt * (1/tau + nu[ic]) + diag[ic]
            df[ic] += tt.multifuncrs([incr, diag_temp], mydivide, eps = tol, verb = 0)
#            df[ic] += incr * (1./(diag_scal[ic] + 1/tau + nu[ic] ))
            df[ic] = df[ic].round(tol)
        #
        # Update values
        #
        for ic in range(mesh.nc):
            f[ic] += df[ic]
            f[ic] = f[ic].round(tol)
        '''
        end of LU-SGS iteration
        '''
        
        if ((it % 100) == 0):     
            fig, ax = plt.subplots(figsize = (20,10))
            line, = ax.semilogy(frob_norm_iter/frob_norm_iter[0])
            ax.set(title='$Steps =$' + str(it))
            plt.savefig('norm_iter.png')
            plt.close()
                
            data[:, 0] = n[:] # now n instead of rho
            data[:, 1] = ux[:]
            data[:, 2] = uy[:]
            data[:, 3] = uz[:]
            data[:, 4] = p[:]
            data[:, 5] = T[:]
            data[:, 6] = rank[:]
            
            write_tecplot(mesh, data, 'cyl.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))
        
    for ic in range(mesh.nc):
        frob_norm_rhs[ic] = rhs[ic].norm()

    save_tt(filename, f, mesh.nc, nv)
    
    Return = namedtuple('Return', ['f', 'n', 'ux', 'uy', 'uz', 'T', 'p', 'rank', 'frob_norm_iter', 'frob_norm_rhs'])
    
    S = Return(f, n, ux, uy, uz, T, p, rank, frob_norm_iter, frob_norm_rhs)

    return S
