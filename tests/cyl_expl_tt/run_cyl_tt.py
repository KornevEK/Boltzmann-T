from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time
import tt

from read_starcd import Mesh
from read_starcd import write_tecplot

import solver_tt as Boltzmann_cyl
import pickle

log = open('log.txt', 'w+') #log file

# compute parameters for flow around cylinder

# Parameters for argon (default)
gas_params = Boltzmann_cyl.GasParams()

Mach = 10.
Kn = 0.564
delta = 8.0 / (5 * np.pi**0.5 * Kn)
n_l = 2e+23
T_l = 200.
u_l = Mach * ((gas_params.g * gas_params.Rg * T_l) ** 0.5)
T_w = 5.0 * T_l
r = 1e-7

n_s = n_l
T_s = T_l
   
p_s = gas_params.m * n_s * gas_params.Rg * T_s
    
v_s = np.sqrt(2. * gas_params.Rg * T_s)
mu_s = gas_params.mu(T_s)

l_s = delta * mu_s * v_s / p_s

#print 'l_s = ', l_s 

#print 'v_s = ', v_s

nv = 64
vmax = 22 * v_s

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes
    
vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')

f_init = lambda x, y, z, vx, vy, vz: tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_l, n_l, u_l, 0., 0., gas_params.Rg))
f_bound = tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_l, n_l, u_l, 0., 0., gas_params.Rg))
#print(f_bound)
fmax = tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_w, 1., 0., 0., 0., gas_params.Rg))
#print(fmax)
problem = Boltzmann_cyl.Problem(bc_type_list = ['sym-z', 'in', 'out', 'wall', 'sym-y'],
                                bc_data = [[],
                                           [f_bound],
                                           [f_bound],
                                           [fmax],
                                           []], f_init = f_init)


#print 'vmax =', vmax 

CFL = .5

f = open('../mesh-cyl/mesh-cyl.pickle', 'rb')

mesh = pickle.load(file = f)

f.close()

log = open('log.txt', 'a')
log.write('Mach  = ' + str(Mach) + '\n')
log.close()

nt = 2000
t1 = time.time()
S = Boltzmann_cyl.solver_tt(gas_params, problem, mesh, nt, nv, vx_, vx, vy, vz, 
                            CFL, r, filename = 'file-out.npy') #, init = 'macro_restart.txt') # restart from macroparameters array
t2 = time.time()

log = open('log.txt', 'a')
log.write('Time  = ' + str(t2 - t1) + '\n')
log.close()

fig, ax = plt.subplots(figsize = (20,10))
line, = ax.semilogy(S.frob_norm_iter/S.frob_norm_iter[0])
ax.set(title='$Steps =$' + str(nt))
plt.savefig('norm_iter.png')
plt.close()

data = np.zeros((mesh.nc, 7))
    
data[:, 0] = S.n[:]
data[:, 1] = S.ux[:]
data[:, 2] = S.uy[:]
data[:, 3] = S.uz[:]
data[:, 4] = S.p[:]
data[:, 5] = S.T[:]
data[:, 6] = S.rank[:]

np.savetxt('macroparameters_data.txt', data) # save macroparameters

write_tecplot(mesh, data, 'tec_tt.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))

log = open('log.txt', 'a')
log.write('Residual = ' + str('{0:5.2e}'.format(S.frob_norm_iter[-1]/S.frob_norm_iter[0])) + '\n')
log.close()
