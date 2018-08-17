import numpy as np
from mhdsystem import System, Plot
from mhdcartesian import NumericalEquilibriumCar, LinearCar, EvolveCar
import matplotlib.pyplot as plt

sys  = System(n_ghost=1, resR=200, r_max=2*np.pi, resZ=200, z_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
rho0 = 0.5 * (1 - np.tanh((sys.grid_r.r - np.pi) / 0.5)) + 0.02
equ  = NumericalEquilibriumCar(sys, rho0, g=1)

plt.plot(sys.grid_r.r, equ.rho, sys.grid_r.r, equ.p,
         sys.grid_r.r, equ.t, sys.grid_r.r, 0.1 * equ.b)
plt.legend(['rho', 'P', 'T', 'B'])
plt.show()

bc_array = {'rho': ['value', 'value'],
            'Btheta': ['value', 'value'],
            'Vr': ['derivative', 'derivative'],
            'Vz': ['derivative', 'derivative'],
            'p': ['value', 'value'],
            }

# bc_array = {'rho': ['value', 'value'],
#             'Btheta': ['value', 'value'],
#             'Vr': ['exp_inc', 'exp_dec'],
#             'Vz': ['exp_inc', 'exp_dec'],
#             'p': ['value', 'value'],
#             }
#
# bc_array = {'rho': ['value', 'value'],
#             'Btheta': ['value', 'value'],
#             'Vr': ['exp_dec', 'exp_inc'],
#             'Vz': ['exp_dec', 'exp_inc'],
#             'p': ['value', 'value'],
#             }

# lin = LinearCar(equ, k=1, m=0, rosh=2, bc_array=bc_array)
# evo = EvolveCar(sys, equ, lin, k=1, rosh=2, D_nu=0)

# lin.solve(num_modes=3)
# evo.evolve_Euler(courant=0.8, t_max=0.0001)
# evo.evolve_WENO(courant=0.8, t_max=0)

# plots = Plot(sys, equ, None, None)
# plots.plot_equilibrium()
# plots.plot_eigenvalues()
# plots.plot_VB(-1, epsilon=0.05)
# plots.plot_VB(-2, epsilon=0.05)
# plots.plot_VB(-3, epsilon=0.05)
# plots.plot_EJ()
# plots.plot_evolution()








