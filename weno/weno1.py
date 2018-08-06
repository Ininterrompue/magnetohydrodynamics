from mhdweno import MHDGrid, MHDEquilibrium, MHDEvolution
import numpy as np
import matplotlib.pyplot as plt

grid_r = MHDGrid(res=96, n_ghost=3, r_max=np.pi)
grid_z = MHDGrid(res=192, n_ghost=3, r_max=2*np.pi)

equ = MHDEquilibrium(grid_r, p_exp=4)
# equ.plot_equilibrium()

evo = MHDEvolution(grid_r, grid_z, equ, k=1, rosh=2, D_eta=0, D_nu=0)
evo.evolve(courant=0.8, t_max=0.0001)
# evo.plot_lineouts()
# evo.plot_evolved()
