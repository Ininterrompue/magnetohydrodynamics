import numpy as np
from mhdsystem import System, Plot
from mhdcylindrical import AnalyticalEquilibriumCyl, LinearCyl


sys = System(n_ghost=1, resR=256, r_max=2*np.pi, resZ=256, z_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0, V_Z0=0.015)
equ_a = AnalyticalEquilibriumCyl(sys, p_exp=4, v_exp=4)
# equ_n = NumericalEquilibriumCyl(sys, p_exp=4)
lin = LinearCyl(equ_a, k=1, m=0, rosh=5/3)
evo = None  # EvolveCyl(sys, equ_a, lin, k=1, rosh=5/3, D_nu=0)

# lin.solve()
# lin.remove_oscillatory()
lin.solve(num_modes=1)
# evo.evolve_Euler(courant=0.8, t_max=0)
# evo.evolve_WENO(courant=0.8, t_max=0)

plots = Plot(sys, equ_a, lin, evo)
# plots.plot_equilibrium()
# plots.plot_eigenvalues()
plots.plot_VB()
# plots.plot_EJ()
# plots.plot_evolution()
