from mhdsolver import MHDSystem, MHDEquilibrium
import numpy as np
import matplotlib.pyplot as plt

sys = MHDSystem(N_r=100, r_max=5)
pressure = np.exp(-sys.grid.r**4) + 0.05
equ = MHDEquilibrium(sys, pressure)

plt.plot(sys.grid.r, equ.p, sys.grid.r, equ.B, sys.grid.r, equ.J)
plt.show()

'''pert = MHDPerturbation(sys, pressure)
pert.plot_eigenvalues()
pert.plot_parameters()'''