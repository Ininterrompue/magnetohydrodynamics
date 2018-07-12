from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt

sys = MHDSystem(N_r=200, r_max=2*np.pi)
pressure = np.exp(-sys.grid.rr**4) + 0.05
equ = MHDEquilibrium(sys, pressure)

lin = LinearizedMHD(equ, k=1, D_eta=1e-6, D_H=0, D_P=0, B_Z0=0.5)
lin.solve()
lin.plot_eigenvalues()
lin.plot_mode(-1)

# kvals = [0.1, 0.5, 1, 1.5, 2]
# gammas = []
# for k in kvals:
#     lin.set_z_mode(k, D_eta=1, D_H=0, D_P=0, B_Z0=0)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(kvals, gammas)
# plt.title('Growth rates')
# plt.xlabel('k')
# plt.ylabel('gamma')
# plt.show()
# 
# nvals = [50, 100, 200, 300, 400]
# gammas = []
# for N in nvals:
#     sys = MHDSystem(N_r=N, r_max=5)
#     pressure = np.exp(-sys.grid.rr**4) + 0.05
#     equ = MHDEquilibrium(sys, pressure)
#     lin = LinearizedMHD(equ, k=4, D_eta=1, D_H=0, D_P=0, B_Z0=0.5)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(nvals, gammas)
# plt.title('Convergence')
# plt.xlabel('Resolution')
# plt.ylabel('gamma')
# plt.show()


