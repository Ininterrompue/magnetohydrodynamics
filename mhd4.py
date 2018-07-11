from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

sys = MHDSystem(N_r=200, r_max=5)
pressure = np.exp(-sys.grid.rr**4) + 0.05
equ = MHDEquilibrium(sys, pressure)

lin = LinearizedMHD(equ, k=1, D_eta=0, D_H=0, D_P=0, B_Z0=0)
lin.solve()
lin.plot_eigenvalues()
lin.plot_mode(-1)

# lin.fastest_mode()

# kvals = [0.1, 0.5, 1, 1.5, 2]
# evals_v_k = []
# 
# for k in kvals:
#     lin.set_z_mode(k)
#     lin.solve()
#     evals_v_k.append(lin.fastest_mode())
# 
# plt.plot(kvals, evals_v_k)
# plt.show()
# 
# nvals = [10, 50, 100, 200]
# evals = []
# for N in nvals:
#     print(N)
#     sys = MHDSystem(N_r=N, r_max=5)
#     pressure = np.exp(-sys.grid.rr**4) + 0.05
#     equ = MHDEquilibrium(sys, pressure)
#     lin = LinearizedMHD(equ, k=2)
#     lin.solve()
#     evals.append(lin.fastest_mode())
# 
# plt.plot(nvals, evals)
# plt.show()


