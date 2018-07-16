from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

sys = MHDSystem(N_r=400, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
# equ = MHDEquilibrium(sys, p_exp=4)

# B_exact_2 = np.sqrt(np.pi) / sys.grid.rr**2 * erf(sys.grid.rr**2) - 2 * np.exp(-sys.grid.rr**4)
# B_exact = np.sqrt(4 * np.pi) * np.sign(B_exact_2) * np.sqrt(np.abs(B_exact_2))
# plt.plot(sys.grid.r, equ.p, sys.grid.r, equ.B)
# plt.show()

# lin = LinearizedMHD(equ, k=1)
# lin.solve(num_modes=1)
# # lin.plot_eigenvalues()
# lin.plot_mode(-1)


# Asymptotic gamma routine
pexp_vals = range(1, 16, 1)
gammas = []
for pexp in pexp_vals:
    equ = MHDEquilibrium(sys, pexp)
    lin = LinearizedMHD(equ, k=50)
    lin.set_z_mode(k=50)
    gammas.append(lin.solve_for_gamma())
    
plt.plot(pexp_vals, gammas)
plt.title('Asymptotic growth rates')
plt.xlabel('Pressure exponent')
plt.ylabel('gamma')
plt.show()


## gamma vs. k
# Need to increase resolution to find gammas for very low k.
# k_vals = np.reshape(np.linspace(0.01, 0.2, 10), (10, 1))
# gammas_k = []
# for k in k_vals:
#     lin.set_z_mode(k)
#     gammas_k.append(lin.solve_for_gamma())
# 
# plt.plot(k_vals, gammas_k)
# plt.title('Growth rates')
# plt.xlabel('k')
# plt.ylabel('gamma')
# plt.show()


## gamma vs. B_Z0    
# B_Z0_vals = np.linspace(0, 0.8, 20)
# gammas = []
# for B_Z in B_Z0_vals:
#     sys = MHDSystem(N_r=100, r_max=2*np.pi, D_eta=1e-6, D_H=0, D_P=0, B_Z0=B_Z)
#     pressure = np.exp(-(sys.grid.rr)**4) + 0.05
#     equ = MHDEquilibrium(sys, pressure)
#     lin = LinearizedMHD(equ, k=1)
#     
#     lin.set_z_mode(k=1)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(B_Z0_vals, np.square(gammas))
# plt.title('Growth rates')
# plt.xlabel('B_Z0')
# plt.ylabel('gamma^2')
# plt.show()


## Convergence
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


