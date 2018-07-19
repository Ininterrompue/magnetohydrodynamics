from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

sys = MHDSystem(N_r=400, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=100, B_Z0=0)
equ = MHDEquilibrium(sys, p_exp=4)
lin = LinearizedMHD(equ, k=1)

lin.solve(num_modes=1)
# lin.plot_eigenvalues()
lin.plot_mode(-1)

## Exact solution comparison for p_exp = 4
# B_exact_2 = np.sqrt(np.pi) / sys.grid.rr**2 * erf(sys.grid.rr**2) - 2 * np.exp(-sys.grid.rr**4)
# B_exact = np.sqrt(4 * np.pi) * np.sign(B_exact_2) * np.sqrt(np.abs(B_exact_2))

def find_nearest(array, value): return (np.abs(array - value)).argmin()

# print(sys.grid.dr * find_nearest(equ.p, 0.5))
# plt.plot(sys.grid.r[1:-1], equ.p[1:-1], sys.grid.r[1:-1], equ.B[1:-1])
# plt.legend(['P', 'B'])
# plt.show()

########################
# sys = MHDSystem(N_r=100, r_max=2*np.pi, D_eta=1e-3, D_H=1e-3, D_P=0, B_Z0=0)
# pressure = np.exp(-sys.grid.rr**4) + 0.05
# sys = MHDSystem(N_r=100, r_max=3, D_eta=1e-3, D_H=1e-3, D_P=0, B_Z0=0)
# #pressure = np.exp(-sys.grid.rr**15) + 0.05
# pressure = 0.5*(np.tanh( 10*(1-sys.grid.rr) )+1)
# equ = MHDEquilibrium(sys, pressure)
# 
# plt.plot(sys.grid.rr, equ.p,
#          sys.grid.rr, equ.B)
# plt.show()
# 
# test = 0 * sys.grid.rr
# dp = 0.5*10* np.tanh( 10*(1-sys.grid.rr) )**2 - 5
# for i in range(sys.grid.N):
#     test = test - sys.grid.rr[i]**2 * dp[i] * (sys.grid.rr > sys.grid.rr[i])/(sys.grid.rr**2)
# plt.plot(sys.grid.rr, equ.p,
#          sys.grid.rr, equ.B,
#          sys.grid.rr, np.sqrt(test) )
# plt.show()
# 
# lin = LinearizedMHD(equ, k=1)
########################

# Asymptotic gamma routine
# pexp_vals = range(1, 16, 1)
# gammas = []
# for pexp in pexp_vals:
#     print(pexp)
#     equ = MHDEquilibrium(sys, pexp)
#     lin = LinearizedMHD(equ, k=50)
#     lin.set_z_mode(k=50)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(pexp_vals, gammas, '.-')
# plt.title('Asymptotic growth rates')
# plt.xlabel('Pressure exponent')
# plt.ylabel('gamma')
# plt.show()



# Asymptotic gamma routine
# pexp_vals = range(1, 16, 1)
# gammas = []
# l_char = []
# for pexp in pexp_vals:
#     print(pexp)
#     equ = MHDEquilibrium(sys, pexp)
#     lin = LinearizedMHD(equ, k=50)
#     lin.set_z_mode(k=50)
#     gammas.append(lin.solve_for_gamma())
#     l_char.append((find_nearest(equ.p, 0.15) - find_nearest(equ.p, 0.95)) * sys.grid.dr)
# 
# plt.scatter(l_char, gammas, s=1)
# plt.title('Asymptotic growth rates')
# plt.xlabel('Characteristic gradient length')
# plt.ylabel('gamma')
# plt.show()


## gamma vs. k
## Need to increase resolution to find gammas for very low k.
# k_vals = np.reshape(np.linspace(0.01, 10, 30), (30, 1))
# gammas_k0 = []
# gammas_k05 = []
# gammas_k1 = []
# 
# sys = MHDSystem(N_r=200, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k)
#     gammas_k0.append(lin.solve_for_gamma())
# 
# sys = MHDSystem(N_r=200, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0.5)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k)
#     gammas_k05.append(lin.solve_for_gamma())
#     
# sys = MHDSystem(N_r=200, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=1)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k)
#     gammas_k1.append(lin.solve_for_gamma())
# 
# plt.plot(k_vals, gammas_k0, k_vals, gammas_k05, k_vals, gammas_k1)
# plt.title('Fastest growing mode')
# plt.xlabel('k')
# plt.ylabel('gamma')
# plt.legend(['B_Z0 = 0', 'B_Z0 = 0.5', 'B_Z0 = 1'])
# plt.show()


## gamma vs. B_Z0    
# B_Z0_vals = np.linspace(0, 2.5, 30)
# gammas = []
# for B_Z in B_Z0_vals:
#     print(B_Z)
#     sys = MHDSystem(N_r=400, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=B_Z)
#     equ = MHDEquilibrium(sys, p_exp=4)
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
# nvals = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800]
# gammas = []
# for N in nvals:
#     sys = MHDSystem(N_r=N, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
#     equ = MHDEquilibrium(sys, p_exp=15)
#     lin = LinearizedMHD(equ, k=50)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(nvals, gammas, '.-')
# plt.title('Convergence')
# plt.xlabel('Grid points')
# plt.ylabel('gamma')
# plt.show()


