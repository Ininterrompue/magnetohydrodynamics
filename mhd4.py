from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt

# sys = MHDSystem(N_r=100, r_max=2+0.5*np.pi, D_eta=1e-6, D_H=0, D_P=0, B_Z0=0)
# pressure = np.exp(-(sys.grid.rr / 0.5)**4) + 0.05
# equ = MHDEquilibrium(sys, pressure)
# 
# lin = LinearizedMHD(equ, k=1)
# lin.solve(num_modes=1)
# # lin.plot_eigenvalues()
# lin.plot_mode(-1)

# r0_vals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
r0_vals = np.reshape(np.linspace(1, 10, 50), (50, 1))
print(r0_vals)
gammas = []
for r0 in r0_vals:
    sys = MHDSystem(N_r=100, r_max=2+r0*np.pi, D_eta=1e-6, D_H=0, D_P=0, B_Z0=0)
    pressure = np.exp(-(sys.grid.rr / r0)**4) + 0.05
    equ = MHDEquilibrium(sys, pressure)
    lin = LinearizedMHD(equ, k=50)
    lin.set_z_mode(k=50)
    gammas.append(lin.solve_for_gamma())
    
plt.plot(r0_vals, gammas, r0_vals, np.pi / r0_vals)
plt.title('Asymptotic growth rates vs. characteristic gradient length')
plt.xlabel('r0')
plt.ylabel('gamma')
plt.show()

# k_vals = [0.5, 1, 1.5, 2, 2.5, 3]
# gammas = []
# for k in k_vals:
#     lin.set_z_mode(k)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(k_vals, gammas)
# plt.title('Growth rates')
# plt.xlabel('k')
# plt.ylabel('gamma')
# plt.show()
    
# B_Z0_vals = np.linspace(0, 0.8, 18)
# gammas = []
# for B_Z in B_Z0_vals:
#     lin.set_z_mode(k=1, D_eta=2, D_H=0, D_P=0, B_Z0=B_Z)
#     gammas.append(lin.solve_for_gamma())
#     
# plt.plot(B_Z0_vals, gammas)
# plt.title('Growth rates')
# plt.xlabel('B_Z0')
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


