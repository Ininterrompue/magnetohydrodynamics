from mhd4solver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt

sys = MHDSystem(N_r=100, r_max=2*np.pi, D_eta=1e-3, D_H=1e-3, D_P=0, B_Z0=0)
pressure = np.exp(-sys.grid.rr**4) + 0.05
sys = MHDSystem(N_r=100, r_max=3, D_eta=1e-3, D_H=1e-3, D_P=0, B_Z0=0)
#pressure = np.exp(-sys.grid.rr**15) + 0.05
pressure = 0.5*(np.tanh( 10*(1-sys.grid.rr) )+1)
equ = MHDEquilibrium(sys, pressure)

plt.plot(sys.grid.rr, equ.p,
         sys.grid.rr, equ.B)
plt.show()

test = 0 * sys.grid.rr
dp = 0.5*10* np.tanh( 10*(1-sys.grid.rr) )**2 - 5
for i in range(sys.grid.N):
    test = test - sys.grid.rr[i]**2 * dp[i] * (sys.grid.rr > sys.grid.rr[i])/(sys.grid.rr**2)
plt.plot(sys.grid.rr, equ.p,
         sys.grid.rr, equ.B,
         sys.grid.rr, np.sqrt(test) )
plt.show()
lin = LinearizedMHD(equ, k=1)
lin.solve(num_modes=1)
lin.plot_eigenvalues()
lin.plot_mode(-1)

# k_vals = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
# gammas = []
# for k in k_vals:
#     lin.set_z_mode(k, D_eta=0, D_H=0.1, D_P=0, B_Z0=0)
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


