from mhd5solverv3 import Const, MHDSystem, MHDEquilibrium, LinearizedMHD, MHDEvolution
import numpy as np
import matplotlib.pyplot as plt

sys = MHDSystem(N_r=200, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0.01, D_P=0, B_Z0=0)
equ = MHDEquilibrium(sys, p_exp=4)
lin = LinearizedMHD(equ, k=1, m=0)
# # # 
lin.solve(num_modes=1)
# lin.plot_eigenvalues()
lin.plot_VB(-1, epsilon=0.05)
# lin.plot_EJ(-1, epsilon=0.05)

# i = find_nearest(sys.grid.rr, 1)
# g = equ.B**2 / (8 * np.pi * equ.rho)
# print(g[i])
# plt.plot(sys.grid.r[1:-1], equ.p[1:-1], sys.grid.r[1:-1], equ.B[1:-1], sys.grid.r[1:-1], equ.J[1:-1])
# plt.title('Equilibrium configuration')
# plt.xlabel('r/r0')
# plt.legend(['P', 'B', 'J'])
# plt.show()

# nonlin = MHDEvolution(equ, t_max=1500)
# nonlin.evolve(k=1)

def find_nearest(array, value): return (np.abs(array - value)).argmin()



# i = np.argmax(equ.B)
# g = equ.B[i]**2 / (8 * np.pi * 0.55 * sys.grid.rr[i])
# print(g)

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
# pexp_vals = range(4, 17, 1)
# gammas = []
# l_char = []
# gs = []
# i = find_nearest(sys.grid.rr, 1) 
# for pexp in pexp_vals:
#     print(pexp)
#     equ = MHDEquilibrium(sys, pexp)
#     gs.append(equ.B[i]**2 / (4 * np.pi * equ.rho[i]))
#     lin = LinearizedMHD(equ, k=50)
#     lin.set_z_mode(k=50, m=0)
#     gammas.append(lin.solve_for_gamma())
#     l_char.append((find_nearest(equ.p, 0.15) - find_nearest(equ.p, 0.95)) * sys.grid.dr)
# 
# l_char = np.reshape(l_char, (13, 1))
# gs = np.reshape(gs, (13, 1))
# product = l_char * np.square(gammas) / gs
# 
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# 
# ax.scatter(l_char, np.square(gammas), s=2, label='gamma^2')
# ax.scatter(l_char, gs / l_char, s=2, label='l_char * gamma^2 / g')
# plt.title('Asymptotic growth rates')
# plt.xlabel('Characteristic gradient length')
# plt.legend(loc='upper right')
# # plt.ylabel('gamma^2')
# plt.show()
# 
# gammas2 = np.square(gammas)
# gr1 = gs / l_char
# np.savetxt('l_char.csv', l_char, delimiter=',')
# np.savetxt('gr1.csv', gr1, delimiter=',')
# np.savetxt('gammas2.csv', gammas2, delimiter=',')
# print('Finished exporting.')



## gamma vs. k
## Need to increase resolution to find gammas for very low k.
# k_vals = np.reshape(np.linspace(0.05, 0.2, 5), (5, 1))
# gammas0 = []
# gammas1 = []
# gammas2 = []
# 
# sys = MHDSystem(N_r=400, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1, m=0)
# 
# i = find_nearest(sys.grid.rr, 1)
# g = equ.B[i]**2 / (4 * np.pi * equ.rho[i])
# print(g)
# 
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k, m=0)
#     gammas0.append(lin.solve_for_gamma())
# 
# sys = MHDSystem(N_r=400, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0.008, D_P=0.1, B_Z0=0)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1, m=0)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k, m=0)
#     gammas1.append(lin.solve_for_gamma())
#     
# sys = MHDSystem(N_r=400, N_ghost=1, r_max=2*np.pi, D_eta=1, D_H=0.008, D_P=0.1, B_Z0=0)
# equ = MHDEquilibrium(sys, p_exp=4)
# lin = LinearizedMHD(equ, k=1, m=0)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k, m=0)
#     gammas2.append(lin.solve_for_gamma())
#     
# plt.plot(k_vals, np.square(gammas0), k_vals, np.square(gammas1), k_vals, np.square(gammas2), k_vals, k_vals * g)
# plt.title('Fastest growing mode')
# plt.xlabel('k')
# plt.ylabel('gamma')
# plt.legend(['D_eta = 0, D_H = 0, D_P = 0', 'D_eta = 0, D_H = 0.008, D_P = 0.1', 'D_eta = 1, D_H = 0.008, D_P = 0.1', 'gk'])
# plt.show()




## gamma vs. B_Z0    
# B_Z0_vals = np.linspace(0, 2.5, 20)
# gammas = []
# for B_Z in B_Z0_vals:
#     print(B_Z)
#     sys = MHDSystem(N_r=400, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=B_Z)
#     equ = MHDEquilibrium(sys, p_exp=4)
#     lin = LinearizedMHD(equ, k=1, m=0)
#     
#     lin.set_z_mode(k=1, m=0)
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
#     print(N)
#     sys = MHDSystem(N_r=N, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0)
#     equ = MHDEquilibrium(sys, p_exp=4)
#     lin = LinearizedMHD(equ, k=1)
#     gammas.append(lin.solve_for_gamma())
# 
# plt.plot(nvals, gammas, '.-')
# plt.title('Convergence')
# plt.xlabel('Grid points')
# plt.ylabel('gamma')
# plt.show()


