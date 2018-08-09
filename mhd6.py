from mhd6solver import Const, MHDSystem, MHDEquilibrium0, LinearizedMHD, CartesianEquilibrium, FDSystem
#from mhdweno2 import MHDGrid, MHDEquilibrium, MHDEvolution
import numpy as np
import matplotlib.pyplot as plt

sys = MHDSystem(N_r=256, N_ghost=1, r_max=6, g=0.1, D_eta=1e-4, D_H=0, D_P=0, B_Z0=0)

rho0 = 0.5 * (1 - np.tanh((sys.grid.r - 3) / 0.1)) + 0.02
plt.plot(sys.grid.r, rho0)
plt.show()

equ0 = CartesianEquilibrium(sys, rho0)

plt.plot(sys.grid.r, equ0.rho, sys.grid.r, equ0.p,
         sys.grid.r, equ0.t, sys.grid.r, equ0.B)
plt.legend(['rho', 'P', 'T', 'B'])
plt.show()

lin = LinearizedMHD(equ0, k=.1, m=0)

lin.solve()
lin.plot_VB(-1, epsilon=0.05)
lin.plot_eigenvalues()

# # #
# lin.solve(num_modes=None)
# lin.plot_eigenvalues()
# lin.plot_VB(-5, epsilon=0.05)
# # lin.plot_EJ(-1, epsilon=0.05)
#
# nonlin = MHDEvolution(lin, equ0, t_max=0.0001)
# nonlin.evolve(k=1)
#
# grid_r = MHDGrid(res=64, n_ghost=3, r_max=2*np.pi)
# grid_z = MHDGrid(res=64, n_ghost=3, r_max=2*np.pi)
# equ = MHDEquilibrium(grid_r, p_exp=4)
# evo = MHDEvolution(lin, grid_r, grid_z, equ, k=1, rosh=2, D_eta=0, D_nu=0)
# evo.evolve(courant=0.5, t_max=200)
# evo.plot_lineouts()
# evo.plot_evolved()

def find_nearest(array, value): return (np.abs(array - value)).argmin()

# i = find_nearest(sys.grid.rr, 1)
# g = equ.B**2 / (4 * np.pi * equ.rho)
# print(g[i])
plt.plot(sys.grid.r[1:-1], equ0.p[1:-1], sys.grid.r[1:-1], equ0.B[1:-1], sys.grid.r[1:-1], equ0.J[1:-1])#, sys.grid.r[1:-1], g[1:-1])
plt.title('Equilibrium configuration')
plt.xlabel('x/x0')
plt.legend(['P', 'B', 'J'])#, 'g'])
plt.show()

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
# gs = np.ones(13)
# gs = np.reshape(gs, (13, 1))
# product = l_char * np.square(gammas) / gs
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
#
# ax.scatter(l_char, np.square(gammas), s=2, label='gamma^2')
# ax.scatter(l_char, product, s=2, label='l_char * gamma^2 / g')
# plt.title('Asymptotic growth rates')
# plt.xlabel('Characteristic gradient length')
# plt.legend(loc='upper right')
# # plt.ylabel('gamma^2')
# plt.show()


## gamma vs. k
k_vals = np.linspace(0.01, 1, 10)
gammas = np.zeros(k_vals.shape)
sys = MHDSystem(N_r=128, N_ghost=1, r_max=6, g=0.1, D_eta=1e-4, D_H=0, D_P=0, B_Z0=0)
rho0 = 0.5 * (1 - np.tanh((sys.grid.r - 3) / 0.1)) + 0.05
equ0 = CartesianEquilibrium(sys, rho0)

for i, k in enumerate(k_vals):
    print(k)
    lin = LinearizedMHD(equ0, k=k, m=0)
    lin.set_z_mode(k, m=0)
    gammas[i] = lin.solve_for_gamma()
    # lin.solve()
    # evals = np.imag(lin.evals)
    # gammas[i] = np.max(evals)

plt.plot(k_vals, gammas, k_vals, 0.36*k_vals**0.5)
plt.title('Fastest growing mode')
plt.xlabel('k')
plt.ylabel('gamma')
plt.legend(['gamma', 'gk'])
plt.show()

## gamma vs. g
g_vals = np.linspace(-10, 10, 20)
gammas = np.zeros(g_vals.shape)

for i,G in enumerate(g_vals):
    print(G)
    sys = MHDSystem(N_r=64, N_ghost=1, r_max=4, g=G, D_eta=0, D_H=0, D_P=0, B_Z0=0)
    p0 = 0.5 * (1 - np.tanh((sys.grid.r - 2) / 0.1)) + 0.01
    equ0 = CartesianEquilibrium(sys, p0)

    plt.plot(sys.grid.r, equ0.p, sys.grid.r, equ0.B, sys.grid.r,
             equ0.J)
    plt.title('Equilibrium configuration')
    plt.xlabel('x/x0')
    plt.legend(['P', 'B', 'J'])
    plt.show()


    lin = LinearizedMHD(equ0, k=1, m=0)
    lin.set_z_mode(k=1, m=0)
    gammas[i] = lin.solve_for_gamma()

plt.plot(g_vals, gammas)
plt.title('Fastest growing mode')
plt.xlabel('g')
plt.ylabel('gamma')
plt.show()


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


