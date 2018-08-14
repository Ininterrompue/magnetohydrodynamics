from mhdsolver import MHDSystem, AnalyticalEquilibrium, LinearizedMHD,\
    CartesianEquilibrium, FDSystem, MHDEquilibrium, AnalyticalEquilibriumCylindrical
#from mhdweno2 import MHDGrid, MHDEquilibrium, MHDEvolution
import numpy as np
import matplotlib.pyplot as plt

# Cartesian RT example
sys = MHDSystem(N_r=64, N_ghost=1, r_max=32, g=1, D_eta=0, D_H=0, D_P=0, B_Z0=0)
rho0 = 0.5 * (1 - np.tanh((sys.grid.r - 16) / 0.5)) + 0.02
equ0 = CartesianEquilibrium(sys, rho0)

plt.plot(sys.grid.r, equ0.rho, sys.grid.r, equ0.p,
         sys.grid.r, equ0.t, sys.grid.r, 0.1 * equ0.B)
plt.legend(['rho', 'P', 'T', 'B'])
plt.show()


bc_array = {'rho': ['value', 'value'],
            'Btheta': ['value', 'value'],
            'Vr': ['derivative', 'derivative'],
            'Vz': ['derivative', 'derivative'],
            'p': ['value', 'value'],
            }


# bc_array = {'rho': ['value', 'value'],
#             'Btheta': ['value', 'value'],
#             'Vr': ['exp_inc', 'exp_dec'],
#             'Vz': ['exp_inc', 'exp_dec'],
#             'p': ['value', 'value'],
#             }
#
# bc_array = {'rho': ['value', 'value'],
#             'Btheta': ['value', 'value'],
#             'Vr': ['exp_dec', 'exp_inc'],
#             'Vz': ['exp_dec', 'exp_inc'],
#             'p': ['value', 'value'],
#             }
#


lin = LinearizedMHD(equ0, k=10, m=0, bc_array=bc_array)
lin.solve(num_modes=3)
lin.plot_VB(-1, epsilon=0.05)
lin.plot_VB(-2, epsilon=0.05)
lin.plot_VB(-3, epsilon=0.05)



# Cylindrical RT example
sys = MHDSystem(N_r=64, N_ghost=1, r_max=3, g=1, D_eta=0, D_H=0, D_P=0, B_Z0=0, geom='cylindrical')
p0 = 1 * (0.05 + np.exp(-(sys.grid.r) ** 4))
equ = MHDEquilibrium(sys, p0)

equ = AnalyticalEquilibriumCylindrical(sys, 4)


plt.plot(sys.grid.r, equ.rho, sys.grid.r, equ.p,
         sys.grid.r, equ.t, sys.grid.r, equ.B)
plt.legend(['rho', 'P', 'T', 'B'])
plt.show()

bc_array = {'rho': ['value', 'value'],
            'Btheta': ['value', 'value'],
            'Vr': ['derivative', 'derivative'],
            'Vz': ['derivative', 'derivative'],
            'p': ['value', 'value'],
            }
lin = LinearizedMHD(equ, k=10, m=0, bc_array=bc_array)
lin.solve(num_modes=3)
lin.plot_VB(-1, epsilon=0.05)
lin.plot_VB(-2, epsilon=0.05)
lin.plot_VB(-3, epsilon=0.05)



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
# plt.plot(sys.grid.r[1:-1], equ0.p[1:-1], sys.grid.r[1:-1], equ0.B[1:-1], sys.grid.r[1:-1], equ0.J[1:-1])#, sys.grid.r[1:-1], g[1:-1])
# plt.title('Equilibrium configuration')
# plt.xlabel('x/x0')
# plt.legend(['P', 'B', 'J'])#, 'g'])
# plt.show()


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
sys = MHDSystem(N_r=256, N_ghost=1, r_max=16, g=1, D_eta=0, D_H=0, D_P=0, B_Z0=0)
rho0 = 0.5 * (1 - np.tanh((sys.grid.r - 8) / 0.5)) + 0.02
equ0 = CartesianEquilibrium(sys, rho0)

sig0 = 1.5j
for i, K in enumerate(k_vals):
    print(K)
    lin = LinearizedMHD(equ0, k=K, m=0)
    lin.set_z_mode(K, m=0)
    gammas[i] = lin.solve_for_gamma(sigma=sig0)
    sig0 = gammas[i] * 1j + 0.5j
    # lin.solve()
    # evals = np.imag(lin.evals)
    # gammas[i] = np.max(evals)

plt.plot(k_vals, gammas, k_vals, k_vals**0.5)
plt.title('Fastest growing mode')
plt.xlabel('k')
plt.ylabel('gamma')
plt.legend(['gamma', 'gk'])
plt.show()


## gamma vs. k with hall
k_vals = np.linspace(0.01, 1, 10)
gammas_hall = np.zeros(k_vals.shape)
sys = MHDSystem(N_r=256, N_ghost=1, r_max=16, g=1, D_eta=0, D_H=0.2, D_P=0, B_Z0=0)
rho0 = 0.5 * (1 - np.tanh((sys.grid.r - 8) / 0.5)) + 0.02
equ0 = CartesianEquilibrium(sys, rho0)

sig0 = 1.5j
for i, K in enumerate(k_vals):
    print(K)
    lin = LinearizedMHD(equ0, k=K, m=0)
    lin.set_z_mode(K, m=0)
    gammas_hall[i] = lin.solve_for_gamma(sigma=sig0)
    sig0 = gammas_hall[i]
    # lin.solve()
    # evals = np.imag(lin.evals)
    # gammas[i] = np.max(evals)

plt.plot(k_vals, gammas**2, k_vals, gammas_hall**2, k_vals, k_vals)
plt.title('Fastest growing mode')
plt.xlabel('k')
plt.ylabel('gamma^2')
plt.legend(['gamma','gamma_Hall', 'gk'])
plt.show()





## gamma vs. g
g_vals = np.linspace(0.05, 1, 20)
gammas = np.zeros(g_vals.shape)

for i,G in enumerate(g_vals):
    print(G)
    sys = MHDSystem(N_r=64, N_ghost=1, r_max=15, g=G, D_eta=0, D_H=0, D_P=0, B_Z0=0)
    p0 = 0.5 * (1 - np.tanh((sys.grid.r - 5) / 0.5)) + 0.02
    equ0 = CartesianEquilibrium(sys, p0)

#     plt.plot(sys.grid.r, equ0.p, sys.grid.r, equ0.B, sys.grid.r,
#              equ0.J)
#     plt.title('Equilibrium configuration')
#     plt.xlabel('x/x0')
#     plt.legend(['P', 'B', 'J'])
#     plt.show()


    lin = LinearizedMHD(equ0, k=1, m=0)
    lin.set_z_mode(k=1, m=0)
    gammas[i] = lin.solve_for_gamma()

plt.plot(g_vals, np.square(gammas), g_vals, 1 * g_vals)
plt.title('Fastest growing mode')
plt.xlabel('g')
plt.ylabel('gamma^2')
plt.legend(['gamma^2', 'gk'])
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


