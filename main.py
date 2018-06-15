from mhdsolver import MHDSystem, MHDEquilibrium, LinearizedMHD
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

sys = MHDSystem(N_r=100, r_max=5)
pressure = np.exp(-sys.grid.r**4) + 0.05
equ = MHDEquilibrium(sys, pressure)

plt.plot(sys.grid.r[1:-1], equ.p[1:-1], sys.grid.r[1:-1], equ.B[1:-1], sys.grid.r[1:-1], equ.J[1:-1])
plt.show()

lin = LinearizedMHD(equ)
lin.solve()
lin.plot_eigenvalues()
lin.plot_mode(-1)

lin.fastest_mode()

kvals = [0.1, 0.5, 1, 1.5, 2]
evals_v_k = []

for k in kvals:
    lin.set_z_mode(k)
    lin.solve()
    evals_v_k.append(lin.fastest_mode())

plt.plot(kvals, evals_v_k)
plt.show()

nvals = [10, 50, 100, 200]
evals = []
for N in nvals:
    print(N)
    sys = MHDSystem(N_r=N, r_max=5)
    pressure = np.exp(-sys.grid.r ** 4) + 0.05
    equ = MHDEquilibrium(sys, pressure)
    lin = LinearizedMHD(equ, k=2)
    lin.solve()
    evals.append(lin.fastest_mode())

plt.plot(nvals, evals)
plt.show()


