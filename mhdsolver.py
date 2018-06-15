import numpy as np
import scipy
import matplotlib.pyplot as plt


class MHDSystem:
    def __init__(self, N_r=50, N_ghost=1, r_max=1):
        self.grid = Grid(N_r, N_ghost, r_max)
        self.fd = FDSystem(self.grid)

    def solve_eos(self, pressure):
        rho = pressure / 2.0
        return rho


class Grid:
    def __init__(self, N_r=50, N_ghost=1, r_max=1):
        # set up grid, with ghost cells
        N = 2 * N_ghost + N_r
        dr = r_max / N_r
        r = np.linspace(-dr / 2, r_max + dr / 2, N)
        self.r = r
        self.N = N
        self.dr = dr
        self.N_r = N_r
        self.N_ghost = N_ghost
        self.r_max = r_max


class MHDEquilibrium:
    def __init__(self, sys, p):
        # given a pressure, solve for magnetic field
        self.sys = sys
        self.p = p
        self.rho = self.compute_rho_from_p()
        self.B = self.compute_b_from_p()
        self.J = self.compute_j_from_b()

    def compute_rho_from_p(self):
        # use the EOS in the MHDSystem
        rho = self.sys.solve_eos(self.p)
        return rho

    def compute_b_from_p(self):
        # solve equilibrium ideal MHD equation
        #   (d/dr)(P + 0.5*B^2) + B^2/r = 0
        #  or
        #   (0.5 * (d/dr) + 1/r) B^2 = - (d/dr) P
        ddr = self.sys.fd.ddr()
        rhs = -(ddr @ self.p)
        r_inv = self.sys.fd.diag(1 / self.sys.grid.r)
        # set boundary conditions
        rhs[0] = 0
        rhs[-1] = 0
        r_inv[0] = 0
        r_inv[-1] = 0

        ddr[0,0] = 1
        ddr[0,1] = 1
        # no current at boundary... should be that current is continuous
        ddr[-1,-1] = self.sys.grid.r[-1]
        ddr[-1,-2] = -self.sys.grid.r[-2]

        b_squared = np.linalg.solve(0.5 * ddr + r_inv, rhs)
        return np.sign(b_squared)*np.sqrt(np.abs(b_squared))

    def compute_j_from_b(self):
        dv = self.sys.fd.ddr_product(self.sys.grid.r)
        # dv = self.sys.fd.ddr()
        # J = (1 / self.sys.grid.r) * (dv @ (self.sys.grid.r * self.B) )
        J = (1 / self.sys.grid.r) * (dv @ (self.B) )
        return J


class FDSystem:
    def __init__(self, grid):
        self.grid = grid
        self.bc_rows = {'value': [1, 1],
                        'derivative': [1, -1]}

    def ddr(self):
        # centered FD matrix for derivative
        one = np.ones(self.grid.N - 1)
        dv = (np.diag(one, 1) - np.diag(one, -1)) / (2 * self.grid.dr)
        dv = self.zero_bc(dv)
        return dv

    def ddr_product(self, vec):
        dv = (np.diag(vec[1:], 1) - np.diag(vec[:-1], -1))/(2 * self.grid.dr)
        dv = self.zero_bc(dv)
        return dv

    def diag(self,vec):
        M = np.diag(vec, 0)
        M = self.zero_bc(M)
        return M

    def zeros(self):
        return np.zeros((self.grid.N, self.grid.N))

    def zero_bc(self, M):
        # set rows corresponding to BC equations to zero
        M[0, :] = 0
        M[-1, :] = 0
        return M

    def lhs_bc(self, bc_type='value'):
        entries = self.bc_rows[bc_type]
        row = np.zeros(self.grid.N)
        row[0] = entries[0]
        row[1] = entries[1]
        M = self.zeros()
        M[0] = row
        return M

    def rhs_bc(self, bc_type='value'):
        entries = self.bc_rows[bc_type]
        row = np.zeros(self.grid.N)
        row[-2] = entries[0]
        row[-1] = entries[1]
        M = self.zeros()
        M[-1] = row
        return M


class LinearizedMHD:
    def __init__(self, equilibrium, k=1):
        self.equilibrium = equilibrium
        self.fd_operator = None
        self.fd_rhs = None
        self.evals = None
        self.evects = None
        self.set_z_mode(k)

    def set_z_mode(self, k):
        self.fd_operator = self.construct_operator(k)
        self.fd_rhs = self.construct_rhs()
        self.evals = None
        self.evects = None

    def construct_operator(self, k=1):
        fd = self.equilibrium.sys.fd
        r = self.equilibrium.sys.grid.r
        rho = self.equilibrium.rho
        B = self.equilibrium.B

        # Elements of the block matrix, of which 8 are zero.
        m3 = 1j * fd.ddr_product(r * rho) / r[:, np.newaxis]
        m4 = fd.diag(-k * rho)
        m7 = 1j * fd.ddr_product(B)
        m8 = fd.diag(-k * B)
        m9 = 2j * fd.ddr() / rho[:, np.newaxis]
        m10 = 1j * fd.ddr_product(r**2 * B) / (4 * np.pi * rho * r ** 2)[:, np.newaxis]
        m13 = fd.diag(-2.0 * k / rho)
        m14 = fd.diag(-k * B / (4.0 * np.pi * rho))

        m0 = fd.zeros()
        m1 = fd.zeros()
        m6 = fd.zeros()
        m11 = fd.zeros()
        m16 = fd.zeros()

        m1 = m1 + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m6 = m6 + fd.lhs_bc('value') + fd.rhs_bc('derivative')
        m11 = m11 + fd.lhs_bc('value') + fd.rhs_bc('derivative')
        m16 = m16 + fd.lhs_bc('derivative') + fd.rhs_bc('value')

        M = np.block([[m1, m0, m3, m4], [m0, m6, m7, m8], [m9, m10, m11, m0], [m13, m14, m0, m16]])
        return M

    def construct_rhs(self):
        # Generalized eigenvalue problem matrix
        nr = self.equilibrium.sys.grid.N
        G = -np.identity(4 * nr)
        G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2 * nr - 1, 2 * nr - 1] = 0
        G[2 * nr, 2 * nr] = G[3 * nr - 1, 3 * nr - 1] = G[3 * nr, 3 * nr] = G[-1, -1] = 0
        return G

    def solve(self):
        self.evals, self.evects = scipy.linalg.eig(self.fd_operator, self.fd_rhs)

    # ith mode by magnitude of imaginary part
    def plot_mode(self, i):
        if self.evects is None:
            return

        nr = self.equilibrium.sys.grid.N
        r = self.equilibrium.sys.grid.r

        index = np.argsort(self.evals.imag)

        omega = self.evals[index[i]]
        v_omega = self.evects[:, index[i]]

        f = plt.figure()
        f.suptitle(omega)

        # def f1(x): return np.abs(x)
        # def f2(x): return np.unwrap(np.angle(x)) / (2*3.14159)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)

        ax = plt.subplot(2, 2, 1)
        ax.set_title("rho")
        rho = v_omega[0: nr]
        t = np.exp(-1j * np.angle(rho[0]))
        ax.plot(r[1: -1], f1(t * rho[1: -1]),
                r[1: -1], f2(t * rho[1: -1]))

        ax = plt.subplot(2, 2, 2)
        ax.set_title("B_theta")
        B = v_omega[nr: 2 * nr]
        ax.plot(r[1: -1], f1(t * B[1: -1]),
                r[1: -1], f2(t * B[1: -1]))

        ax = plt.subplot(2, 2, 3)
        ax.set_title("V_r")
        V_r = v_omega[2 * nr: 3 * nr]
        ax.plot(r[1: -1], f1(t * V_r[1: -1]),
                r[1: -1], f2(t * V_r[1: -1]))

        ax = plt.subplot(2, 2, 4)
        ax.set_title("V_z")
        V_z = v_omega[3 * nr: 4 * nr]
        ax.plot(r[1: -1], f1(t * V_z[1: -1]),
                r[1: -1], f2(t * V_z[1: -1]))

        plt.show()

    def plot_eigenvalues(self):
        if self.evals is None:
            return

        plt.scatter(self.evals.real, self.evals.imag, s=1)
        plt.title('Omega')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()

    def fastest_mode(self):
        if self.evals is None:
            return None

        return max(self.evals.imag)
