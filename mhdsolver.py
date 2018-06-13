import numpy as np


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
        r_inv = np.diag(1 / self.sys.grid.r)
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

    def ddr(self):
        # centered FD matrix for derivative
        one = np.ones(self.grid.N - 1)
        dv = (np.diag(one, 1) - np.diag(one, -1)) / (2 * self.grid.dr)
        # set rows corresponding to BC equations to zero
        dv[0, :] = 0
        dv[-1, :] = 0
        return dv

    def ddr_product(self, vec):
        dv = (np.diag(vec[1:], 1) - np.diag(vec[:-1], -1))/(2 * self.grid.dr)
        # set rows corresponding to BC equations to zero
        dv[0, :] = 0
        dv[-1, :] = 0
        return dv

