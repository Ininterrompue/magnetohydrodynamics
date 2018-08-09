import numpy as np
from scipy.special import erf
from scipy.special import gamma
from scipy.special import gammainc
from scipy.linalg import eig
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

class Const:
    # c = cm/ns
#     c   = 30
# 
#     # m_i = MeV/c^2
#     m_i = 1.04
# 
#     # I = 1 MA in Gaussian units
#     I   = 1e6 * 2.37e3
# 
#     # T_0 = 100000 K
#     T_0 = 1e5 * 8.62e-11
# 
#     # r_0 = 1 cm
#     r_0 = 1
#     
#     # g = 9.81 cm/ns^2
#     g = 9.81e-16
    
    
    

    # Normalized constants
    c   = 1
    m_i = 1
    I   = np.sqrt(np.pi)
    T_0 = 1
    r_0 = 1
#     g   = 1e-3

    P_0 = I**2 / (np.pi * r_0**2 * c**2)

class MHDSystem:
    def __init__(self, N_r=50, N_ghost=1, r_max=2*np.pi, g=0, D_eta=0, D_H=0, D_P=0, B_Z0=0):
        self.grid = Grid(N_r, N_ghost, r_max)
        self.fd = FDSystem(self.grid)

        # set plasma related parameters
        self.g = g
        self.D_eta = D_eta
        self.D_H = D_H
        self.D_P = D_P
        self.B_Z0 = B_Z0


class Grid:
    def __init__(self, N_r=50, N_ghost=1, r_max=1):
        # set up grid, with ghost cells
        N = 2 * N_ghost + N_r
        dr = r_max / N_r
        r = np.linspace(-dr / 2 * (2 * N_ghost - 1), r_max + dr / 2 * (2 * N_ghost - 1), N)
        rr = np.reshape(r, (N, 1))
        self.r = r
        self.rr = rr
        self.N = N
        self.dr = dr
        self.N_r = N_r
        self.N_ghost = N_ghost
        self.r_max = r_max


class MHDEquilibrium0:
    def __init__(self, sys, p_exp):
        # given a pressure, solve for magnetic field
        self.sys = sys
        self.p_exp = p_exp
        self.p = self.compute_p()
        self.rho = self.compute_rho_from_p()
        self.B = self.compute_b_from_p()
        self.J = self.compute_j_from_b()

    def compute_p(self):
        return Const.P_0 * (0.05 + np.exp(-(self.sys.grid.rr)**self.p_exp))
        
    def compute_rho_from_p(self):
        # Equation of state. Initial conditions: T = 1 uniform.
        rho = Const.m_i * self.p / (2 * Const.T_0)
        return rho

    def compute_b_from_p(self):
        a = Const.m_i * self.sys.g / (2 * Const.T_0)
        b_pressure = Const.P_0 * (a / self.p_exp * gamma(1 / self.p_exp) * gammainc(1 / self.p_exp, self.sys.grid.rr**self.p_exp)
                                  - np.exp(-(self.sys.grid.rr)**self.p_exp) + 1 + 0.05 * a * self.sys.grid.rr)
        b = np.sqrt(8 * np.pi) * np.sign(b_pressure) * np.sqrt(np.abs(b_pressure))
        # boundary condition to ensure no NaNs
        b[0] = b[1]
        return b

    def compute_j_from_b(self):
        return (self.sys.fd.ddr(1) @ self.B) * Const.c / (4 * np.pi)


class FDSystem:
    def __init__(self, grid):
        self.grid = grid
        self.bc_rows = {'value': [1, 1],
                        'derivative': [1, -1]}

    def ddr(self, order):
        one = np.ones(self.grid.N - 1)
        if order == 1:
            dv = (np.diag(one, 1) - np.diag(one, -1)) / (2 * self.grid.dr)
        elif order == 2:
            dv = (dia_matrix((one, 1), shape=(self.grid.N, self.grid.N)).toarray() 
                  - 2 * np.identity(self.grid.N) + dia_matrix((one, -1), shape=(self.grid.N, self.grid.N)).toarray()) / self.grid.dr**2  
      
        dv = self.zero_bc(dv)
        return dv

    def ddr_product(self, vec):
        dv = (np.diagflat(vec[1: ], 1) - np.diagflat(vec[: -1], -1)) / (2 * self.grid.dr)
        dv = self.zero_bc(dv)
        return dv

    def diag(self,vec):
        M = np.diagflat(vec, 0)
        M = self.zero_bc(M)
        return M
    
    def diag_I(self):
        return np.identity(self.grid.N)
        
    def zeros(self):
        return np.zeros((self.grid.N, self.grid.N))

    def zero_bc(self, M_0):
        M = M_0.copy()
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


class MHDEvolution:
    def __init__(self, lin, equilibrium, t_max):
        self.lin = lin
        self.equilibrium = equilibrium
        self.t_max = t_max
        
    def evolve(self, k=1):
        nr = self.equilibrium.sys.grid.N
        r = self.equilibrium.sys.grid.r
        rr = self.equilibrium.sys.grid.rr * np.ones(nr).T
        z = self.equilibrium.sys.grid.r
        zz = self.equilibrium.sys.grid.rr
        dr = self.equilibrium.sys.grid.dr
        g = self.equilibrium.sys.g
        t_max = self.t_max

        pert = 1 - 0.00 * np.cos(k * zz)
        rho = self.equilibrium.rho * pert.T
        B = self.equilibrium.B * np.ones(nr).T
        p = self.equilibrium.p * pert.T
        Vr = np.zeros((nr, nr))
        Vz = np.zeros((nr, nr))
        T = np.ones((nr, nr))
    
        # ideal for now
        D_eta = 0
        ratio = 2
        
        t = 0
        iteration = 0
        counter = 0


#         B, Vr, Vz, rho, p, T = self.lin.plot_VB(-1, epsilon=0.05)
#         B = B.T
#         Vr = Vr.T
#         Vz = Vz.T
#         rho = rho.T
#         p = p.T
#         T = T.T

        while t < t_max:
            iteration += 1
            counter += 1
            rho_temp = rho.copy()
            B_temp = B.copy()
            Vr_temp = Vr.copy()  
            Vz_temp = Vz.copy()
            p_temp = p.copy()
    
            # Courant condition
            v_fluid = np.amax(np.abs(Vr))
            v_alfven2 = np.amax(np.abs(B)**2 / (4 * np.pi * np.abs(rho)))
            v_sound2 = np.amax(2 * np.abs(T))
            v_magnetosonic = v_alfven2 + v_sound2
            v_courant = v_fluid + v_magnetosonic
        
            dt = dr / v_courant * 1e-1 * 0.5
            if dt < 1e-7:
                print('Solution has not converged')
                break
                
            t += dt
            if counter == 100:
                print(iteration, v_courant, t)
                counter = 0
                
            # Finite difference procedure                      
            rho[1: -1, 1: -1] = (rho_temp[1: -1, 1: -1] 
                                 - dt / (2 * dr) * (rho_temp[2: , 1: -1] * Vr_temp[2: , 1: -1] - rho_temp[: -2, 1: -1] * Vr_temp[: -2, 1: -1])
                                                    - dt / (2 * dr) * (rho_temp[1: -1, 2: ] * Vz_temp[1: -1, 2: ]
                                                    - rho_temp[1: -1, : -2] * Vz_temp[1: -1, : -2]))
            B[1: -1, 1: -1] = (B_temp[1: -1, 1: -1] - dt / (2 * dr) * (Vr_temp[2: , 1: -1] * B_temp[2: , 1: -1] - Vr_temp[: -2, 1: -1] * B_temp[: -2, 1: -1])
                                                    - dt / (2 * dr) * (Vz_temp[1: -1, 2: ] * B_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2] * B_temp[1: -1, : -2]))
                                                    
            Vr[1: -1, 1: -1] = (Vr_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (Vr_temp[2: , 1: -1] - Vr_temp[: -2, 1: -1])
                                                      - dt / (2 * dr) * Vz_temp[1: -1, 1: -1] * (Vr_temp[1: -1, 2: ] - Vr_temp[1: -1, : -2]) 
                                                      - dt / (2 * dr * rho_temp[1: -1, 1: -1]) * (p_temp[2: , 1: -1] - p_temp[: -2, 1: -1]) 
                                                      - B_temp[1: -1, 1: -1] * dt / (4 * np.pi * rho_temp[1: -1, 1: -1] * 2 * dr) * (B_temp[2: , 1: -1] - B_temp[: -2, 1: -1]) 
                                                      + dt * g)
                                                      
            Vz[1: -1, 1: -1] = (Vz_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (Vz_temp[2: , 1: -1] - Vz_temp[: -2, 1: -1])
                                                      - dt / (2 * dr) * Vz_temp[1: -1, 1: -1] * (Vz_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2])
                                                      - dt / (2 * dr * rho_temp[1: -1, 1: -1]) * (p_temp[1: -1, 2: ] - p_temp[1: -1, : -2])
                                                      - B_temp[1: -1, 1: -1] * dt / (4 * np.pi * rho_temp[1: -1, 1: -1] * 2 * dr) * (B_temp[1: -1, 2: ] - B_temp[1: -1, : -2]))
                                         
            p[1: -1, 1: -1] = (p_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (p_temp[2: , 1: -1] - p_temp[: -2, 1: -1]) 
                                                    - dt / (2 * dr) * Vz_temp[1: -1, 1: -1] * (p_temp[1: -1, 2: ] - p_temp[1: -1, : -2])
                               - ratio * dt / (2 * dr) * p_temp[1: -1, 1: -1] * (Vr_temp[2: , 1: -1] - Vr_temp[: -2, 1: -1])
                               - ratio * dt / (2 * dr) * p_temp[1: -1, 1: -1] * (Vz_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2]))
            T[1: -1, 1: -1] = Const.m_i * p[1: -1, 1: -1] / (2 * rho[1: -1, 1: -1])
            
            # Boundary conditions
            rho[0, :] = rho[1, :]
            rho[-1, :] = rho[-2, :]
            rho[: , 0] = rho[: , -2] 
            rho[: , -1] = rho[: , 1]
            B[0, :] = -B[1, :]
            B[-1, :] = B[-2, :] + 0.05 * Const.m_i * g / (2 * Const.T_0) * Const.P_0 * dr
            B[: , 0] = B[: , -2]
            B[: , -1] = B[: , 1]
            Vr[0, :] = -Vr[1, :]
            Vr[-1, :] = -Vr[-2, :]
            Vr[:, 0] = Vr[:, -2]
            Vr[:, -1] = Vr[:, 1]
            Vz[0, :] = Vz[1, :]
            Vz[-1, :] = -Vz[-2, :]
            Vz[:, 0] = Vz[:, -2]
            Vz[:, -1] = Vz[:, 1]
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]
            T[0, :] = T[1, :]
            T[-1, :] = T[-2, :]
            T[:, 0] = T[:, -2]
            T[:, -1] = T[:, 1]
    
        plt.plot(r[1: -1], B[1: -1, 1], r[1: -1], rho[1: -1, 1], r[1: -1], Vr[1: -1, 1], r[1: -1], p[1: -1, 1], r[1: -1], T[1: -1, 1])
        plt.legend(['B', 'rho', 'V_x', 'p', 'T'])
        plt.xlabel('x')
        plt.title('Time evolution')
        plt.show()
#         
#         plt.plot(r[1: -1], B[1, 1: -1], r[1: -1], rho[1, 1: -1], r[1: -1], Vr[1, 1: -1], r[1: -1], Vz[1, 1: -1], r[1: -1], p[1, 1: -1], r[1: -1], T[1, 1: -1])
#         plt.legend(['B', 'rho', 'V_x', 'V_z', 'p', 'T'])
#         plt.xlabel('z')
#         plt.title('x = 1 lineout')
#         plt.show()
        
        R, Z = np.meshgrid(r[1: -1], z[1: -1]) 
        
        ax = plt.subplot(2,3,1)
        ax.set_title('rho')
        plot_1 = ax.contourf(R, Z, rho[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_1)
    
        ax = plt.subplot(2,3,2)
        ax.set_title('p')
        plot_2 = ax.contourf(R, Z, p[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_2)
    
        ax = plt.subplot(2,3,3)
        ax.set_title('T')
        plot_3 = ax.contourf(R, Z, T[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_3)
    
        ax = plt.subplot(2,3,4)
        ax.set_title('B_theta')
        plot_4 = ax.contourf(R, Z, B[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_4)
        
        ax = plt.subplot(2,3,5)
        ax.set_title('V_x')
        plot_5 = ax.contourf(R, Z, Vr[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_5)
    
        ax = plt.subplot(2,3,6)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, Vz[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_6)     
        
        plt.show()    
        
        d_vec=10
        plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Vr[::d_vec, ::d_vec].T, Vz[::d_vec, ::d_vec].T, 
                   pivot='mid', width=0.002, scale=10)
        plt.show()   

class LinearizedMHD:
    def __init__(self, equilibrium, k=1, m=0):
        self.equilibrium = equilibrium
        self.fd_operator = None
        self.fd_rhs = None
        self.evals = None
        self.evects = None
        self.set_z_mode(k, m)
        self.k = k
        self.m = m
        self._sigma = None

    def set_z_mode(self, k, m):
        self.fd_operator = self.construct_operator(k, m)
        self.fd_rhs = self.construct_rhs()
        self.evals = None
        self.evects = None

    def construct_operator(self, k=1, m=0):
        fd = self.equilibrium.sys.fd
        r = self.equilibrium.sys.grid.r
        rr = self.equilibrium.sys.grid.rr
        rho = self.equilibrium.rho
        B = self.equilibrium.B
        p = self.equilibrium.p

        # get plasma parameters from the system
        D_eta = self.equilibrium.sys.D_eta
        D_H = self.equilibrium.sys.D_H
        D_P = self.equilibrium.sys.D_P
        B_Z0 = self.equilibrium.sys.B_Z0
        
        # Ratio of specific heats 
        ratio = 2
        
        # Gravity
        g = 1

        # Ideal MHD
        m_rho_Vr = -1j * fd.ddr_product(rho)
        m_rho_Vz = fd.diag(k * rho)

        m_Btheta_Vr = -1j * fd.ddr_product(B)
        m_Btheta_Vz = fd.diag(k * B)
        
        m_Vr_rho = fd.diag(1j * g / rho)
        m_Vr_p = -1j * fd.ddr(1) / rho
        m_Vr_Btheta = -1j * fd.ddr_product(B) / (4 * np.pi * rho)
        
        m_Vz_p = fd.diag(k / rho)
        m_Vz_Btheta = fd.diag(k * B / (4 * np.pi * rho))
        
        m_p_Vr = fd.diag(-1j * (fd.ddr(1) @ p)) - ratio * 1j * p * fd.ddr(1)
        m_p_Vz = fd.diag(ratio * p * k)
       
        m0 = fd.zeros()
        m_rho_rho = fd.zeros()
#         m_Br_Br = fd.zeros()
#         m_Br_Btheta = fd.zeros()
#         m_Btheta_rho = fd.zeros()
#         m_Btheta_Br = fd.zeros()
        m_Btheta_Btheta = fd.zeros()
#         m_Btheta_Bz = fd.zeros()
#         m_Btheta_p = fd.zeros()
#         m_Bz_Br = fd.zeros()
#         m_Bz_Btheta = fd.zeros()
#         m_Bz_Bz = fd.zeros()
        m_Vr_Vr = fd.zeros()
#         m_Vtheta_Vtheta = fd.zeros()
        m_Vz_Vz = fd.zeros()
        m_p_p = fd.zeros()
        
        # Resistive term
#         m_Br_Br = m_Br_Br + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
#         m_Br_Btheta = m_Br_Btheta + D_eta * fd.diag(2 * m / rr**2)
#         m_Btheta_Btheta = m_Btheta_Btheta + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
#         m_Btheta_Br = m_Btheta_Br + D_eta * fd.diag(-2 * m / rr**2)
#         m_Bz_Bz = m_Bz_Bz + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2)) 
        
        # Hall term (m = 0 only)
#         m_Br_Br = m_Br_Br + D_H * fd.diag(-k / (rr * rho) * (fd.ddr(1) @ (rr * B)))
#         m_Br_Btheta = m_Br_Btheta + D_H * fd.diag(-1j * B_Z0 * k**2 / rho)    
#         m_Btheta_rho = m_Btheta_rho + D_H * fd.diag(-8 * np.pi * k / rho**2 * (fd.ddr(1) @ rho))
#         m_Btheta_Br = m_Btheta_Br + D_H * fd.diag(1j * B_Z0 * k**2 / rho)
#         m_Btheta_Btheta = m_Btheta_Btheta + D_H * fd.diag(-k * B / rho**2 * (fd.ddr(1) @ rho) - 2 * k * B / (rho * rr))
#         m_Btheta_Bz = m_Btheta_Bz + D_H * ((-B_Z0 * k / rho) * fd.ddr(1))
        
#         m_Bz_Br = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B)) - 1j / (rho * rr) * (fd.ddr(2) @ (rr * B)))
#                                    - 1j / (rho * rr) * (fd.ddr(1) @ (rr * B)) * fd.ddr(1))
# #         m_Bz_Br = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B))) - 1j / (rho * rr) * (fd.ddr_product(fd.ddr(1) @ (rr * B))))
#         m_Bz_Btheta = m_Bz_Btheta + D_H * (B_Z0 * k / (rr * rho) * fd.ddr_product(rr) - fd.diag(k * B_Z0 / rho**2 * (fd.ddr(1) @ rho)))
        
        # Electron pressure term is 0 for our current equation of state
#         m_Btheta_rho = m_Btheta_rho + D_P * fd.diag(-k / rho**2 * (fd.ddr(1) @ p))
#         m_Btheta_p = m_Btheta_p + D_P * fd.diag(k / rho**2 * (fd.ddr(1) @ rho))
        
        # Boundary conditions
        m_rho_rho       = m_rho_rho       + fd.lhs_bc('derivative') + fd.rhs_bc('value')
#         m_Br_Br         = m_Br_Br         + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Btheta_Btheta = m_Btheta_Btheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
#         m_Bz_Bz         = m_Bz_Bz         + fd.lhs_bc('derivative') + fd.rhs_bc('derivative')
        m_Vr_Vr         = m_Vr_Vr         + fd.lhs_bc('value')      + fd.rhs_bc('derivative')
#         m_Vtheta_Vtheta = m_Vtheta_Vtheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Vz_Vz         = m_Vz_Vz         + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m_p_p           = m_p_p           + fd.lhs_bc('derivative') + fd.rhs_bc('value')
                      
        M = np.block([[m_rho_rho,    m0,              m_rho_Vr,    m_rho_Vz,    m0        ],
                      [m0,           m_Btheta_Btheta, m_Btheta_Vr, m_Btheta_Vz, m0        ],
                      [m_Vr_rho,     m_Vr_Btheta,     m_Vr_Vr,     m0,          m_Vr_p    ],
                      [m0,           m_Vz_Btheta,     m0,          m_Vz_Vz,     m_Vz_p    ],
                      [m0,           m0,              m_p_Vr,      m_p_Vz,      m_p_p     ]])
        return M

    def construct_rhs(self):
        # Generalized eigenvalue problem matrix
        nr = self.equilibrium.sys.grid.N
        G = np.identity(5 * nr)
        G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
        G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = 0
        G[4*nr - 1, 4*nr - 1] = G[4*nr, 4*nr] = G[5*nr - 1, 5*nr - 1] = 0
#         G[5*nr, 5*nr] = G[6*nr - 1, 6*nr - 1] = G[6*nr - 6*nr] = 0
#         G[7*nr - 1, 7*nr - 1] = G[7*nr, 7*nr] = G[-1, -1] = 0
        return G

    def solve(self, num_modes=None):
        if num_modes:
            self.evals, self.evects = eigs(self.fd_operator, k=num_modes, M=self.fd_rhs,
                                           sigma=3j, which='LI', return_eigenvectors=True)
        else:
            self.evals, self.evects = eig(self.fd_operator, self.fd_rhs)

    def solve_for_gamma(self):
        return eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=1j, which='LI', return_eigenvectors=False).imag

    # ith mode by magnitude of imaginary part
    def plot_VB(self, i, epsilon=1):
        if self.evects is None:
            return
            
        fd = self.equilibrium.sys.fd
        nr = self.equilibrium.sys.grid.N
        r = self.equilibrium.sys.grid.r
        z = self.equilibrium.sys.grid.r
        zz = self.equilibrium.sys.grid.rr
        rho_0 = self.equilibrium.rho
        p_0 = self.equilibrium.p
        B_0 = self.equilibrium.B
        B_Z0 = self.equilibrium.sys.B_Z0

        index = np.argsort(self.evals.imag)
        omega = self.evals[index[i]]
        v_omega = self.evects[:, index[i]]
        
        print(omega)
        
        rho = v_omega[0: nr]
        phase = np.exp(-1j * np.angle(rho[0]))
        rho = epsilon * phase * rho
        B_theta = epsilon * phase * v_omega[nr: 2*nr]
        V_r = epsilon * phase * v_omega[2*nr: 3*nr]
        V_z = epsilon * phase * v_omega[3*nr: 4*nr]
        p = epsilon * phase * v_omega[4*nr: 5*nr]
        
        # print(np.amax(B_0) / np.amax(B_theta))
        
        p_0 = np.reshape(p_0, (nr, ))
        rho_0 = np.reshape(rho_0, (nr, ))
        temp = (p + p_0) / (2 * (rho + rho_0))
        temp_1 = (p - 2 * rho) / (2 * (rho + rho_0))
              
        # 1D eigenvectors
        f = plt.figure()
        f.suptitle(omega.imag)

#         def f1(x): return np.abs(x)
#         def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)
              
        ax = plt.subplot(2,3,1)
        ax.set_title('B_y')
        ax.plot(r[1: -1], f1(B_theta[1: -1]),
                r[1: -1], f2(B_theta[1: -1]))
               
        ax = plt.subplot(2,3,2)
        ax.set_title('V_x')
        ax.plot(r[1: -1], f1(V_r[1: -1]),
                r[1: -1], f2(V_r[1: -1]))
           
        ax = plt.subplot(2,3,3)
        ax.set_title('V_z')	
        ax.plot(r[1: -1], f1(V_z[1: -1]),
                r[1: -1], f2(V_z[1: -1]))
            
        ax = plt.subplot(2,3,4)
        ax.set_title('rho')
        ax.plot(r[1: -1], f1(rho[1: -1]),
                r[1: -1], f2(rho[1: -1]))
                
        ax = plt.subplot(2,3,5)
        ax.set_title('p')
        ax.plot(r[1: -1], f1(p[1: -1]),
                r[1: -1], f2(p[1: -1]))
                
        ax = plt.subplot(2,3,6)
        ax.set_title('T')
        ax.plot(r[1: -1], f1(temp[1: -1]) - 1)

        plt.show()
        
        # 2D contour plots
        z_osc = np.exp(1j * self.k * zz)
        
        rho_contour = rho_0.T + f1(z_osc * rho)
        B_theta_contour = B_0.T + f1(z_osc * B_theta)
        V_r_contour = f1(z_osc * V_r)
        V_z_contour = f1(z_osc * V_z)
        p_contour = p_0.T + f1(z_osc * p)
        temp_contour = Const.T_0 + f1(z_osc * temp_1)
        
        return B_theta_contour, V_r_contour, V_z_contour, rho_contour, p_contour, temp_contour
        
        f = plt.figure()
        f.suptitle(omega.imag)
        R, Z = np.meshgrid(r[1: -1], z[1: -1])
    
        ax = plt.subplot(2,3,1)
        ax.set_title('B_y')
        plot_2 = ax.contourf(R, Z, B_theta_contour, 20)
        plt.colorbar(plot_2)
    
        ax = plt.subplot(2,3,2)
        ax.set_title('V_x')
        plot_4 = ax.contourf(R, Z, V_r_contour, 20)
        plt.colorbar(plot_4)
    
        ax = plt.subplot(2,3,3)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, V_z_contour, 20)
        plt.colorbar(plot_6)
    
        ax = plt.subplot(2,3,4)
        ax.set_title('rho')
        plot_7 = ax.contourf(R, Z, rho_contour, 20)
        plt.colorbar(plot_7)
        
        ax = plt.subplot(2,3,5)
        ax.set_title('p')
        plot_8 = ax.contourf(R, Z, p_contour, 20)
        plt.colorbar(plot_8)
        
        ax = plt.subplot(2,3,6)
        ax.set_title('T')
        plot_9 = ax.contourf(R, Z, temp_contour, 20)
        plt.colorbar(plot_9)
    
        plt.show()
        
        # V and vorticity
        vort_theta = np.reshape(1j * self.k * V_r - (fd.ddr(1) @ V_z), (nr, ))
        vort_theta_contour = epsilon * f1(z_osc[1: -1] * vort_theta[1: -1])

        plot = plt.contourf(R, Z, vort_theta_contour, 200, cmap='plasma')
        plt.colorbar(plot)
        
        d_vec = 15
        plt.title('Flow velocity and vorticity')
        plt.xlabel('r')
        plt.ylabel('z')
        quiv = plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   V_r_contour[::d_vec, ::d_vec], V_z_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=0.2)
        
        plt.show()
        
    def plot_EJ(self, i, epsilon=1):
        if self.evects is None:
            return
            
        fd = self.equilibrium.sys.fd
        nr = self.equilibrium.sys.grid.N
        r = self.equilibrium.sys.grid.r
        rr = self.equilibrium.sys.grid.rr
        z = self.equilibrium.sys.grid.r
        zz = self.equilibrium.sys.grid.rr
        rho_0 = self.equilibrium.rho
        B_0 = self.equilibrium.B
        J_0 = self.equilibrium.J
        D_eta = self.equilibrium.sys.D_eta
        D_H = self.equilibrium.sys.D_H
        D_P = self.equilibrium.sys.D_P
        B_Z0 = self.equilibrium.sys.B_Z0
        p_0 = self.equilibrium.p

        index = np.argsort(self.evals.imag)
        omega = self.evals[index[i]]
        v_omega = self.evects[:, index[i]]
        
        rho = v_omega[0: nr]
        phase = np.exp(-1j * np.angle(rho[0]))
        rho = epsilon * phase * rho
        B_r = epsilon * phase * v_omega[nr: 2*nr]
        B_theta = epsilon * phase * v_omega[2*nr: 3*nr]
        B_z = epsilon * phase * v_omega[3*nr: 4*nr]
        V_r = epsilon * phase * v_omega[4*nr: 5*nr]
        V_theta = epsilon * phase * v_omega[5*nr: 6*nr]
        V_z = epsilon * phase * v_omega[6*nr: 7*nr]
        p = epsilon * phase * v_omega[7*nr: 8*nr]
        
#         def f1(x): return np.abs(x)
#         def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)
        
        # Post-processing
        rho_1 = np.reshape(rho, (nr, 1))
        B_r = np.reshape(B_r, (nr, 1))
        B_1 = np.reshape(B_theta, (nr, 1))
        B_z = np.reshape(B_z, (nr, 1))  
        V_r = np.reshape(V_r, (nr, 1))
        V_theta = np.reshape(V_theta, (nr, 1))
        V_z = np.reshape(V_z, (nr, 1))
        B_Z1 = np.reshape(B_z, (nr, 1))
        p_1 = np.reshape(p, (nr, 1))     
        
        rho = rho_0 + rho_1
        B_theta = B_0 + B_1
        B_z = B_Z0 * np.reshape(np.ones(nr), (nr, 1)) + B_Z1
        p = p_0 + p_1
        
        d_rB_dr = fd.ddr(1) @ (rr * B_theta)
        d_Bz_dr = fd.ddr(1) @ B_z    
        
        J_r = 1 / (4 * np.pi) * -1j * self.k * B_1
        J_theta = 1 / (4 * np.pi) * (1j * self.k * B_r - d_Bz_dr)
        J_z1 = 1 / (4 * np.pi * rr) * (fd.ddr(1) @ (rr * B_1))
        J_z = J_0 + J_z1
        
        E_r_ideal = V_z * B_theta - V_theta * B_z
        E_theta_ideal = V_r * B_z - V_z * B_r
        E_z_ideal = V_theta * B_r - V_r * B_theta
        
        E_r_resistive = 4 * np.pi * D_eta * J_r
        E_theta_resistive = 4 * np.pi * D_eta * J_theta
        E_z0_resistive = 4 * np.pi * D_eta * J_0
        E_z1_resistive = 4 * np.pi * D_eta * J_z1
        
        E_r0_hall = 4 * np.pi * D_H / rho_0 * (-J_0 * B_0)
        E_r1_hall = 4 * np.pi * D_H / rho * (J_theta * B_z - J_z * B_theta) - 4 * np.pi * D_H / rho_0 * (-J_0 * B_0)
        E_theta_hall = 4 * np.pi * D_H / rho * (J_z * B_r - J_r * B_z)
        E_z_hall = 4 * np.pi * D_H / rho * (J_r * B_theta - J_theta * B_r)
        
        E_r0_pressure = -D_P / rho_0 * (fd.ddr(1) @ p_0)
        E_r1_pressure = -D_P / rho * (fd.ddr(1) @ p) - E_r0_pressure
        E_theta_pressure = np.zeros(nr)
        E_z_pressure = -D_P / rho * 1j * self.k * p_1
        
        # 1D perturbations of J and E
        ax = plt.subplot(1,3,1)
        ax.set_title('J_r')
        ax.plot(r[1: -1], f1(J_r[1: -1]),
                r[1: -1], f2(J_r[1: -1]))  
              
        ax = plt.subplot(1,3,2)
        ax.set_title('J_theta')
        ax.plot(r[1: -1], f1(J_theta[1: -1]),
                r[1: -1], f2(J_theta[1: -1]))
            
        ax = plt.subplot(1,3,3)
        ax.set_title('J_z')
        ax.plot(r[1: -1], f1(J_z1[1: -1]),
                r[1: -1], f2(J_z1[1: -1]))
                
        plt.show()    
                    
        ax = plt.subplot(4,3,1)
        ax.set_title('E_r_ideal')
        ax.plot(r[1: -1], f1(E_r_ideal[1: -1]),
                r[1: -1], epsilon * f2(E_r_ideal[1: -1]))
            
        ax = plt.subplot(4,3,2)
        ax.set_title('E_theta_ideal')
        ax.plot(r[1: -1], f1(E_theta_ideal[1: -1]),
                r[1: -1], f2(E_theta_ideal[1: -1]))
           
        ax = plt.subplot(4,3,3)
        ax.set_title('E_z_ideal')	
        ax.plot(r[1: -1], f1(E_z_ideal[1: -1]),
                r[1: -1], f2(E_z_ideal[1: -1]))
            
        ax = plt.subplot(4,3,4)
        ax.set_title('E_r_resistive')
        ax.plot(r[1: -1], f1(E_r_resistive[1: -1]),
                r[1: -1], f2(E_r_resistive[1: -1]))
            
        ax = plt.subplot(4,3,5)
        ax.set_title('E_theta_resistive')
        ax.plot(r[1: -1], f1(E_theta_resistive[1: -1]),
                r[1: -1], f2(E_theta_resistive[1: -1]))
           
        ax = plt.subplot(4,3,6)
        ax.set_title('E_z_resistive')	
        ax.plot(r[1: -1], f1(E_z1_resistive[1: -1]),
                r[1: -1], f2(E_z1_resistive[1: -1]))
                
        ax = plt.subplot(4,3,7)
        ax.set_title('E_r_hall')
        ax.plot(r[1: -1], f1(E_r1_hall[1: -1]),
                r[1: -1], f2(E_r1_hall[1: -1]))
            
        ax = plt.subplot(4,3,8)
        ax.set_title('E_theta_hall')
        ax.plot(r[1: -1], f1(E_theta_hall[1: -1]),
                r[1: -1], f2(E_theta_hall[1: -1]))
           
        ax = plt.subplot(4,3,9)
        ax.set_title('E_z_hall')	
        ax.plot(r[1: -1], f1(E_z_hall[1: -1]),
                r[1: -1], f2(E_z_hall[1: -1]))
                
        ax = plt.subplot(4,3,10)
        ax.set_title('E_r_pressure')
        ax.plot(r[1: -1], f1(E_r1_pressure[1: -1]),
                r[1: -1], f2(E_r1_pressure[1: -1]))
            
        ax = plt.subplot(4,3,11)
        ax.set_title('E_theta_pressure')
        ax.plot(r[1: -1], f1(E_theta_pressure[1: -1]),
                r[1: -1], f2(E_theta_pressure[1: -1]))
           
        ax = plt.subplot(4,3,12)
        ax.set_title('E_z_pressure')	
        ax.plot(r[1: -1], f1(E_z_pressure[1: -1]),
                r[1: -1], f2(E_z_pressure[1: -1]))

        plt.show()
        
        # 2D contour plots of J and E
        z_osc = np.exp(1j * self.k * zz)
        R, Z = np.meshgrid(r[1: -1], z[1: -1])
        
        J_r = np.reshape(J_r, (nr, ))
        J_theta = np.reshape(J_theta, (nr, ))
        J_z = np.reshape(J_z, (nr, ))
        E_r_ideal = np.reshape(E_r_ideal, (nr, ))
        E_theta_ideal = np.reshape(E_theta_ideal, (nr, ))
        E_z_ideal = np.reshape(E_z_ideal, (nr, ))
        E_r_resistive = np.reshape(E_r_resistive, (nr, ))
        E_theta_resistive = np.reshape(E_theta_resistive, (nr, ))
        E_z0_resistive = np.reshape(E_z0_resistive, (nr, ))
        E_z1_resistive = np.reshape(E_z1_resistive, (nr, ))
        E_r0_hall = np.reshape(E_r0_hall, (nr, ))
        E_r1_hall = np.reshape(E_r1_hall, (nr, ))
        E_theta_hall = np.reshape(E_theta_hall, (nr, ))
        E_z_hall = np.reshape(E_z_hall, (nr, ))
        E_r0_pressure = np.reshape(E_r0_pressure, (nr, ))
        E_r1_pressure = np.reshape(E_r1_pressure, (nr, ))
        E_z_pressure = np.reshape(E_z_pressure, (nr, ))
        
        J_r_contour = f1(z_osc[1: -1] * J_r[1: -1])
        J_theta_contour = f1(z_osc[1: -1] * J_theta[1: -1])
        J_z_contour = J_0[1: -1].T + f1(z_osc[1: -1] * J_z1[1: -1])
        E_r_ideal_contour = f1(z_osc[1: -1] * E_r_ideal[1: -1])
        E_theta_ideal_contour = f1(z_osc[1: -1] * E_theta_ideal[1: -1])
        E_z_ideal_contour = f1(z_osc[1: -1] * E_z_ideal[1: -1])
        E_r_resistive_contour = f1(z_osc[1: -1] * E_r_resistive[1: -1])
        E_theta_resistive_contour = f1(z_osc[1: -1] * E_theta_resistive[1: -1])
        E_z_resistive_contour = E_z0_resistive[1: -1].T + f1(z_osc[1: -1] * E_z1_resistive[1: -1])
        E_r_hall_contour = E_r0_hall[1: -1].T + f1(z_osc[1: -1] * E_r1_hall[1: -1])
        E_theta_hall_contour = f1(z_osc[1: -1] * E_theta_hall[1: -1])
        E_z_hall_contour = f1(z_osc[1: -1] * E_z_hall[1: -1])
        E_r_pressure_contour = E_r0_pressure[1: -1].T + f1(z_osc[1: -1] * E_r1_pressure[1: -1])
        E_theta_pressure_contour = f1(z_osc[1: -1] * E_theta_pressure[1: -1])
        E_z_pressure_contour = f1(z_osc[1: -1] * E_z_pressure[1: -1])
        
        ax = plt.subplot(1,3,1)
        ax.set_title('J_r')
        plot_1 = ax.contourf(R, Z, J_r_contour, 20)
        plt.colorbar(plot_1)
    
        ax = plt.subplot(1,3,2)
        ax.set_title('J_theta')
        plot_2 = ax.contourf(R, Z, J_theta_contour, 20)
        plt.colorbar(plot_2)
    
        ax = plt.subplot(1,3,3)
        ax.set_title('J_z')
        plot_3 = ax.contourf(R, Z, J_z_contour, 20)
        plt.colorbar(plot_3)
    
        ax = plt.subplot(4,3,1)
        ax.set_title('E_r_ideal')
        plot_4 = ax.contourf(R, Z, E_r_ideal_contour, 20)
        plt.colorbar(plot_4)
        
        ax = plt.subplot(4,3,2)
        ax.set_title('E_theta_ideal')
        plot_5 = ax.contourf(R, Z, E_theta_ideal_contour, 20)
        plt.colorbar(plot_5)
    
        ax = plt.subplot(4,3,3)
        ax.set_title('E_z_ideal')
        plot_6 = ax.contourf(R, Z, E_z_ideal_contour, 20)
        plt.colorbar(plot_6)
        
        ax = plt.subplot(4,3,4)
        ax.set_title('E_r_resistive')
        plot_7 = ax.contourf(R, Z, E_r_resistive_contour, 20)
        plt.colorbar(plot_7)
        
        ax = plt.subplot(4,3,5)
        ax.set_title('E_theta_resistive')
        plot_8 = ax.contourf(R, Z, E_theta_resistive_contour, 20)
        plt.colorbar(plot_8)
    
        ax = plt.subplot(4,3,6)
        ax.set_title('E_z_resistive')
        plot_9 = ax.contourf(R, Z, E_z_resistive_contour, 20)
        plt.colorbar(plot_9)
        
        ax = plt.subplot(4,3,7)
        ax.set_title('E_r_hall')
        plot_10 = ax.contourf(R, Z, E_r_hall_contour, 20)
        plt.colorbar(plot_10)
        
        ax = plt.subplot(4,3,8)
        ax.set_title('E_theta_hall')
        plot_11 = ax.contourf(R, Z, E_theta_hall_contour, 20)
        plt.colorbar(plot_11)
    
        ax = plt.subplot(4,3,9)
        ax.set_title('E_z_hall')
        plot_12 = ax.contourf(R, Z, E_z_hall_contour, 20)
        plt.colorbar(plot_12)
        
        ax = plt.subplot(4,3,10)
        ax.set_title('E_r_pressure')
        plot_13 = ax.contourf(R, Z, E_r_pressure_contour, 20)
        plt.colorbar(plot_13)
        
        ax = plt.subplot(4,3,11)
        ax.set_title('E_theta_pressure')
        plot_14 = ax.contourf(R, Z, E_theta_pressure_contour, 20)
        plt.colorbar(plot_14)
    
        ax = plt.subplot(4,3,12)
        ax.set_title('E_z_pressure')
        plot_15 = ax.contourf(R, Z, E_z_pressure_contour, 20)
        plt.colorbar(plot_15)

        plt.show()
        
        # 2D quiver plots of J and E
        d_vec = 10
        
        ax = plt.subplot(3,2,1)
        ax.set_title('J')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   J_r_contour[::d_vec, ::d_vec], J_z_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=20)

        ax = plt.subplot(3,2,3)
        ax.set_title('E_ideal')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_ideal_contour[::d_vec, ::d_vec], E_z_ideal_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=0.5)
    
        ax = plt.subplot(3,2,4)
        ax.set_title('E_resistive')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_resistive_contour[::d_vec, ::d_vec], E_z_resistive_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=2)
    
        ax = plt.subplot(3,2,5)
        ax.set_title('E_hall')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_hall_contour[::d_vec, ::d_vec], E_z_hall_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=2)
                   
        ax = plt.subplot(3,2,6)
        ax.set_title('E_pressure')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_pressure_contour[::d_vec, ::d_vec], E_z_pressure_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=0.5)
        
        plt.show()
        
        # Total electric field
        ax = plt.subplot(2, 2, 1)
        ax.set_title('E_r_total')
        plot_1 = ax.contourf(R, Z, E_r_ideal_contour + E_r_resistive_contour + E_r_hall_contour + E_r_pressure_contour, 20, cmap='plasma')
        plt.colorbar(plot_1)
        
        ax = plt.subplot(2, 2, 2)
        ax.set_title('E_theta_total')
        plot_2 = ax.contourf(R, Z, E_theta_ideal_contour + E_theta_resistive_contour + E_theta_hall_contour + E_theta_pressure_contour, 20, cmap='plasma')
        plt.colorbar(plot_2)
        
        ax = plt.subplot(2, 2, 3)
        ax.set_title('E_z_total')
        plot_3 = ax.contourf(R, Z, E_z_ideal_contour + E_z_resistive_contour + E_z_hall_contour + E_z_pressure_contour, 20, cmap='plasma')
        plt.colorbar(plot_3)
        
        ax = plt.subplot(2, 2, 4)
        ax.set_title('E_total')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_ideal_contour[::d_vec, ::d_vec] + E_r_resistive_contour[::d_vec, ::d_vec] + E_r_hall_contour[::d_vec, ::d_vec] + E_r_pressure_contour[::d_vec, ::d_vec], 
                   E_z_ideal_contour[::d_vec, ::d_vec] + E_z_resistive_contour[::d_vec, ::d_vec] + E_z_hall_contour[::d_vec, ::d_vec] + E_z_pressure_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=2)
        
        plt.show()
        
    def plot_eigenvalues(self):
        if self.evals is None:
            return

        plt.scatter(self.evals.real, self.evals.imag, s=1)
        plt.title('Omega')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()
