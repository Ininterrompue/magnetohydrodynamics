import numpy as np
from scipy.special import erf
from scipy.special import gamma
from scipy.special import gammainc
from scipy.linalg import eig
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


class MHDSystem:
    def __init__(self, N_r=50, N_ghost=1, r_max=2*np.pi, m=0, D_eta=0, D_H=0, D_P=0, B_Z0=0):
        self.grid = Grid(N_r, N_ghost, r_max)
        self.fd = FDSystem(self.grid)

        # set plasma related parameters
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


class MHDEquilibrium:
    def __init__(self, sys, p_exp):
        # given a pressure, solve for magnetic field
        self.sys = sys
        self.p_exp = p_exp
        self.p = self.compute_p()
        self.rho = self.compute_rho_from_p()
        self.B = self.compute_b_from_p()
        self.J = self.compute_j_from_b()

    def compute_p(self):
        return 0.05 + np.exp(-self.sys.grid.rr**self.p_exp)
        
    def compute_rho_from_p(self):
        # Equation of state
        rho = self.p / 2
        return rho

    def compute_b_from_p(self):
        # solve equilibrium ideal MHD equation
        #   (d/dr)(P + 0.5*B^2) + B^2/r = 0
        lhs = 1/2 * self.sys.fd.ddr(1) + self.sys.fd.diag(1 / self.sys.grid.r)
        rhs = -(self.sys.fd.ddr(1) @ self.p)
        
        # set boundary conditions
        if self.sys.grid.N_ghost == 1:
            rhs[0] = 0
            rhs[-1] = 0
            lhs[0, 0] = 1
            lhs[0, 1] = 1
            lhs[-1, -1] = -self.sys.grid.r[-1] / self.sys.grid.r[-2]
            lhs[-1, -2] = 1
        elif self.sys.grid.N_ghost == 2:
            rhs[0] = 0
            rhs[1] = 0
            rhs[-2] = 0
            rhs[-1] = 0
            lhs[1, 1] = 1
            lhs[1, 2] = 1
            lhs[0, 0] = 1
            lhs[0, 2] = 3
            lhs[-2, -1] = -self.sys.grid.r[-1] / self.sys.grid.r[-2]
            lhs[-2, -2] = 1
            lhs[-1, -2] = -self.sys.grid.r[-2] / self.sys.grid.r[-3]
            lhs[-1, -3] = 1

        # b_squared = np.linalg.solve(lhs, rhs)
        # b_squared = np.sqrt(np.pi) / self.sys.grid.rr**2 * erf(self.sys.grid.rr**2) - 2 * np.exp(-self.sys.grid.rr**4)
        b_squared = (4 / (self.p_exp * self.sys.grid.rr**2) * gamma(2 / self.p_exp) * gammainc(2 / self.p_exp, self.sys.grid.rr**self.p_exp) 
                    - 2 * np.exp(-self.sys.grid.rr**self.p_exp))
        b = np.sqrt(4 * np.pi) * np.sign(b_squared) * np.sqrt(np.abs(b_squared))
        # boundary condition to ensure no NaNs
        b[0] = b[1]
        return b 

    def compute_j_from_b(self):
        dv = self.sys.fd.ddr_product(self.sys.grid.rr)
        J = 1 / self.sys.grid.rr * (dv @ self.B)
        return J


class FDSystem:
    def __init__(self, grid):
        self.grid = grid
        self.bc_rows = {'value': [1, 1],
                        'derivative': [1, -1]}

    def ddr(self, order):
        one = np.ones(self.grid.N - 1)
        if self.grid.N_ghost == 1:
            if order == 1:
                dv = (np.diag(one, 1) - np.diag(one, -1)) / (2 * self.grid.dr)
            elif order == 2:
                dv = (dia_matrix((one, 1), shape=(self.grid.N, self.grid.N)).toarray() 
                      - 2 * np.identity(self.grid.N) + dia_matrix((one, -1), shape=(self.grid.N, self.grid.N)).toarray()) / self.grid.dr**2  
                        
        elif self.grid.N_ghost == 2:
            if order == 1:
                dv = (- 1/12 * dia_matrix((one, +2), shape=(self.grid.N, self.grid.N)) 
                      + 2/3  * dia_matrix((one, +1), shape=(self.grid.N, self.grid.N)) 
                      - 2/3  * dia_matrix((one, -1), shape=(self.grid.N, self.grid.N)) 
                      + 1/12 * dia_matrix((one, -2), shape=(self.grid.N, self.grid.N))).toarray() / self.grid.dr
            elif order == 2:
                dv = (- 1/12 * dia_matrix((one, +2), shape=(self.grid.N, self.grid.N))
                      + 4/3  * dia_matrix((one, +1), shape=(self.grid.N, self.grid.N))
                      - 5/2  * dia_matrix((one, +0), shape=(self.grid.N, self.grid.N))
                      + 4/3  * dia_matrix((one, -1), shape=(self.grid.N, self.grid.N))
                      - 1/12 * dia_matrix((one, -2), shape=(self.grid.N, self.grid.N))).toarray() / self.grid.dr**2
                      
        dv = self.zero_bc(dv)
        return dv

    def ddr_product(self, vec):
        if self.grid.N_ghost == 1:
            dv = (np.diagflat(vec[1: ], 1) - np.diagflat(vec[: -1], -1)) / (2 * self.grid.dr)
        elif self.grid.N_ghost == 2:
            dv = (-1/12 * np.diagflat(vec[2: ], 2) + 2/3 * np.diagflat(vec[1: ], 1) 
                  - 2/3 * np.diagflat(vec[: -1], -1) + 1/12 * np.diagflat(vec[: -2], -2)) / self.grid.dr
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
        
        if self.grid.N_ghost == 2:
            M[1, :] = 0
            M[-2, :] = 0
            
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

        # get plasma parameters from the system
        D_eta = self.equilibrium.sys.D_eta
        D_H = self.equilibrium.sys.D_H
        D_P = self.equilibrium.sys.D_P
        B_Z0 = self.equilibrium.sys.B_Z0

        # Ideal MHD
        m_rho_Vr = -1j * fd.ddr_product(rr * rho) / rr
        m_rho_Vtheta = fd.diag(m * rho / rr)
        m_rho_Vz = fd.diag(k * rho)
        
        m_Br_Vr = -B_Z0 * k * fd.diag_I() - fd.diag(m * B / rr)

        m_Btheta_Vr = -1j * fd.ddr_product(B)
        m_Btheta_Vtheta = -B_Z0 * k * fd.diag_I()
        m_Btheta_Vz = fd.diag(k * B)
        
        m_Bz_Vr = -1j * B_Z0 / rr * fd.ddr_product(rr)
        m_Bz_Vtheta = fd.diag(m * B_Z0 / rr)
        m_Bz_Vz = fd.diag(-m * B / rr)
        
        m_Vr_rho = -2j * fd.ddr(1) / rho
        m_Vr_Br = fd.diag(-B_Z0 * k / (4 * np.pi * rho) - m * B / (4 * np.pi * rho * rr))
        m_Vr_Btheta = -1j * fd.ddr_product(rr**2 * B) / (4 * np.pi * rho * rr**2)
        m_Vr_Bz = -1j * B_Z0 / (4 * np.pi * rho) * fd.ddr(1)
        
        m_Vtheta_rho = fd.diag(2 * m / (rho * rr))
        m_Vtheta_Br = fd.diag(1j * (fd.ddr(1) @ (rr * B)) / (4 * np.pi * rho * rr))
        m_Vtheta_Btheta = fd.diag(-B_Z0 * k / (4 * np.pi * rho))
        m_Vtheta_Bz = fd.diag(m * B_Z0 / (4 * np.pi * rho * rr))
        
        m_Vz_rho = fd.diag(2.0 * k / rho)
        m_Vz_Btheta = fd.diag(k * B / (4 * np.pi * rho))
        m_Vz_Bz = fd.diag(-m * B / (4 * np.pi * rho * rr))
       
        m0 = fd.zeros()
        m_rho_rho = fd.zeros()
        m_Br_Br = fd.zeros()
        m_Br_Btheta = fd.zeros()
        m_Btheta_rho = fd.zeros()
        m_Btheta_Br = fd.zeros()
        m_Btheta_Btheta = fd.zeros()
        m_Btheta_Bz = fd.zeros()
        m_Bz_Br = fd.zeros()
        m_Bz_Btheta = fd.zeros()
        m_Bz_Bz = fd.zeros()
        m_Vr_Vr = fd.zeros()
        m_Vtheta_Vtheta = fd.zeros()
        m_Vz_Vz = fd.zeros()
        
        # Resistive term
        m_Br_Br = m_Br_Br + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
        m_Br_Btheta = m_Br_Btheta + D_eta * fd.diag(2 * m / rr**2)
        m_Btheta_Btheta = m_Btheta_Btheta + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
        m_Btheta_Br = m_Btheta_Br + D_eta * fd.diag(-2 * m / rr**2)
        m_Bz_Bz = m_Bz_Bz + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2)) 
        
        # Hall term (m = 0 only)
        m_Br_Br = m_Br_Br + D_H * fd.diag(-k / (rr * rho) * (fd.ddr(1) @ (rr * B)))
        m_Br_Btheta = m_Br_Btheta + D_H * fd.diag(-1j * B_Z0 * k**2 / rho)    
        m_Btheta_rho = m_Btheta_rho + D_H * fd.diag(-8 * np.pi * k / rho**2 * (fd.ddr(1) @ rho))
        m_Btheta_Br = m_Btheta_Br + D_H * fd.diag(1j * B_Z0 * k**2 / rho)
        m_Btheta_Btheta = m_Btheta_Btheta + D_H * fd.diag(-k * B / rho**2 * (fd.ddr(1) @ rho) - 2 * k * B / (rho * rr))
        m_Btheta_Bz = m_Btheta_Bz + D_H * ((-B_Z0 * k / rho) * fd.ddr(1))
        
        m_Bz_Br = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B)) - 1j / (rho * rr) * (fd.ddr(2) @ (rr * B)))
                                   - 1j / (rho * rr) * (fd.ddr(1) @ (rr * B)) * fd.ddr(1))
#         m_Bz_Br = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B))) - 1j / (rho * rr) * (fd.ddr_product(fd.ddr(1) @ (rr * B))))
        m_Bz_Btheta = m_Bz_Btheta + D_H * (B_Z0 * k / (rr * rho) * fd.ddr_product(rr) - fd.diag(k * B_Z0 / rho**2 * (fd.ddr(1) @ rho)))
        
        # Electron pressure term is 0 for our current equation of state
        # m_Btheta_rho = m_Btheta_rho + D_P * fd.diag(2 * k / rho**2 * (fd.ddr(1) @ rho) + 2 * k * (fd.ddr(1) @ (1 / rho)))
        
        # Boundary conditions
        m_rho_rho       = m_rho_rho       + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m_Br_Br         = m_Br_Br         + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Btheta_Btheta = m_Btheta_Btheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Bz_Bz         = m_Bz_Bz         + fd.lhs_bc('derivative') + fd.rhs_bc('derivative')
        m_Vr_Vr         = m_Vr_Vr         + fd.lhs_bc('value')      + fd.rhs_bc('derivative')
        m_Vtheta_Vtheta = m_Vtheta_Vtheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Vz_Vz         = m_Vz_Vz         + fd.lhs_bc('derivative') + fd.rhs_bc('value')
                      
        M = np.block([[m_rho_rho,    m0,          m0,              m0,          m_rho_Vr,    m_rho_Vtheta,    m_rho_Vz   ], 
                      [m0,           m_Br_Br,     m_Br_Btheta,     m0,          m_Br_Vr,     m0,              m0         ],
                      [m_Btheta_rho, m_Btheta_Br, m_Btheta_Btheta, m_Btheta_Bz, m_Btheta_Vr, m_Btheta_Vtheta, m_Btheta_Vz],
                      [m0,           m_Bz_Br,     m_Bz_Btheta,     m_Bz_Bz,     m_Bz_Vr,     m_Bz_Vtheta,     m_Bz_Vz    ],
                      [m_Vr_rho,     m_Vr_Br,     m_Vr_Btheta,     m_Vr_Bz,     m_Vr_Vr,     m0,              m0         ],
                      [m_Vtheta_rho, m_Vtheta_Br, m_Vtheta_Btheta, m_Vtheta_Bz, m0,          m_Vtheta_Vtheta, m0         ],
                      [m_Vz_rho,     m0,          m_Vz_Btheta,     m_Vz_Bz,     m0,          m0,              m_Vz_Vz    ]])
        return M

    def construct_rhs(self):
        # Generalized eigenvalue problem matrix
        nr = self.equilibrium.sys.grid.N
        G = np.identity(7 * nr)
        G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
        G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = 0
        G[4*nr - 1, 4*nr - 1] = G[4*nr, 4*nr] = G[5*nr - 1, 5*nr - 1] = 0
        G[5*nr, 5*nr] = G[6*nr - 1, 6*nr - 1] = G[6*nr - 6*nr] = 0
        G[-1, -1] = 0
        return G

    def solve(self, num_modes=None):
        if num_modes:
            self.evals, self.evects = eigs(self.fd_operator, k=num_modes, M=self.fd_rhs,
                                           sigma=2j, which='LI', return_eigenvectors=True)
        else:
            self.evals, self.evects = eig(self.fd_operator, self.fd_rhs)
        
#     def solve_for_gamma(self, use_cached_sigma=False):
#         if use_cached_sigma or self._sigma:
#             e = eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=self._sigma*1j,
#                         which='LI', return_eigenvectors=False).imag
#             self._sigma = e
#             print(e)
#             return e
#         else:
#             self._sigma = eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=1j,
#                         which='LI', return_eigenvectors=False).imag
#             print(self._sigma)
#             return self._sigma

    def solve_for_gamma(self):
        return eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=0.5j, which='LI', return_eigenvectors=False).imag

    # ith mode by magnitude of imaginary part
    def plot_mode(self, i):
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
        B_Z0 = self.equilibrium.sys.B_Z0
        J_0 = self.equilibrium.J
        D_eta = self.equilibrium.sys.D_eta
        D_H = self.equilibrium.sys.D_H

        index = np.argsort(self.evals.imag)

        omega = self.evals[index[i]]
        v_omega = self.evects[:, index[i]]
        
        rho = v_omega[0: nr]
        B_r = v_omega[nr: 2*nr]
        B_theta = v_omega[2*nr: 3*nr]
        B_z = v_omega[3*nr: 4*nr]
        V_r = v_omega[4*nr: 5*nr]
        V_theta = v_omega[5*nr: 6*nr]
        V_z = v_omega[6*nr: 7*nr]
        phase = np.exp(-1j * np.angle(rho[0]))
        
        # 1D eigenvectors
        f = plt.figure()
        f.suptitle(omega.imag)

#         def f1(x): return np.abs(x)
#         def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)

        ax = plt.subplot(3,3,1)
        ax.set_title('B_r')
        ax.plot(r[1: -1], f1(phase * B_r[1: -1]),
                r[1: -1], f2(phase * B_r[1: -1]))  
              
        ax = plt.subplot(3,3,2)
        ax.set_title('B_theta')
        ax.plot(r[1: -1], f1(phase * B_theta[1: -1]),
                r[1: -1], f2(phase * B_theta[1: -1]) )
            
        ax = plt.subplot(3,3,3)
        ax.set_title('B_z')
        ax.plot(r[1: -1], f1(phase * B_z[1: -1]),
                r[1: -1], f2(phase * B_z[1: -1]))    
                    
        ax = plt.subplot(3,3,4)
        ax.set_title('V_r')
        ax.plot(r[1: -1], f1(phase * V_r[1: -1]),
                r[1: -1], f2(phase * V_r[1: -1]) )
            
        ax = plt.subplot(3,3,5)
        ax.set_title('V_theta')
        ax.plot(r[1: -1], f1(phase * V_theta[1: -1]),
                r[1: -1], f2(phase * V_theta[1: -1]) )
           
        ax = plt.subplot(3,3,6)
        ax.set_title('V_z')	
        ax.plot(r[1: -1], f1(phase * V_z[1: -1]),
                r[1: -1], f2(phase * V_z[1: -1]) )
            
        ax = plt.subplot(3,3,7)
        ax.set_title('rho')
        ax.plot(r[1: -1], f1(phase * rho[1: -1]),
                r[1: -1], f2(phase * rho[1: -1]))

        plt.show()
        
        # 2D contour plots
        z_osc = np.exp(1j * self.k * zz)
        rho_contour = rho_0[1: -1].T + f1(z_osc[1: -1] * phase * rho[1: -1])
        B_r_contour = f1(z_osc[1: -1] * phase * B_r[1: -1])
        B_theta_contour = B_0[1: -1].T + f1(z_osc[1: -1] * phase * B_theta[1: -1])
        B_z_contour = f1(z_osc[1: -1] * phase * B_z[1: -1])
        V_r_contour = f1(z_osc[1: -1] * phase * V_r[1: -1])
        V_theta_contour = f1(z_osc[1: -1] * phase * V_theta[1: -1])
        V_z_contour = f1(z_osc[1: -1] * phase * V_z[1: -1])
        
        f = plt.figure()
        f.suptitle(omega.imag)
        R, Z = np.meshgrid(r[1: -1], z[1: -1])
    
        ax = plt.subplot(3,3,1)
        ax.set_title('B_r')
        plot_1 = ax.contourf(R, Z, B_r_contour, 20)
        plt.colorbar(plot_1)
    
        ax = plt.subplot(3,3,2)
        ax.set_title('B_theta')
        plot_2 = ax.contourf(R, Z, B_theta_contour, 20)
        plt.colorbar(plot_2)
    
        ax = plt.subplot(3,3,3)
        ax.set_title('B_z')
        plot_3 = ax.contourf(R, Z, B_z_contour, 20)
        plt.colorbar(plot_3)
    
        ax = plt.subplot(3,3,4)
        ax.set_title('V_r')
        plot_4 = ax.contourf(R, Z, V_r_contour, 20)
        plt.colorbar(plot_4)
        
        ax = plt.subplot(3,3,5)
        ax.set_title('V_theta')
        plot_5 = ax.contourf(R, Z, V_theta_contour, 20)
        plt.colorbar(plot_5)
    
        ax = plt.subplot(3,3,6)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, V_z_contour, 20)
        plt.colorbar(plot_6)
    
        ax = plt.subplot(3,3,7)
        ax.set_title('rho')
        plot_7 = ax.contourf(R, Z, rho_contour, 20)
        plt.colorbar(plot_7)
    
        plt.show()
        
        # Post-processing
        rho = np.reshape(rho, (nr, 1))
        B_r = np.reshape(B_r, (nr, 1))
        B_theta = B_0 + np.reshape(B_theta, (nr, 1))
        B_z = np.reshape(B_z, (nr, 1))  
        V_r = np.reshape(V_r, (nr, 1))
        V_theta = np.reshape(V_theta, (nr, 1))
        V_z = np.reshape(V_z, (nr, 1))     
        rho = rho + rho_0
        B_z = np.reshape(B_z + B_Z0, (nr, 1))
        d_rB_dr = fd.ddr(1) @ (rr * B_theta)
        d_Bz_dr = fd.ddr(1) @ B_z    
        
        J_r = 1 / (4 * np.pi) * -1j * self.k * B_theta
        J_theta = 1 / (4 * np.pi) * (1j * self.k * B_r - d_Bz_dr)
        J_z = J_0 + 1 / (4 * np.pi * rr) * d_rB_dr 
        E_r_ideal = V_z * B_theta - V_theta * B_z
        E_theta_ideal = V_r * B_z - V_z * B_r
        E_z_ideal = V_theta * B_r - V_r * B_theta
        E_r_resistive = 4 * np.pi * D_eta * J_r
        E_theta_resistive = 4 * np.pi * D_eta * J_theta
        E_z_resistive = 4 * np.pi * D_eta * J_z
        E_r_hall = 4 * np.pi * D_H / rho * (J_theta * B_z - J_z * B_theta)
        E_theta_hall = 4 * np.pi * D_H / rho * (J_z * B_r - J_r * B_z)
        E_z_hall = 4 * np.pi * D_H / rho * (J_r * B_theta - J_theta * B_r)
        vort_theta = np.reshape(1j * self.k * V_r - (fd.ddr(1) @ V_z), (nr, ))
        vort_theta_contour = f1(z_osc[1: -1] * phase * vort_theta[1: -1])

        # V and vorticity
        plot = plt.contourf(R, Z, vort_theta_contour, 200, cmap='plasma')
        plt.colorbar(plot)
        
        d_vec = 15
        plt.title('Flow velocity and vorticity (V_theta = 0)')
        plt.xlabel('r')
        plt.ylabel('z')
        quiv = plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   V_r_contour[::d_vec, ::d_vec], V_z_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=4)
        
        plt.show()
        
        # 1D eigenvectors of J and E
#         ax = plt.subplot(4,3,1)
#         ax.set_title('J_r')
#         ax.plot(r[1: -1], f1(phase * J_r[1: -1]),
#                 r[1: -1], f2(phase * J_r[1: -1]))  
#               
#         ax = plt.subplot(4,3,2)
#         ax.set_title('J_theta')
#         ax.plot(r[1: -1], f1(phase * J_theta[1: -1]),
#                 r[1: -1], f2(phase * J_theta[1: -1]) )
#             
#         ax = plt.subplot(4,3,3)
#         ax.set_title('J_z')
#         ax.plot(r[1: -1], f1(phase * J_z[1: -1]),
#                 r[1: -1], f2(phase * J_z[1: -1]))    
#                     
#         ax = plt.subplot(4,3,4)
#         ax.set_title('E_r_ideal')
#         ax.plot(r[1: -1], f1(phase * E_r_ideal[1: -1]),
#                 r[1: -1], f2(phase * E_r_ideal[1: -1]) )
#             
#         ax = plt.subplot(4,3,5)
#         ax.set_title('E_theta_ideal')
#         ax.plot(r[1: -1], f1(phase * E_theta_ideal[1: -1]),
#                 r[1: -1], f2(phase * E_theta_ideal[1: -1]) )
#            
#         ax = plt.subplot(4,3,6)
#         ax.set_title('E_z_ideal')	
#         ax.plot(r[1: -1], f1(phase * E_z_ideal[1: -1]),
#                 r[1: -1], f2(phase * E_z_ideal[1: -1]) )
#             
#         ax = plt.subplot(4,3,7)
#         ax.set_title('E_r_resistive')
#         ax.plot(r[1: -1], f1(phase * E_r_resistive[1: -1]),
#                 r[1: -1], f2(phase * E_r_resistive[1: -1]) )
#             
#         ax = plt.subplot(4,3,8)
#         ax.set_title('E_theta_resistive')
#         ax.plot(r[1: -1], f1(phase * E_theta_resistive[1: -1]),
#                 r[1: -1], f2(phase * E_theta_resistive[1: -1]) )
#            
#         ax = plt.subplot(4,3,9)
#         ax.set_title('E_z_resistive')	
#         ax.plot(r[1: -1], f1(phase * E_z_resistive[1: -1]),
#                 r[1: -1], f2(phase * E_z_resistive[1: -1]) )
#                 
#         ax = plt.subplot(4,3,10)
#         ax.set_title('E_r_hall')
#         ax.plot(r[1: -1], f1(phase * E_r_hall[1: -1]),
#                 r[1: -1], f2(phase * E_r_hall[1: -1]) )
#             
#         ax = plt.subplot(4,3,11)
#         ax.set_title('E_theta_hall')
#         ax.plot(r[1: -1], f1(phase * E_theta_hall[1: -1]),
#                 r[1: -1], f2(phase * E_theta_hall[1: -1]) )
#            
#         ax = plt.subplot(4,3,12)
#         ax.set_title('E_z_hall')	
#         ax.plot(r[1: -1], f1(phase * E_z_hall[1: -1]),
#                 r[1: -1], f2(phase * E_z_hall[1: -1]) )
# 
#         plt.show()
        
        # 2D contour plots of J and E
        J_r = np.reshape(J_r, (nr, ))
        J_theta = np.reshape(J_theta, (nr, ))
        J_z = np.reshape(J_z, (nr, ))
        E_r_ideal = np.reshape(E_r_ideal, (nr, ))
        E_theta_ideal = np.reshape(E_theta_ideal, (nr, ))
        E_z_ideal = np.reshape(E_z_ideal, (nr, ))
        E_r_resistive = np.reshape(E_r_resistive, (nr, ))
        E_theta_resistive = np.reshape(E_theta_resistive, (nr, ))
        E_z_resistive = np.reshape(E_z_resistive, (nr, ))
        E_r_hall = np.reshape(E_r_hall, (nr, ))
        E_theta_hall = np.reshape(E_theta_hall, (nr, ))
        E_z_hall = np.reshape(E_z_hall, (nr, ))
        
        J_r_contour = f1(z_osc[1: -1] * phase * J_r[1: -1])
        J_theta_contour = f1(z_osc[1: -1] * phase * J_theta[1: -1])
        J_z_contour = f1(z_osc[1: -1] * phase * J_z[1: -1])
        E_r_ideal_contour = f1(z_osc[1: -1] * phase * E_r_ideal[1: -1])
        E_theta_ideal_contour = f1(z_osc[1: -1] * phase * E_theta_ideal[1: -1])
        E_z_ideal_contour = f1(z_osc[1: -1] * phase * E_z_ideal[1: -1])
        E_r_resistive_contour = f1(z_osc[1: -1] * phase * E_r_resistive[1: -1])
        E_theta_resistive_contour = f1(z_osc[1: -1] * phase * E_theta_resistive[1: -1])
        E_z_resistive_contour = f1(z_osc[1: -1] * phase * E_z_resistive[1: -1])
        E_r_hall_contour = f1(z_osc[1: -1] * phase * E_r_hall[1: -1])
        E_theta_hall_contour = f1(z_osc[1: -1] * phase * E_theta_hall[1: -1])
        E_z_hall_contour = f1(z_osc[1: -1] * phase * E_z_hall[1: -1])
        
        ax = plt.subplot(4,3,1)
        ax.set_title('J_r')
        plot_1 = ax.contourf(R, Z, J_r_contour, 20)
        plt.colorbar(plot_1)
    
        ax = plt.subplot(4,3,2)
        ax.set_title('J_theta')
        plot_2 = ax.contourf(R, Z, J_theta_contour, 20)
        plt.colorbar(plot_2)
    
        ax = plt.subplot(4,3,3)
        ax.set_title('J_z')
        plot_3 = ax.contourf(R, Z, J_z_contour, 20)
        plt.colorbar(plot_3)
    
        ax = plt.subplot(4,3,4)
        ax.set_title('E_r_ideal')
        plot_4 = ax.contourf(R, Z, E_r_ideal_contour, 20)
        plt.colorbar(plot_4)
        
        ax = plt.subplot(4,3,5)
        ax.set_title('E_theta_ideal')
        plot_5 = ax.contourf(R, Z, E_theta_ideal_contour, 20)
        plt.colorbar(plot_5)
    
        ax = plt.subplot(4,3,6)
        ax.set_title('E_z_ideal')
        plot_6 = ax.contourf(R, Z, E_z_ideal_contour, 20)
        plt.colorbar(plot_6)
        
        ax = plt.subplot(4,3,7)
        ax.set_title('E_r_resistive')
        plot_7 = ax.contourf(R, Z, E_r_resistive_contour, 20)
        plt.colorbar(plot_7)
        
        ax = plt.subplot(4,3,8)
        ax.set_title('E_theta_resistive')
        plot_8 = ax.contourf(R, Z, E_theta_resistive_contour, 20)
        plt.colorbar(plot_8)
    
        ax = plt.subplot(4,3,9)
        ax.set_title('E_z_resistive')
        plot_9 = ax.contourf(R, Z, E_z_resistive_contour, 20)
        plt.colorbar(plot_9)
        
        ax = plt.subplot(4,3,10)
        ax.set_title('E_r_hall')
        plot_10 = ax.contourf(R, Z, E_r_hall_contour, 20)
        plt.colorbar(plot_10)
        
        ax = plt.subplot(4,3,11)
        ax.set_title('E_theta_hall')
        plot_11 = ax.contourf(R, Z, E_theta_hall_contour, 20)
        plt.colorbar(plot_11)
    
        ax = plt.subplot(4,3,12)
        ax.set_title('E_z_hall')
        plot_12 = ax.contourf(R, Z, E_z_hall_contour, 20)
        plt.colorbar(plot_12)

        plt.show()
        
        # 2D quiver plots of J and E
        ax = plt.subplot(2,2,1)
        ax.set_title('J')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   J_r_contour[::d_vec, ::d_vec], J_z_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=200)

        ax = plt.subplot(2,2,2)
        ax.set_title('E_ideal')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_ideal_contour[::d_vec, ::d_vec], E_z_ideal_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=20)
    
        ax = plt.subplot(2,2,3)
        ax.set_title('E_resistive')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_resistive_contour[::d_vec, ::d_vec], E_z_resistive_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=100)
    
        ax = plt.subplot(2,2,4)
        ax.set_title('E_hall')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_hall_contour[::d_vec, ::d_vec], E_z_hall_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=50)
        
        plt.show()
        
        # Total electric field
        ax = plt.subplot(2, 2, 1)
        ax.set_title('E_r_total')
        plot_1 = ax.contourf(R, Z, E_r_ideal_contour + E_r_resistive_contour + E_r_hall_contour, 20, cmap='plasma')
        plt.colorbar(plot_1)
        
        ax = plt.subplot(2, 2, 2)
        ax.set_title('E_theta_total')
        plot_2 = ax.contourf(R, Z, E_theta_ideal_contour + E_theta_resistive_contour + E_theta_hall_contour, 20, cmap='plasma')
        plt.colorbar(plot_2)
        
        ax = plt.subplot(2, 2, 3)
        ax.set_title('E_z_total')
        plot_3 = ax.contourf(R, Z, E_z_ideal_contour + E_z_resistive_contour + E_z_hall_contour, 20, cmap='plasma')
        plt.colorbar(plot_3)
        
        ax = plt.subplot(2, 2, 4)
        ax.set_title('E_total')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   E_r_ideal_contour[::d_vec, ::d_vec] + E_r_resistive_contour[::d_vec, ::d_vec] + E_r_hall_contour[::d_vec, ::d_vec], 
                   E_z_ideal_contour[::d_vec, ::d_vec] + E_z_resistive_contour[::d_vec, ::d_vec] + E_z_hall_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=50)
        
        plt.show()
        
    def plot_eigenvalues(self):
        if self.evals is None:
            return

        plt.scatter(self.evals.real, self.evals.imag, s=1)
        plt.title('Omega')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()
