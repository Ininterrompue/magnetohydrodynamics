import numpy as np
from scipy.special import erf
from scipy.special import gamma
from scipy.special import gammainc
from scipy.linalg import eig
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


class MHDSystem:
    def __init__(self, N_r=50, N_ghost=1, r_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0):
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
    def __init__(self, equilibrium, k=1):
        self.equilibrium = equilibrium
        self.fd_operator = None
        self.fd_rhs = None
        self.evals = None
        self.evects = None
        self.set_z_mode(k)
        self.k = k
        self._sigma = None

    def set_z_mode(self, k):
        self.fd_operator = self.construct_operator(k)
        self.fd_rhs = self.construct_rhs()
        self.evals = None
        self.evects = None

    def construct_operator(self, k=1):
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
        m_rho_Vz = fd.diag(k * rho)
        
        m_Br_Vr = -B_Z0 * k * fd.diag_I()

        m_Btheta_Vr = -1j * fd.ddr_product(B)
        m_Btheta_Vtheta = -B_Z0 * k * fd.diag_I()
        m_Btheta_Vz = fd.diag(k * B)
        
        m_Bz_Vr = -1j * B_Z0 / rr * fd.ddr_product(rr)
        
        m_Vr_rho = -2j * fd.ddr(1) / rho
        m_Vr_Br = fd.diag(-B_Z0 * k / (4 * np.pi * rho))
        m_Vr_Btheta = -1j * fd.ddr_product(rr**2 * B) / (4 * np.pi * rho * rr**2)
        m_Vr_Bz = -1j * B_Z0 / (4 * np.pi * rho) * fd.ddr(1)
        
        m_Vtheta_Br = fd.diag(1j * (fd.ddr(1) @ (rr * B)) / (4 * np.pi * rho * rr))
        m_Vtheta_Btheta = fd.diag(-B_Z0 * k / (4 * np.pi * rho))
        
        m_Vz_rho = fd.diag(2.0 * k / rho)
        m_Vz_Btheta = fd.diag(k * B / (4.0 * np.pi * rho))
       
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
        m_Br_Br = m_Br_Br + 1j * D_eta * ((1 / rr * fd.ddr(1) + fd.ddr(2)) - fd.diag(k**2 + 1 / rr**2))
        m_Btheta_Btheta = m_Btheta_Btheta + 1j * D_eta * ((1 / rr * fd.ddr(1) + fd.ddr(2)) - fd.diag(k**2 + 1 / rr**2))
        m_Bz_Bz = m_Bz_Bz + 1j * D_eta * ((1 / rr * fd.ddr(1) + fd.ddr(2)) - fd.diag(k**2)) 
        
        # Hall term
        m_Br_Br = m_Br_Br + D_H * fd.diag(-k / (rr * rho) * (fd.ddr(1) @ (rr * B)))
        m_Br_Btheta = m_Br_Btheta + D_H * fd.diag(-1j * B_Z0 * k**2 / rho)
        m_Btheta_rho = m_Btheta_rho + D_H * fd.diag(k * B / (rho**2 * rr) * (fd.ddr(1) @ (rr * B)))
        m_Btheta_Br = m_Btheta_Br + D_H * fd.diag(1j * B_Z0 * k**2 / rho)
        m_Btheta_Btheta = m_Btheta_Btheta + D_H * fd.diag(k * B * (fd.ddr(1) @ (1 / rho)) - 2 * B * k / (rr * rho))
        m_Btheta_Bz = m_Btheta_Bz + D_H * ((-B_Z0 * k / rho) * fd.ddr(1))
        m_Bz_Br = m_Bz_Br + D_H * (-1j / (rr * rho) * (fd.ddr(1) @ (rr * B)) * fd.ddr(1) - fd.diag(1j / (rr * rho) * (fd.ddr(2) @ (rr * B)))
                                   - fd.diag(1j / rr * (fd.ddr(1) @ (1 / rho)) * (fd.ddr(1) @ (rr * B))))    
        m_Bz_Btheta = m_Bz_Btheta + D_H * (B_Z0 * k / (rr * rho) * fd.ddr_product(rr) + fd.diag(B_Z0 * k * (fd.ddr(1) @ (1 / rho))))
        
        # Electron pressure term (Terms including B_Z0 have not been added yet)
        m_Btheta_rho = m_Btheta_rho + D_P * fd.diag(k * (1 / rho**2 * (fd.ddr(1) @ rho) + (fd.ddr(1) @ (1 / rho))))
        
        # Boundary conditions
        m_rho_rho       = m_rho_rho       + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m_Br_Br         = m_Br_Br         + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Btheta_Btheta = m_Btheta_Btheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Bz_Bz         = m_Bz_Bz         + fd.lhs_bc('derivative') + fd.rhs_bc('derivative')
        m_Vr_Vr         = m_Vr_Vr         + fd.lhs_bc('value')      + fd.rhs_bc('derivative')
        m_Vtheta_Vtheta = m_Vtheta_Vtheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Vz_Vz         = m_Vz_Vz         + fd.lhs_bc('derivative') + fd.rhs_bc('value')
                      
        M = np.block([[m_rho_rho,    m0,          m0,              m0,          m_rho_Vr,    m0,              m_rho_Vz   ], 
                      [m0,           m_Br_Br,     m_Br_Btheta,     m0,          m_Br_Vr,     m0,              m0         ],
                      [m_Btheta_rho, m_Btheta_Br, m_Btheta_Btheta, m_Btheta_Bz, m_Btheta_Vr, m_Btheta_Vtheta, m_Btheta_Vz],
                      [m0,           m_Bz_Br,     m_Bz_Btheta,     m_Bz_Bz,     m_Bz_Vr,     m0,              m0         ],
                      [m_Vr_rho,     m_Vr_Br,     m_Vr_Btheta,     m_Vr_Bz,     m_Vr_Vr,     m0,              m0         ],
                      [m0,           m_Vtheta_Br, m_Vtheta_Btheta, m0,          m0,          m_Vtheta_Vtheta, m0         ],
                      [m_Vz_rho,     m0,          m_Vz_Btheta,     m0,          m0,          m0,              m_Vz_Vz    ]])
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
                                           sigma=5j, which='LI', return_eigenvectors=True)
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
        return eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=5j, which='LI', return_eigenvectors=False).imag

    # ith mode by magnitude of imaginary part
    def plot_mode(self, i):
        if self.evects is None:
            return

        nr = self.equilibrium.sys.grid.N
        r = self.equilibrium.sys.grid.r
        z = self.equilibrium.sys.grid.r
        zz = self.equilibrium.sys.grid.rr
        rho_0 = self.equilibrium.rho
        B_0 = self.equilibrium.B

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
        plot_1 = ax.contourf(R, Z, B_r_contour, 100)
        plt.colorbar(plot_1)
    
        ax = plt.subplot(3,3,2)
        ax.set_title('B_theta')
        plot_2 = ax.contourf(R, Z, B_theta_contour, 100)
        plt.colorbar(plot_2)
    
        ax = plt.subplot(3,3,3)
        ax.set_title('B_z')
        plot_3 = ax.contourf(R, Z, B_z_contour, 100)
        plt.colorbar(plot_3)
    
        ax = plt.subplot(3,3,4)
        ax.set_title('V_r')
        plot_4 = ax.contourf(R, Z, V_r_contour, 100)
        plt.colorbar(plot_4)
        
        ax = plt.subplot(3,3,5)
        ax.set_title('V_theta')
        plot_5 = ax.contourf(R, Z, V_theta_contour, 100)
        plt.colorbar(plot_5)
    
        ax = plt.subplot(3,3,6)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, V_z_contour, 100)
        plt.colorbar(plot_6)
    
        ax = plt.subplot(3,3,7)
        ax.set_title('rho')
        plot_7 = ax.contourf(R, Z, rho_contour, 100)
        plt.colorbar(plot_7)
    
        plt.show()
    
        # 2D quiver plot of V
        R, Z = np.meshgrid(r[1: -1], z[1: -1])
        d_vec = 10
        plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   V_r_contour[::d_vec, ::d_vec], V_z_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=5)
        plt.title('Flow velocity')
        plt.xlabel('r')
        plt.ylabel('z')
        plt.show()

    def plot_eigenvalues(self):
        if self.evals is None:
            return

        plt.scatter(self.evals.real, self.evals.imag, s=1)
        plt.title('Omega')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show()
