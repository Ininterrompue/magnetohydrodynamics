import numpy as np
from mhdsystem import Const
from scipy.special import gamma, gammainc
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from aaa_source import apply_along_axis as aaa
import time

        
class AnalyticalEquilibriumCyl:
    def __init__(self, sys, p_exp, v_exp):
        self.sys = sys
        self.p_exp = p_exp
        self.v_exp = v_exp
        # self.n_ghost = self.sys.grid_r.n_ghost
        self.p = self.compute_p()
        self.rho = self.compute_rho()
        self.b = self.compute_b()
        self.j = self.compute_j()
        self.v = self.compute_v()

    def compute_p(self):
        # Initial pressure profile P = P0 * (0.05 * exp(-r^n))
        # p_0 = Const.I**2 / (np.pi * Const.r_0**2 * Const.c**2)
        p = Const.P_0 * (0.05 + np.exp(-self.sys.grid_r.rr**self.p_exp))
        return p

    def compute_rho(self):
        # Equation of state. Initial condition of T=1 uniform
        rho = Const.m_i * self.p / (2 * Const.T_0)
        return rho

    def compute_b(self):
        n = self.p_exp
        rr = self.sys.grid_r.rr
        # p_0 = Const.I**2 / (np.pi * Const.r_0**2 * Const.c**2)
        
        # Magnetic pressure B^2/8pi
        b_pressure = Const.P_0 * (2 / (n * rr**2) * gamma(2 / n) * gammainc(2 / n, rr**n) - np.exp(-rr**n))
        b = np.sqrt(8 * np.pi) * np.sign(b_pressure) * np.sqrt(np.abs(b_pressure))
        return b

    def compute_j(self):
        n = self.p_exp
        rr = self.sys.grid_r.rr
        
        num = np.exp(-rr**n) * np.sqrt(2 * np.pi) * n * rr**(n - 1)
        insqrt = 2 / (n * rr**2) * gamma(2 / n) * gammainc(2 / n, rr**n) - np.exp(-rr**n)
        denom = np.sign(insqrt) * np.sqrt(np.abs(insqrt))
        j = Const.c / (4 * np.pi) * num / denom
        return j

    def compute_v(self):
        v = self.sys.V_Z0 * (np.exp(-self.sys.grid_r.rr**self.v_exp) - 0.5)
        return v
        

class NumericalEquilibriumCyl:
    def __init__(self, sys, p_exp, v_exp):
        self.sys = sys
        self.p_exp = p_exp
        self.v_exp = v_exp
        # self.n_ghost = self.sys.grid_r.n_ghost
        self.p = self.compute_p()
        self.rho = self.compute_rho()
        self.b = self.compute_b()
        self.j = self.compute_j()
        self.v = self.compute_v()
    
    def compute_p(self):
        # Initial pressure profile P = P0 * (0.05 * exp(-r^n))
        p_0 = Const.I**2 / (np.pi * Const.r_0**2 * Const.c**2)
        p = p_0 * (0.05 + np.exp(-self.sys.grid_r.rr**self.p_exp))
        return p

    def compute_rho(self):
        # Equation of state. Initial condition of T=1 uniform
        rho = Const.m_i * self.p / (2 * Const.T_0)
        return rho
        
    def compute_b(self):
        # solve equilibrium ideal MHD equation
        # (d/dr)(P + B^2/8pi) + 2/r * B^2/8pi = 0
        lhs = self.sys.fd.ddr(1) + self.sys.fd.diag(2 / self.sys.grid_r.r)
        rhs = -(self.sys.fd.ddr(1) @ self.p)
        
        rhs[0]  = 0
        rhs[-1] = 0
        lhs[0, 0] = 1
        lhs[0, 1] = 1
        lhs[-1, -1] = -self.sys.grid_r.r[-1] / self.sys.grid_r.r[-2]
        lhs[-1, -2] = 1

        b_pressure = np.linalg.solve(lhs, rhs)
        b = np.sqrt(8 * np.pi) * np.sign(b_pressure) * np.sqrt(np.abs(b_pressure))
        return b
    
    def compute_j(self):  
        dv = self.sys.fd.ddr_product(self.sys.grid_r.rr)
        j = 1 / self.sys.grid_r.rr * (dv @ self.b) * Const.c / (4 * np.pi)
        return j

    def compute_v(self):
        v = self.sys.V_Z0 * (np.exp(-self.sys.grid_r.rr**self.v_exp) - 0.5)
        return v
        

class LinearCyl:
    def __init__(self, equ, k=1, m=0, rosh=5/3):
        self.equ = equ
        self.k = k
        self.m = m
        self.rosh = rosh
        self.fd_operator = None
        self.fd_rhs = None
        self.evals = None
        self.evecs = None
        self.set_z_mode(k, m)

    def set_z_mode(self, k, m):
        self.fd_operator = self.construct_operator(k, m)
        self.fd_rhs = self.construct_rhs()
        self.evals = None
        self.evecs = None

    def construct_operator(self, k=1, m=0):
        fd  = self.equ.sys.fd
        r   = self.equ.sys.grid_r.r
        rr  = self.equ.sys.grid_r.rr
        rho = self.equ.rho
        p   = self.equ.p
        B   = self.equ.b
        VZ  = self.equ.v

        D_eta = self.equ.sys.D_eta
        D_H   = self.equ.sys.D_H
        D_P   = self.equ.sys.D_P
        B_Z0  = self.equ.sys.B_Z0
        rosh  = self.rosh

        # Ideal MHD
        m_rho_rho    = fd.diag(k * VZ)
        m_rho_Vr     = -1j * fd.ddr_product(rr * rho) / rr
        m_rho_Vtheta = fd.diag(m * rho / rr)
        m_rho_Vz     = fd.diag(k * rho)
        
        m_Br_Vr = -B_Z0 * k * fd.diag_I() - fd.diag(m * B / rr)

        m_Btheta_Btheta = fd.diag(k * VZ)
        m_Btheta_Vr     = -1j * fd.ddr_product(B)
        m_Btheta_Vtheta = -B_Z0 * k * fd.diag_I()
        m_Btheta_Vz     = fd.diag(k * B)
        
        m_Bz_Vr     = -1j * B_Z0 / rr * fd.ddr_product(rr)
        m_Bz_Vtheta = fd.diag(m * B_Z0 / rr)
        m_Bz_Vz     = fd.diag(-m * B / rr)
        
        m_Vr_p      = -1j * fd.ddr(1) / rho
        m_Vr_Br     = fd.diag(-B_Z0 * k / (4 * np.pi * rho) - m * B / (4 * np.pi * rho * rr))
        m_Vr_Btheta = -1j * fd.ddr_product(rr**2 * B) / (4 * np.pi * rho * rr**2)
        m_Vr_Bz     = -1j * B_Z0 / (4 * np.pi * rho) * fd.ddr(1)
        
        m_Vtheta_rho    = fd.diag(2 * m / (rho * rr))
        m_Vtheta_Br     = fd.diag(1j * (fd.ddr(1) @ (rr * B)) / (4 * np.pi * rho * rr))
        m_Vtheta_Btheta = fd.diag(-B_Z0 * k / (4 * np.pi * rho))
        m_Vtheta_Bz     = fd.diag(m * B_Z0 / (4 * np.pi * rho * rr))
        
        m_Vz_p      = fd.diag(k / rho)
        m_Vz_Btheta = fd.diag(k * B / (4 * np.pi * rho))
        m_Vz_Bz     = fd.diag(-m * B / (4 * np.pi * rho * rr))

        m_p_p  = fd.diag(k * VZ)
        m_p_Vr = fd.diag(-1j * (fd.ddr(1) @ p)) - rosh * 1j * p / rr * fd.ddr_product(rr)
        m_p_Vz = fd.diag(rosh * p * k)
       
        m0              = fd.zeros()
        m_Br_Br         = fd.zeros()
        m_Br_Btheta     = fd.zeros()
        m_Btheta_rho    = fd.zeros()
        m_Btheta_Br     = fd.zeros()
        m_Btheta_Bz     = fd.zeros()
        m_Btheta_p      = fd.zeros()
        m_Bz_Br         = fd.zeros()
        m_Bz_Btheta     = fd.zeros()
        m_Bz_Bz         = fd.zeros()
        m_Vr_Vr         = fd.zeros()
        m_Vtheta_Vtheta = fd.zeros()
        m_Vz_Vz         = fd.zeros()
        
        # Resistive term
        m_Br_Br         = m_Br_Br + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
        m_Br_Btheta     = m_Br_Btheta + D_eta * fd.diag(2 * m / rr**2)
        m_Btheta_Btheta = m_Btheta_Btheta + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2 + 1j / rr**2))
        m_Btheta_Br     = m_Btheta_Br + D_eta * fd.diag(-2 * m / rr**2)
        m_Bz_Bz         = m_Bz_Bz + D_eta * (1j / rr * fd.ddr(1) + 1j * fd.ddr(2) - fd.diag(1j * m**2 / rr**2 + 1j * k**2)) 
        
        # Hall term (m = 0 only)
        m_Br_Br         = m_Br_Br + D_H * fd.diag(-k / (rr * rho) * (fd.ddr(1) @ (rr * B)))
        m_Br_Btheta     = m_Br_Btheta + D_H * fd.diag(-1j * B_Z0 * k**2 / rho)    
        m_Btheta_rho    = m_Btheta_rho + D_H * fd.diag(-8 * np.pi * k / rho**2 * (fd.ddr(1) @ rho))
        m_Btheta_Br     = m_Btheta_Br + D_H * fd.diag(1j * B_Z0 * k**2 / rho)
        m_Btheta_Btheta = m_Btheta_Btheta + D_H * fd.diag(-k * B / rho**2 * (fd.ddr(1) @ rho) - 2 * k * B / (rho * rr))
        m_Btheta_Bz     = m_Btheta_Bz + D_H * ((-B_Z0 * k / rho) * fd.ddr(1))
        
        m_Bz_Br     = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B)) - 1j / (rho * rr) * (fd.ddr(2) @ (rr * B)))
                                       - 1j / (rho * rr) * (fd.ddr(1) @ (rr * B)) * fd.ddr(1))
        # m_Bz_Br     = m_Bz_Br + D_H * (fd.diag(1j / (rho**2 * rr) * (fd.ddr(1) @ rho) * (fd.ddr(1) @ (rr * B))) - 1j / (rho * rr) * (fd.ddr_product(fd.ddr(1) @ (rr * B))))
        m_Bz_Btheta = m_Bz_Btheta + D_H * (B_Z0 * k / (rr * rho) * fd.ddr_product(rr) - fd.diag(k * B_Z0 / rho**2 * (fd.ddr(1) @ rho)))
        
        # Electron pressure term
        m_Btheta_rho = m_Btheta_rho + D_P * fd.diag(-k / rho**2 * (fd.ddr(1) @ p))
        m_Btheta_p   = m_Btheta_p + D_P * fd.diag(k / rho**2 * (fd.ddr(1) @ rho))
        
        # Boundary conditions
        m_rho_rho       = m_rho_rho       + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m_Br_Br         = m_Br_Br         + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Btheta_Btheta = m_Btheta_Btheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Bz_Bz         = m_Bz_Bz         + fd.lhs_bc('derivative') + fd.rhs_bc('derivative')
        m_Vr_Vr         = m_Vr_Vr         + fd.lhs_bc('value')      + fd.rhs_bc('derivative')
        m_Vtheta_Vtheta = m_Vtheta_Vtheta + fd.lhs_bc('value')      + fd.rhs_bc('value')
        m_Vz_Vz         = m_Vz_Vz         + fd.lhs_bc('derivative') + fd.rhs_bc('value')
        m_p_p           = m_p_p           + fd.lhs_bc('derivative') + fd.rhs_bc('value')
                      
        M = np.block([[m_rho_rho,    m0,          m0,              m0,          m_rho_Vr,    m_rho_Vtheta,    m_rho_Vz,    m0        ], 
                      [m0,           m_Br_Br,     m_Br_Btheta,     m0,          m_Br_Vr,     m0,              m0 ,         m0        ],
                      [m_Btheta_rho, m_Btheta_Br, m_Btheta_Btheta, m_Btheta_Bz, m_Btheta_Vr, m_Btheta_Vtheta, m_Btheta_Vz, m_Btheta_p],
                      [m0,           m_Bz_Br,     m_Bz_Btheta,     m_Bz_Bz,     m_Bz_Vr,     m_Bz_Vtheta,     m_Bz_Vz,     m0        ],
                      [m0,           m_Vr_Br,     m_Vr_Btheta,     m_Vr_Bz,     m_Vr_Vr,     m0,              m0,          m_Vr_p    ],
                      [m_Vtheta_rho, m_Vtheta_Br, m_Vtheta_Btheta, m_Vtheta_Bz, m0,          m_Vtheta_Vtheta, m0,          m0        ],
                      [m0,           m0,          m_Vz_Btheta,     m_Vz_Bz,     m0,          m0,              m_Vz_Vz,     m_Vz_p    ],
                      [m0,           m0,          m0,              m0,          m_p_Vr,      m0,              m_p_Vz,      m_p_p     ]])
        return M

    def construct_rhs(self):
        # Generalized eigenvalue problem matrix
        nr = self.equ.sys.grid_r.nr
        G = np.identity(8 * nr)
        G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
        G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = 0
        G[4*nr - 1, 4*nr - 1] = G[4*nr, 4*nr] = G[5*nr - 1, 5*nr - 1] = 0
        G[5*nr, 5*nr] = G[6*nr - 1, 6*nr - 1] = G[6*nr - 6*nr] = 0
        G[7*nr - 1, 7*nr - 1] = G[7*nr, 7*nr] = G[-1, -1] = 0
        return G

    def solve(self, num_modes=None):
        if num_modes:
            self.evals, self.evecs = eigs(self.fd_operator, k=num_modes, M=self.fd_rhs, sigma=2j, which='LI', return_eigenvectors=True)
        else:
            self.evals, self.evecs = eig(self.fd_operator, self.fd_rhs)

    def solve_for_gamma(self):
        return eigs(self.fd_operator, k=1, M=self.fd_rhs, sigma=1j, which='LI', return_eigenvectors=False).imag

    # Procedure to remove the highly oscillatory + growing (numerical?) modes
    def remove_oscillatory(self):
        evals = []
        evecs = []
        for i in range(len(self.evals)):
            if np.abs(self.evals[i].real) < 30:
                evals.append(self.evals[i])
                evecs.append(self.evecs[:, i])
        self.evals = np.asarray(evals)
        self.evecs = np.asarray(evecs)
        self.evecs = self.evecs.T

    # Procedure to only keep the growing and damping modes
    def remove_baddies(self):
        evals = []
        evecs = []
        for i in range(len(self.evals)):
            if np.abs(self.evals[i].real) < 1e-6:
                evals.append(self.evals[i])
                evecs.append(self.evecs[:, i])
        self.evals = np.asarray(evals)
        self.evecs = np.asarray(evecs)
        self.evecs = self.evecs.T
        
        index = np.argsort(self.evals.imag)
        omega = self.evals[index[-1]]
        return omega.imag
        
    def extract_from_evec(self, i=-1, epsilon=0.05):
        if self.evecs is None:
            return
            
        self.i = i
        self.epsilon = epsilon
            
        fd    = self.equ.sys.fd
        nr    = self.equ.sys.grid_r.nr
        r     = self.equ.sys.grid_r.r
        rr    = self.equ.sys.grid_r.rr
        z     = self.equ.sys.grid_z.r
        zz    = self.equ.sys.grid_z.rr
        rho_0 = self.equ.rho
        p_0   = self.equ.p
        B_0   = self.equ.b
        B_Z0  = self.equ.sys.B_Z0
        
        index = np.argsort(self.evals.imag)
        omega = self.evals[index[i]]
        v_omega = self.evecs[:, index[i]]
        
        # Correct for overall phase factor
        rho   = v_omega[0: nr]
        phase = np.exp(-1j * np.angle(rho[0]))
        
        # Extract eigenmodes
        rho    = epsilon * phase * rho
        Br     = epsilon * phase * v_omega[1*nr: 2*nr]
        Btheta = epsilon * phase * v_omega[2*nr: 3*nr]
        Bz     = epsilon * phase * v_omega[3*nr: 4*nr]
        Vr     = epsilon * phase * v_omega[4*nr: 5*nr]
        Vtheta = epsilon * phase * v_omega[5*nr: 6*nr]
        Vz     = epsilon * phase * v_omega[6*nr: 7*nr]
        p      = epsilon * phase * v_omega[7*nr: 8*nr]
        
#         p_0    = np.reshape(p_0, (nr, ))
#         rho_0  = np.reshape(rho_0, (nr, ))

        rho    = np.reshape(rho, (nr, 1))
        Br     = np.reshape(Br, (nr, 1))
        Btheta = np.reshape(Btheta, (nr, 1))
        Bz     = np.reshape(Bz, (nr, 1))
        Vr     = np.reshape(Vr, (nr, 1))
        Vtheta = np.reshape(Vtheta, (nr, 1))
        Vz     = np.reshape(Vz, (nr, 1))
        p      = np.reshape(p, (nr, 1))
        
        temp   = (p + p_0) / (2 * (rho + rho_0))
        temp_1 = (p - 2 * rho) / (2 * (rho + rho_0))
        temp   = np.reshape(temp, (nr, 1))
        temp_1 = np.reshape(temp_1, (nr, 1))
        
        return rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1
        
    def compute_divV(self):
        fd = self.equ.sys.fd
        rr = self.equ.sys.grid_r.rr
        nr = self.equ.sys.grid_r.nr
        zz = self.equ.sys.grid_z.rr
        VZ = self.equ.v
        z_osc = np.exp(1j * self.k * zz)
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        divV = 1 / rr * (fd.ddr(1) @ (rr * Vr)) + 1j * self.k * (VZ + Vz)
        divV = np.reshape(divV, (nr, ))
        divV_contour = self.epsilon * np.real(z_osc * divV)
        return divV_contour

    def compute_vort(self):
        fd = self.equ.sys.fd
        nr = self.equ.sys.grid_r.nr
        zz = self.equ.sys.grid_z.rr
        VZ = self.equ.v
        z_osc = np.exp(1j * self.k * zz)
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        vort_theta = np.reshape(1j * self.k * Vr - (fd.ddr(1) @ (VZ + Vz)), (nr, ))
        vort_theta[0: 2] = 0 # Removes edge effects
        vort_theta_contour = self.epsilon * np.real(z_osc * vort_theta)
        return vort_theta_contour
 
    def compute_currents(self):
        fd = self.equ.sys.fd
        rr = self.equ.sys.grid_r.rr
        nr = self.equ.sys.grid_r.nr
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        Jr     = Const.c / (4 * np.pi) * -1j * self.k * Btheta
        Jtheta = Const.c / (4 * np.pi) * (1j * self.k * Br - (fd.ddr(1) @ Bz))
        Jz1    = Const.c / (4 * np.pi * rr) * (fd.ddr(1) @ (rr * Btheta))
        # elif coordinates == 'Cartesian':
        #     Jz1 = Const.c / (4 * np.pi) * (fd.ddr(1) @ B_1)
        return Jr, Jtheta, Jz1
        
    def compute_E_ideal(self):
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        B_0  = self.equ.b
        B_Z0 = self.equ.sys.B_Z0
        nr   = self.equ.sys.grid_r.nr
        Btheta = B_0 + Btheta
        Bz     = B_Z0 * np.reshape(np.ones(nr), (nr, 1)) + Bz
        Er_ideal     = Vz * Btheta - Vtheta * Bz
        Etheta_ideal = Vr * Bz - Vz * Br
        Ez_ideal     = Vtheta * Br - Vr * Btheta
        return Er_ideal, Etheta_ideal, Ez_ideal
        
    def compute_E_resistive(self):
        D_eta = self.equ.sys.D_eta
        J_0   = self.equ.j       
        Jr, Jtheta, Jz1 = self.compute_currents()
        Jz = J_0 + Jz1
        Er_resistive     = 4 * np.pi * D_eta / Const.c**2 * Jr
        Etheta_resistive = 4 * np.pi * D_eta / Const.c**2 * Jtheta
        Ez0_resistive    = 4 * np.pi * D_eta / Const.c**2 * J_0
        Ez1_resistive    = 4 * np.pi * D_eta / Const.c**2 * Jz1
        return Er_resistive, Etheta_resistive, Ez0_resistive, Ez1_resistive
        
    def compute_E_hall(self):
        nr    = self.equ.sys.grid_r.nr
        D_H   = self.equ.sys.D_H
        B_Z0  = self.equ.sys.B_Z0
        rho_0 = self.equ.rho
        B_0   = self.equ.b
        J_0   = self.equ.j
        Jr, Jtheta, Jz1 = self.compute_currents()
        Jz = J_0 + Jz1
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        rho    = rho_0 + rho
        Btheta = B_0 + Btheta
        Bz     = B_Z0 * np.reshape(np.ones(nr), (nr, 1)) + Bz
        Er0_hall    = 4 * np.pi * D_H / Const.c**2 / rho_0 * (-J_0 * B_0)
        Er1_hall    = 4 * np.pi * D_H / Const.c**2 / rho * (Jtheta * Bz - Jz * Btheta) - 4 * np.pi * D_H / rho_0 * (-J_0 * B_0)
        Etheta_hall = 4 * np.pi * D_H / Const.c**2 / rho * (Jz * Br - Jr * Bz)
        Ez_hall     = 4 * np.pi * D_H / Const.c**2 / rho * (Jr * Btheta - Jtheta * Br)
        return Er0_hall, Er1_hall, Etheta_hall, Ez_hall
        
    def compute_E_pressure(self): 
        fd    = self.equ.sys.fd 
        D_P   = self.equ.sys.D_P
        rho_0 = self.equ.rho
        p_0   = self.equ.p
        nr    = self.equ.sys.grid_r.nr
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.extract_from_evec(self.i, self.epsilon)
        rho     = rho_0 + rho
        p_total = p_0 + p
        Er0_pressure    = -D_P / Const.c / rho_0 * (fd.ddr(1) @ p_0)
        Er1_pressure    = -D_P / Const.c / rho * (fd.ddr(1) @ p_total) - Er0_pressure
        Etheta_pressure = np.zeros(nr)
        Ez_pressure     = -D_P / Const.c / rho * 1j * self.k * p
        return Er0_pressure, Er1_pressure, Etheta_pressure, Ez_pressure
        
class EvolveCyl:
    def __init__(self, sys, equ, lin, k=1, rosh=5/3, D_nu=0):
        self.sys = sys
        self.equ = equ
        self.lin = lin
        self.k = k
        self.rosh = rosh
        self.D_nu = D_nu
        self.B, self.Vr, self.Vz, self.rho, self.p, self.T = self.seed()
        
    def seed(self):
        nr = self.sys.grid_r.nr
        nz = self.sys.grid_z.nr
        zz = self.sys.grid_z.rr

        pert = 1 - 0.01 * np.cos(self.k * zz)
        B    = self.equ.b * np.ones(nz).T
        Vr   = np.zeros((nr, nz))
        Vz   = np.zeros((nr, nz))
        rho  = self.equ.rho * pert.T
        p    = self.equ.p * pert.T
        T    = Const.T_0 * np.ones((nr, nz))

#         B   = np.zeros((nr, nz))
#         Vr  = np.zeros((nr, nz))
#         Vz  = np.zeros((nr, nz))
#         rho = np.zeros((nr, nz))
#         p   = np.zeros((nr, nz))
#         T   = np.zeros((nr, nz))
#         B[2:-2,2:-2], Vr[2:-2,2:-2], Vz[2:-2,2:-2], rho[2:-2,2:-2], p[2:-2,2:-2], T[2:-2,2:-2] = self.lin.plot_VB(-1, epsilon=1000)
#         B   = B.T
#         Vr  = Vr.T
#         Vz  = Vz.T
#         rho = rho.T
#         p   = p.T
#         T   = T.T
    
        return B, Vr, Vz, rho, p, T 
        
    def CFL(self, courant, B, Vr, Vz, rho, T):
        # This CFL condition still doesn't give the right dt
        v_fluid = np.amax(np.sqrt(np.abs(Vr)**2 + np.abs(Vz)**2))
        v_alfven2 = np.amax(np.abs(B)**2 / (4 * np.pi * rho))
        v_sound2 = np.amax(2 * self.rosh * T / Const.m_i)
        v_magnetosonic = np.sqrt(v_alfven2 + v_sound2)
        v_courant = v_fluid + v_magnetosonic
        dt = self.sys.grid_r.dr / v_courant * 1e-1 * courant
        return dt
        
    def evolve_Euler(self, courant=0.8, t_max=0):
        nr = self.sys.grid_r.nr
        r  = self.sys.grid_r.r
        rr = self.sys.grid_r.rr * np.ones(nr).T
        z  = self.sys.grid_z.r
        zz = self.sys.grid_z.rr
        dr = self.sys.grid_r.dr
        dz = self.sys.grid_z.dr
        D_eta   = self.sys.D_eta
        D_nu    = self.D_nu
        rosh    = self.rosh
        B   = self.B
        Vr  = self.Vr
        Vz  = self.Vz
        rho = self.rho
        p   = self.p
        T   = self.T
        
        t = 0
        iteration = 0
        counter = 0
        dim = 0
        
        print('Simulation seeded.')
        while t < t_max:
            iteration += 1
            counter += 1
            rho_temp = rho.copy()
            B_temp   = B.copy()
            Vr_temp  = Vr.copy()  
            Vz_temp  = Vz.copy()
            p_temp   = p.copy()
            T_temp   = T.copy()
        
            self.dt = self.CFL(courant, B_temp, Vr_temp, Vz_temp, rho_temp, T_temp)
            dt = self.dt
            if dt < 1e-9:
                print('Solution has not converged')
                break
                
            t += dt
            if counter == 100:
                dim = dim + 1
                counter = 0 
                print(iteration, t, dt)  
                
            # Finite difference procedure                      
            rho[1: -1, 1: -1] = (rho_temp[1: -1, 1: -1] - dt / (2 * dr * rr[1: -1, 1: -1]) * (rr[2: , 1: -1] * rho_temp[2: , 1: -1] * Vr_temp[2: , 1: -1] 
                                                        - rr[: -2, 1: -1] * rho_temp[: -2, 1: -1] * Vr_temp[: -2, 1: -1])
                                                        - dt / (2 * dz) * (rho_temp[1: -1, 2: ] * Vz_temp[1: -1, 2: ]
                                                        - rho_temp[1: -1, : -2] * Vz_temp[1: -1, : -2]))
            B[1: -1, 1: -1] = (B_temp[1: -1, 1: -1] - dt / (2 * dr) * (Vr_temp[2: , 1: -1] * B_temp[2: , 1: -1] - Vr_temp[: -2, 1: -1] * B_temp[: -2, 1: -1])
                                                    - dt / (2 * dz) * (Vz_temp[1: -1, 2: ] * B_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2] * B_temp[1: -1, : -2])
                                                    + dt * D_eta * (1 / dr**2 * (B_temp[2:, 1: -1] - 2 * B_temp[1: -1, 1: -1] + B_temp[:-2, 1: -1])
                                                    + 1 / (rr[1: -1, 1: -1] * 2 * dr) * (B_temp[2:, 1: -1] - B_temp[:-2, 1: -1])
                                                    + 1 / dz**2 * (B_temp[1: -1, 2:] - 2 * B_temp[1: -1, 1: -1] + B_temp[1: -1, :-2])
                                                    - B_temp[1: -1, 1: -1] / rr[1: -1, 1: -1]**2))
                                                    
            Vr[1: -1, 1: -1] = (Vr_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (Vr_temp[2: , 1: -1] - Vr_temp[: -2, 1: -1])
                                                      - dt / (2 * dz) * Vz_temp[1: -1, 1: -1] * (Vr_temp[1: -1, 2: ] - Vr_temp[1: -1, : -2]) 
                                                      - dt / (2 * dr * rho_temp[1: -1, 1: -1]) * (p_temp[2: , 1: -1] - p_temp[: -2, 1: -1]) 
                                                      - B_temp[1: -1, 1: -1] * dt / (4 * np.pi * rho_temp[1: -1, 1: -1] * rr[1: -1, 1: -1] * 2 * dr) 
                                                      * (rr[2: , 1: -1] * B_temp[2: , 1: -1] - rr[: -2, 1: -1] * B_temp[: -2, 1: -1])
                                                      + D_nu / rho_temp[1: -1, 1: -1] * (1 / dr**2 * (Vr_temp[2:, 1: -1] - 2 * Vr_temp[1: -1, 1: -1] + Vr_temp[:-2, 1: -1]) 
                                                      + 1 / (rr[1: -1, 1: -1] * 2 * dr) * (Vr_temp[2:, 1: -1] - Vr_temp[:-2, 1: -1])
                                                      + 1 / dz**2 * (Vr_temp[1: -1, 2:] - 2 * Vr_temp[1: -1, 1: -1] + Vr_temp[1: -1, :-2]) 
                                                      - Vr_temp[1: -1, 1: -1] / rr[1: -1, 1: -1]**2))
                                                      
            Vz[1: -1, 1: -1] = (Vz_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (Vz_temp[2: , 1: -1] - Vz_temp[: -2, 1: -1])
                                                      - dt / (2 * dz) * Vz_temp[1: -1, 1: -1] * (Vz_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2])
                                                      - dt / (2 * dz * rho_temp[1: -1, 1: -1]) * (p_temp[1: -1, 2: ] - p_temp[1: -1, : -2])
                                                      - B_temp[1: -1, 1: -1] * dt / (4 * np.pi * rho_temp[1: -1, 1: -1] * 2 * dz) * (B_temp[1: -1, 2: ] - B_temp[1: -1, : -2])
                                                      + D_nu / rho_temp[1: -1, 1: -1] * (1 / dr**2 * (Vz_temp[2:, 1: -1] - 2 * Vz_temp[1: -1, 1: -1] + Vz_temp[:-2, 1: -1])
                                                      + 1 / (rr[1: -1, 1: -1] * 2 * dr) * (Vz_temp[2:, 1: -1] - Vz_temp[:-2, 1: -1])
                                                      + 1 / dz**2 * (Vz_temp[1: -1, 2:] - 2 * Vz_temp[1: -1, 1: -1] + Vz_temp[1: -1, :-2])))
                                         
            p[1: -1, 1: -1] = (p_temp[1: -1, 1: -1] - dt / (2 * dr) * Vr_temp[1: -1, 1: -1] * (p_temp[2: , 1: -1] - p_temp[: -2, 1: -1]) 
                                                    - dt / (2 * dz) * Vz_temp[1: -1, 1: -1] * (p_temp[1: -1, 2: ] - p_temp[1: -1, : -2])
                               - rosh * dt / (2 * dr) * p_temp[1: -1, 1: -1] / rr[1: -1, 1: -1] * (rr[2: , 1: -1] * Vr_temp[2: , 1: -1] - rr[: -2, 1: -1] * Vr_temp[: -2, 1: -1])
                               - rosh * dt / (2 * dz) * p_temp[1: -1, 1: -1] * (Vz_temp[1: -1, 2: ] - Vz_temp[1: -1, : -2])
                               + (rosh - 1) * dt * (4 * np.pi * D_eta / Const.c**2) * ((1 / (2 * dz) * (B_temp[1: -1, 2:] - B_temp[1: -1, :-2]))**2
                               + 1 / rr[1: -1, 1: -1]**2 * (1 / (2 * dr) * (rr[2:, 1: -1] * B_temp[2:, 1: -1] - rr[:-2, 1: -1] * B_temp[:-2, 1: -1]))**2))
            T[1: -1, 1: -1] = p[1: -1, 1: -1] / (2 * rho[1: -1, 1: -1])
            
            def bc_periodicZ(u):
                u_temp = u
                u_temp[:, 0] = u_temp[:, -2]
                u_temp[:, -1] = u_temp[:, 1]
                return u_temp
            
            # Boundary conditions
            rho = bc_periodicZ(rho)
            B   = bc_periodicZ(B)
            Vr  = bc_periodicZ(Vr)
            Vz  = bc_periodicZ(Vz)
            p   = bc_periodicZ(p)
            T   = bc_periodicZ(T)
            rho[0, :] = rho[1, :]
            rho[-1, :] = rho[-2, :]
            B[0, :] = -B[1, :]
            B[-1, :] = rr[-2, :] * B[-2, :] / rr[-1, :]
            Vr[0, :] = -Vr[1, :]
            Vr[-1, :] = -Vr[-2, :]
            Vz[0, :] = Vz[1, :]
            Vz[-1, :] = -Vz[-2, :]
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            T[0, :] = T[1, :]
            T[-1, :] = T[-2, :]
            
        # Normalization
        self.rho = rho #* 1e-15
        self.p = p #* 1e-9
        self.T = T #* 1e6
        self.B = B #* 1e-5
        self.Vr = Vr #* 1e3
        self.Vz = Vz #* 1e3
        
    def evolve_WENO(self, courant=0.8, t_max=0):
        B = self.B
        Vr = self.Vr
        Vz = self.Vz
        rho = self.rho
        p = self.p
        T = self.T
        
        rr = self.sys.grid_r.rr
        
        t = 0
        cycle = 0
        
        print('Simulation seeded.')
        print('Cycle | t | dt')
        t0 = time.time()
        while t < t_max:
            # Primitive variables
            B_temp   = B.copy()
            Vr_temp  = Vr.copy()
            Vz_temp  = Vz.copy()
            rho_temp = rho.copy()

            # Post-processed variables
            p_temp = p.copy()
            T_temp = T.copy()

            # Conserved variables
            Ur_temp = rho_temp * Vr_temp
            Uz_temp = rho_temp * Vz_temp
            
            # Takes into account boundary conditions set on p and T
            e_temp = self.compute_e(B_temp, Vr_temp, Vz_temp, rho_temp, p_temp)

            # Time step from CFL condition
            self.dt = self.CFL(courant, B_temp[3:-3,3:-3], Vr_temp[3:-3,3:-3], Vz_temp[3:-3,3:-3], rho_temp[3:-3,3:-3], T_temp[3:-3,3:-3])
            if self.dt < 1e-4:
                print('Solution has not converged. Break')
                break
                
            t += self.dt
            cycle += 1
            print(cycle, t, self.dt)

            # Updates fluxes and source terms of conserved variables
            fluxR_B   = Vr_temp * B_temp
            fluxZ_B   = Vz_temp * B_temp
            fluxR_Ur  = rho_temp * Vr_temp**2 + p_temp + B_temp**2 / (8 * np.pi)
            fluxZ_Ur  = rho_temp * Vz_temp * Vr_temp
            fluxR_Uz  = rho_temp * Vr_temp * Vz_temp
            fluxZ_Uz  = rho_temp * Vz_temp**2 + p_temp + B_temp**2 / (8 * np.pi)
            fluxR_rho = rho_temp * Vr_temp
            fluxZ_rho = rho_temp * Vz_temp
            fluxR_e   = Vr_temp * (self.rosh / (self.rosh - 1) * p_temp + 1/2 * rho_temp * (Vr_temp**2 + Vz_temp**2) + B_temp**2 / (4 * np.pi))
            fluxZ_e   = Vz_temp * (self.rosh / (self.rosh - 1) * p_temp + 1/2 * rho_temp * (Vr_temp**2 + Vz_temp**2) + B_temp**2 / (4 * np.pi))

            source_B   = 0
            source_Ur  = -1 / np.abs(rr) * (p_temp * Vr_temp**2 + B_temp**2 / (4 * np.pi))
            source_Uz  = -1 / np.abs(rr) * rho_temp * Vr_temp * Vz_temp
            source_rho = -1 / np.abs(rr) * rho_temp * Vr_temp
            source_e   = -1 / np.abs(rr) * Vr_temp * (self.rosh / (self.rosh - 1) * p_temp + 1/2 * rho_temp * (Vr_temp**2 + Vz_temp**2) + B_temp**2 / (4 * np.pi))

            # Updates largest eigenvalues of the Jacobians of the fluxes
            # eval = V + sqrt(v_s^2 + v_A^2)
            self.evalR = (np.amax(np.abs(Vr_temp[3:-3,3:-3]) 
                          + np.sqrt(self.rosh * p_temp[3:-3,3:-3] / rho_temp[3:-3,3:-3] + np.abs(B_temp[3:-3,3:-3])**2 / (4 * np.pi * rho_temp[3:-3,3:-3]))))
            self.evalZ = (np.amax(np.abs(Vz_temp[3:-3,3:-3]) 
                          + np.sqrt(self.rosh * p_temp[3:-3,3:-3] / rho_temp[3:-3,3:-3] + np.abs(B_temp[3:-3,3:-3])**2 / (4 * np.pi * rho_temp[3:-3,3:-3]))))

            # Updates conserved variables.
            B   = PDESolverCyl(B_temp,   fluxR_B,   fluxZ_B,   source_B,   self).time_step()
            Ur  = PDESolverCyl(Ur_temp,  fluxR_Ur,  fluxZ_Ur,  source_Ur,  self).time_step()
            Uz  = PDESolverCyl(Uz_temp,  fluxR_Uz,  fluxZ_Uz,  source_Uz,  self).time_step()
            rho = PDESolverCyl(rho_temp, fluxR_rho, fluxZ_rho, source_rho, self).time_step()
            e   = PDESolverCyl(e_temp,   fluxR_e,   fluxZ_e,   source_e,   self).time_step()
            
            # Imposes boundary conditions on conserved variables except energy
            B   = self.boundary_conditions(B,   'dirichlet', '1/r')
            Ur  = self.boundary_conditions(Ur,  'dirichlet', 'dirichlet') 
            Uz  = self.boundary_conditions(Uz,  'neumann',   'neumann') 
            rho = self.boundary_conditions(rho, 'neumann',   'neumann') 

            # Updates primitive and post-processed variables
            Vr = Ur / rho
            Vz = Uz / rho
            p  = self.compute_p(B, Vr, Vz, rho, e)
            T  = self.compute_T(rho, p)
            
            # Imposes boundary conditions on p and T
            # Vr and Vz automatically taken care of
            p = self.boundary_conditions(p, 'neumann', 'neumann') 
            T = self.boundary_conditions(T, 'neumann', 'neumann') 
        
        t1 = time.time()
        print('Simulation finished.')
        print('Time elapsed:', t1-t0)

#         self.B = 1e-3 * B
#         self.Vr = 1e2 * Vr
#         self.Vz = 1e2 * Vz
#         self.rho = 1e-5 * rho
#         self.p = 1e-5 * p
#         self.T = 1e0 * T
        self.B = B
        self.Vr = Vr
        self.Vz = Vz
        self.rho = rho
        self.p = p
        self.T = T

    # Pre-processed energy obtained from initial pressure
    def compute_e(self, B, Vr, Vz, rho, p):
        energy = p / (self.rosh - 1) + 1/2 * rho * (Vr**2 + Vz**2) + B**2 / (8 * np.pi)
        return energy

    # Post-processed pressure obtained from time-updated energy
    def compute_p(self, B, Vr, Vz, rho, e):
        pressure = (self.rosh - 1) * (e - 1/2 * rho * (Vr**2 + Vz**2) - B**2 / (8 * np.pi))
        return pressure

    # Equation of state.
    # Post-processed temperature obtained from time-updated pressure
    def compute_T(self, rho, p):
        temperature = Const.m_i * p / (2 * rho)
        return temperature
        
    def boundary_conditions(self, u, r0, rmax):
        u_temp = u
        u_temp[:, 0] = u_temp[:, -6]
        u_temp[:, 1] = u_temp[:, -5]
        u_temp[:, 2] = u_temp[:, -4]
        u_temp[:, -3] = u_temp[:, 3]
        u_temp[:, -2] = u_temp[:, 4]
        u_temp[:, -1] = u_temp[:, 5]
        
        if r0 == 'dirichlet':
            u_temp[0, :] = 0
            u_temp[1, :] = 0
            u_temp[2, :] = 0
        elif r0 == 'neumann':
            u_temp[0, :] = u_temp[3, :]
            u_temp[1, :] = u_temp[3, :]
            u_temp[2, :] = u_temp[3, :]
        elif r0 == 'periodic':
            u_temp[0, :] = u_temp[-6, :]
            u_temp[1, :] = u_temp[-5, :]
            u_temp[2, :] = u_temp[-4, :]
        
        if rmax == 'dirichlet':
            u_temp[-1, :] = 0
            u_temp[-2, :] = 0
            u_temp[-3, :] = 0
        elif rmax == 'neumann':
            u_temp[-1, :] = u_temp[-4, :]
            u_temp[-2, :] = u_temp[-4, :]
            u_temp[-3, :] = u_temp[-4, :]
        elif rmax == '1/r':
            u_temp[-1, :] = u_temp[-4, :] * self.sys.grid_r.rr[-4, :] / self.sys.grid_r.rr[-1, :]
            u_temp[-2, :] = u_temp[-4, :] * self.sys.grid_r.rr[-4, :] / self.sys.grid_r.rr[-2, :]
            u_temp[-3, :] = u_temp[-4, :] * self.sys.grid_r.rr[-4, :] / self.sys.grid_r.rr[-3, :]
        elif rmax == 'periodic':
            u_temp[-1, :] = u_temp[5, :]
            u_temp[-2, :] = u_temp[4, :]
            u_temp[-3, :] = u_temp[3, :]
        
        return u_temp


class PDESolverCyl:
    def __init__(self, u, fluxR, fluxZ, source, evo):
        self.u = u
        self.fluxR = fluxR
        self.fluxZ = fluxZ
        self.source = source
        self.evo = evo

    # 3rd order Runge-Kutta time discretization
    def time_step(self):
        nr = self.evo.sys.grid_r.nr
        nz = self.evo.sys.grid_z.nr
        rr = self.evo.sys.grid_r.rr
        dt = self.evo.dt
        
        u1 = np.zeros((nr, nz))
        u2 = np.zeros((nr, nz))
        u3 = np.zeros((nr, nz))
        
        # aaa is the modified np.apply_along_axis, see aaa_source.py
        u1 = self.u + dt * (aaa(self.derivativeR, 0, self.u, self.fluxR) + aaa(self.derivativeZ, 1, self.u, self.fluxZ) + self.source)
        u2 = 3/4 * self.u + 1/4 * u1 + 1/4 * dt * (aaa(self.derivativeR, 0, u1, self.fluxR) + aaa(self.derivativeZ, 1, u1, self.fluxZ) + self.source)
        u3 = 1/3 * self.u + 2/3 * u2 + 2/3 * dt * (aaa(self.derivativeR, 0, u2, self.fluxR) + aaa(self.derivativeZ, 1, u2, self.fluxZ) + self.source)

        return u3
        
    # 5th order WENO FD scheme, component-wise
    def derivativeR(self, u, flux):
        self.a = self.evo.evalR
        self.flux = flux
        nr = self.evo.sys.grid_r.nr
        dr = self.evo.sys.grid_r.dr

        dv = np.zeros(nr)
        dv[3: -3] = -1 / dr * (self.fhat(u, 'plus', 0) + self.fhat(u, 'minus', 0) - self.fhat(u, 'plus', -1) - self.fhat(u, 'minus', -1))
        return dv
        
    def derivativeZ(self, u, flux):
        self.a = self.evo.evalZ
        self.flux = flux
        nz = self.evo.sys.grid_z.nr
        dz = self.evo.sys.grid_z.dr
        
        dv = np.zeros(nz)
        dv[3: -3] = -1 / dz * (self.fhat(u, 'plus', 0) + self.fhat(u, 'minus', 0) - self.fhat(u, 'plus', -1) - self.fhat(u, 'minus', -1))
        return dv
    
    # g=0 -> j+1/2. g=-1 -> j-1/2.
    def fhat(self, u, pm, g):
        fplus = self.fp(u)
        fminus = self.fm(u)
    
        if pm == 'plus':
            fhat0 =  1/3 * fplus[1+g: -5+g] - 7/6 * fplus[2+g: -4+g] + 11/6 * fplus[3+g: -3+g]
            fhat1 = -1/6 * fplus[2+g: -4+g] + 5/6 * fplus[3+g: -3+g] + 1/3  * fplus[4+g: -2+g]
            fhat2 =  1/3 * fplus[3+g: -3+g] + 5/6 * fplus[4+g: -2+g] - 1/6  * fplus[5+g: -1+g]
        
            w0, w1, w2 = self.SI_weights(u, 'plus', g)
        
        elif pm == 'minus':
            fhat0 =  1/3 * fminus[4+g: -2+g] + 5/6 * fminus[3+g: -3+g] - 1/6 * fminus[2+g: -4+g]
            fhat1 = -1/6 * fminus[5+g: -1+g] + 5/6 * fminus[4+g: -2+g] + 1/3 * fminus[3+g: -3+g]
            
            if g == 0:
                fhat2 = 1/3 * fminus[6+g:  ] - 7/6 * fminus[5+g: -1+g] + 11/6 * fminus[4+g: -2+g]
            elif g == -1:
                fhat2 = 1/3 * fminus[6+g: g] - 7/6 * fminus[5+g: -1+g] + 11/6 * fminus[4+g: -2+g]
            
            w0, w1, w2 = self.SI_weights(u, 'minus', g)

        return w0 * fhat0 + w1 * fhat1 + w2 * fhat2
        
    def fp(self, u):
        return 1/2 * (self.flux + self.a * u)

    def fm(self, u):
        return 1/2 * (self.flux - self.a * u)

    def SI_weights(self, u, pm, g):
        fplus = self.fp(u)
        fminus = self.fm(u)
        
        # Small parameter to avoid division by 0 in weights
        epsilon = 1e-6

        # pm: positive/negative flux.
        if pm == 'plus':
            IS0 = 13/12 * (fplus[1+g: -5+g] - 2 * fplus[2+g: -4+g] + fplus[3+g: -3+g])**2 + 1/4 * (fplus[1+g: -5+g] - 4 * fplus[2+g: -4+g] + 3 * fplus[3+g: -3+g])**2
            IS1 = 13/12 * (fplus[2+g: -4+g] - 2 * fplus[3+g: -3+g] + fplus[4+g: -2+g])**2 + 1/4 * (fplus[2+g: -4+g] - fplus[4+g: -2+g])**2
            IS2 = 13/12 * (fplus[3+g: -3+g] - 2 * fplus[4+g: -2+g] + fplus[5+g: -1+g])**2 + 1/4 * (3 * fplus[3+g: -3+g] - 4 * fplus[4+g: -2+g] + fplus[5+g: -1+g])**2
            
            # Yamaleev and Carpenter (2009)
            tau = (fplus[1+g: -5+g] - 4 * fplus[2+g: -4+g] + 6 * fplus[3+g: -3+g] - 4 * fplus[4+g: -2+g] + fplus[5+g: -1+g])**2

            alpha0 = 1/10 * (1 + (tau / (epsilon + IS0))**3)
            alpha1 = 6/10 * (1 + (tau / (epsilon + IS1))**3)
            alpha2 = 3/10 * (1 + (tau / (epsilon + IS2))**3)
        
        elif pm == 'minus':
            IS0 = 13/12 * (fminus[2+g: -4+g] - 2 * fminus[3+g: -3+g] + fminus[4+g: -2+g])**2 + 1/4 * (fminus[2+g: -4+g] - 4 * fminus[3+g: -3+g] + 3 * fminus[4+g: -2+g])**2
            IS1 = 13/12 * (fminus[3+g: -3+g] - 2 * fminus[4+g: -2+g] + fminus[5+g: -1+g])**2 + 1/4 * (fminus[3+g: -3+g] - fminus[5+g: -1+g])**2

            if g == 0:
                IS2 = 13/12 * (fminus[4+g: -2+g] - 2 * fminus[5+g: -1+g] + fminus[6+g:  ])**2 + 1/4 * (3 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g:  ])**2
                tau = (fminus[2+g: -4+g] - 4 * fminus[3+g: -3+g] + 6 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g:  ])**2
            elif g == -1:
                IS2 = 13/12 * (fminus[4+g: -2+g] - 2 * fminus[5+g: -1+g] + fminus[6+g: g])**2 + 1/4 * (3 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g: g])**2
                tau = (fminus[2+g: -4+g] - 4 * fminus[3+g: -3+g] + 6 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g: g])**2

            alpha0 = 3/10 * (1 + (tau / (epsilon + IS0))**3)
            alpha1 = 6/10 * (1 + (tau / (epsilon + IS1))**3)
            alpha2 = 1/10 * (1 + (tau / (epsilon + IS2))**3)
        
        w0 = alpha0 / (alpha0 + alpha1 + alpha2)
        w1 = alpha1 / (alpha0 + alpha1 + alpha2)
        w2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
        return w0, w1, w2
        
        

        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
           
