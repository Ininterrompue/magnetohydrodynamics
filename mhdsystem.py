import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix


# centimeters-eV-seconds
class ConstCGS:
    c   = 1e10
    m_i = 1.04e-12
    I   = 1e6 * 3e9 * 7.89e5
    T_0 = 1e5 * 8.62e-5
    r_0 = 1


# centimeters-MeV-nanoseconds
class ConstSpecial:
    c   = 30
    m_i = 1.04

    # I = 1 MA
    I   = 1e6 * 2.37e3
    
    # T_0 = 100000 K
    T_0 = 1e5 * 8.62e-11
    
    # r_0 = 1 cm
    r_0 = 1
    
    P_0 = I**2 / (np.pi * r_0**2 * c**2)


class ConstNorm:
    c   = 1
    m_i = 1
    I   = np.sqrt(np.pi)
    T_0 = 1
    r_0 = 1
    P_0 = I**2 / (np.pi * r_0**2 * c**2)

class ConstNorm2:
    # cgs, with B0 = 20 T normalized to 1, mass normalized to m_i, T = 11605 K
    c = 3e10
    m_i = 1
    I = 1e6 * 3e9
    T_0 = 1 # 11605 K
    r_0 = 1
    P_0 = 1 / (4 * np.pi)

# Select units
Const = ConstNorm2


class System:
    def __init__(self, n_ghost=1, resR=64, r_max=2*np.pi, resZ=64, z_max=2*np.pi, D_eta=0, D_H=0, D_P=0, B_Z0=0, V_Z0=0):
        self.grid_r = Grid(resR, n_ghost, r_max)
        self.grid_z = Grid(resZ, n_ghost, z_max)
        self.fd = FDSystem(self.grid_r)
        
        # Set plasma related parameters
        self.D_eta = D_eta
        self.D_H = D_H
        self.D_P = D_P
        self.B_Z0 = B_Z0
        self.V_Z0 = V_Z0
    
    
class Grid:
    def __init__(self, res=64, n_ghost=1, r_max=2*np.pi):
        nr = 2 * n_ghost + res
        dr = r_max / nr
        r = np.linspace(-dr/2 * (2*n_ghost - 1), r_max + dr/2 * (2*n_ghost - 1), nr)
        rr = np.reshape(r, (nr, 1))

        self.nr = nr
        self.dr = dr
        self.r = r
        self.rr = rr
        self.res = res
        self.n_ghost = n_ghost
        self.r_max = r_max
        
       
# Only applicable for n_ghost = 1
class FDSystem:
    def __init__(self, grid):
        self.grid = grid
        self.bc_rows = {'value': [1, 1], 'derivative': [1, -1]}

    def ddr(self, order):
        nr = self.grid.nr
        dr = self.grid.dr

        one = np.ones(nr - 1)
        if order == 1:
            dv = (np.diag(one, 1) - np.diag(one, -1)) / (2 * dr)
        elif order == 2:
            dv = (dia_matrix((one, 1), shape=(nr, nr)).toarray() - 2 * np.identity(nr) + dia_matrix((one, -1), shape=(nr, nr)).toarray()) / dr**2  
      
        dv = self.zero_bc(dv)
        return dv

    def ddr_product(self, vec):
        dv = (np.diagflat(vec[1: ], 1) - np.diagflat(vec[: -1], -1)) / (2 * self.grid.dr)
        dv = self.zero_bc(dv)
        return dv

    def diag(self, vec):
        M = np.diagflat(vec, 0)
        M = self.zero_bc(M)
        return M
    
    def diag_I(self):
        return np.identity(self.grid.nr)
        
    def zeros(self):
        return np.zeros((self.grid.nr, self.grid.nr))

    def zero_bc(self, M_0):
        M = M_0.copy()
        M[0,  :] = 0
        M[-1, :] = 0  
        return M

    def lhs_bc(self, bc_type='value'):
        entries = self.bc_rows[bc_type]
        row = np.zeros(self.grid.nr)
        row[0] = entries[0]
        row[1] = entries[1]
        M = self.zeros()
        M[0] = row
        return M

    def rhs_bc(self, bc_type='value'):
        entries = self.bc_rows[bc_type]
        row = np.zeros(self.grid.nr)
        row[-2] = entries[0]
        row[-1] = entries[1]
        M = self.zeros()
        M[-1] = row
        return M

        
class Plot:
    def __init__(self, sys, equ, lin, evo):
        self.sys = sys
        self.equ = equ
        self.lin = lin
        self.evo = evo
    
    def plot_equilibrium(self):
        r = self.sys.grid_r.r
        gh = self.sys.grid_r.n_ghost
        
        plt.plot(r[gh: -gh], self.equ.p[gh: -gh], 
                 r[gh: -gh], self.equ.b[gh: -gh], 
                 r[gh: -gh], self.equ.j[gh: -gh])
        plt.title('Initial profiles')
        plt.xlabel('r')
        plt.legend(['P', 'B', 'J'])
        plt.show() 
        
    def plot_eigenvalues(self):
        if self.lin.evals is None:
            return

        plt.scatter(self.lin.evals.real, self.lin.evals.imag, s=1)
        plt.title('Omega')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.show() 
        
        
    def plot_VB(self, i=-1, epsilon=0.05):
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.lin.extract_from_evec(i, epsilon)
        rho_0 = self.equ.rho
        p_0   = self.equ.p
        B_0   = self.equ.b
        VZ_0  = self.equ.v
        B_Z0  = self.equ.sys.B_Z0
        r  = self.sys.grid_r.r
        nr = self.sys.grid_r.nr
        z  = self.sys.grid_z.r
        zz = self.sys.grid_z.rr
        k  = self.lin.k

        # def f1(x): return np.abs(x)
        # def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)
        
        f = plt.figure()
        f.suptitle('Fastest growing mode')

#         ax = plt.subplot(3,3,1)
#         ax.set_title('B_r')
#         ax.plot(r[1: -1], f1(Br[1: -1]),
#                 r[1: -1], f2(Br[1: -1]))

        ax = plt.subplot(2,3,1)
        ax.set_title('B_theta')
        ax.plot(r[1: -1], f1(Btheta[1: -1]),
                r[1: -1], f2(Btheta[1: -1]))

#         ax = plt.subplot(3,3,3)
#         ax.set_title('B_z')
#         ax.plot(r[1: -1], f1(Bz[1: -1]),
#                 r[1: -1], f2(Bz[1: -1]))

        ax = plt.subplot(2,3,2)
        ax.set_title('V_r')
        ax.plot(r[1: -1], f1(Vr[1: -1]),
                r[1: -1], f2(Vr[1: -1]))

#         ax = plt.subplot(3,3,5)
#         ax.set_title('V_theta')
#         ax.plot(r[1: -1], f1(Vtheta[1: -1]),
#                 r[1: -1], f2(Vtheta[1: -1]))

        ax = plt.subplot(2,3,3)
        ax.set_title('V_z')
        ax.plot(r[1: -1], f1(Vz[1: -1]),
                r[1: -1], f2(Vz[1: -1]))

        ax = plt.subplot(2,3,4)
        ax.set_title('rho')
        ax.plot(r[1: -1], f1(rho[1: -1]),
                r[1: -1], f2(rho[1: -1]))

        ax = plt.subplot(2,3,5)
        ax.set_title('P')
        ax.plot(r[1: -1], f1(p[1: -1]),
                r[1: -1], f2(p[1: -1]))

        ax = plt.subplot(2,3,6)
        ax.set_title('T')
        ax.plot(r[1: -1], f1(temp[1: -1]) - Const.T_0,
                r[1: -1], f2(temp[1: -1]))

        plt.show()
        
        # 2D contours
        rho    = np.reshape(rho,    (nr, ))
        Br     = np.reshape(Br,     (nr, ))
        Btheta = np.reshape(Btheta, (nr, ))
        Bz     = np.reshape(Bz,     (nr, ))
        Vr     = np.reshape(Vr,     (nr, ))
        Vtheta = np.reshape(Vtheta, (nr, ))
        Vz     = np.reshape(Vz,     (nr, ))
        p      = np.reshape(p,      (nr, ))
        temp_1 = np.reshape(temp_1, (nr, ))

        z_osc = np.exp(1j * k * zz)
        rho_contour    = (rho_0.T + f1(z_osc * rho))
        Br_contour     = f1(z_osc * Br)
        Btheta_contour = B_0.T + f1(z_osc * Btheta)
        Bz_contour     = (B_Z0 * np.ones(nr)).T + f1(z_osc * Bz)
        Vr_contour     = f1(z_osc * Vr)
        Vtheta_contour = f1(z_osc * Vtheta)
        Vz_contour     = VZ_0.T + f1(z_osc * Vz)
        p_contour      = p_0.T + f1(z_osc * p)
        temp_contour   = Const.T_0 + f1(z_osc * temp_1)
        
        f = plt.figure()
        f.suptitle('Fastest growing mode')
        R, Z = np.meshgrid(r[1: nr//2], z[1: -1])

#         ax = plt.subplot(3,3,1)
#         ax.set_title('B_r')
#         plot_1 = ax.contourf(R, Z, Br_contour, 20)
#         plt.colorbar(plot_1)
    
        ax = plt.subplot(2,3,1)
        ax.set_title('B_theta')
        plot_2 = ax.contourf(R, Z, Btheta_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_2)

#         ax = plt.subplot(3,3,3)
#         ax.set_title('B_z')
#         plot_3 = ax.contourf(R, Z, Bz_contour, 20)
#         plt.colorbar(plot_3)

        ax = plt.subplot(2,3,2)
        ax.set_title('V_r')
        plot_4 = ax.contourf(R, Z, Vr_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_4)

#         ax = plt.subplot(3,3,5)
#         ax.set_title('V_theta')
#         plot_5 = ax.contourf(R, Z, Vtheta_contour, 20)
#         plt.colorbar(plot_5)

        ax = plt.subplot(2,3,3)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, Vz_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_6)

        ax = plt.subplot(2,3,4)
        ax.set_title('rho')
        plot_7 = ax.contourf(R, Z, rho_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_7)

        ax = plt.subplot(2,3,5)
        ax.set_title('P')
        plot_8 = ax.contourf(R, Z, p_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_8)

        ax = plt.subplot(2,3,6)
        ax.set_title('T')
        plot_9 = ax.contourf(R, Z, temp_contour[1: -1, 1: nr//2], 20)
        plt.colorbar(plot_9)

        plt.show()
        
        # Vorticity
        vort_theta_contour = self.lin.compute_vort()
        
        plot = plt.contourf(R, Z, 1e3 * vort_theta_contour[1: -1, 1: nr//2], 200, cmap='coolwarm')
        plt.colorbar(plot)
        
        d_vec = 8
        plt.title('Flow velocity and vorticity')
        plt.xlabel('r')
        plt.ylabel('z')
        Vr_view = Vr_contour[1: -1, 1: nr//2]
        Vz_view = Vz_contour[1: -1, 1: nr//2]
        quiv = plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec],
                          Vr_view[::d_vec, ::d_vec], Vz_view[::d_vec, ::d_vec],
                          pivot='mid', width=0.003, scale=0.2)
        plt.show()
        
        # Divergence of V
        divV_contour = self.lin.compute_divV()

        plt.title('Streamlines and div V')
        plt.xlabel('r')
        plt.ylabel('z')
        plot = plt.contourf(R, Z, 1e3 * divV_contour[1: -1, 1: nr//2], 200, cmap='coolwarm')
        plt.colorbar(plot)
        quiv = plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec],
                   Vr_view[::d_vec, ::d_vec], Vz_view[::d_vec, ::d_vec],
                   pivot='mid', width=0.003, scale=0.2)
        strm = plt.streamplot(r[1: nr//2], z[1: -1], Vr_view, Vz_view,
                   linewidth=0.5, arrowsize=0.001, density=2, color='black')
        plt.show()

        
    def plot_EJ(self, i=-1, epsilon=0.05):
        if self.lin.evecs is None:
            return
        
        rho, Br, Btheta, Bz, Vr, Vtheta, Vz, p, temp, temp_1 = self.lin.extract_from_evec(i, epsilon)
        nr    = self.sys.grid_r.nr
        r     = self.sys.grid_r.r
        z     = self.sys.grid_z.r
        zz    = self.sys.grid_z.rr
        J_0   = self.equ.j
        k     = self.lin.k

        # def f1(x): return np.abs(x)
        # def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
        def f1(x): return np.real(x)
        def f2(x): return np.imag(x)
        
        Jr, Jtheta, Jz1 = self.lin.compute_currents()
        Jz = J_0 + Jz1
        Er_ideal, Etheta_ideal, Ez_ideal = self.lin.compute_E_ideal()
        Er_resistive, Etheta_resistive, Ez0_resistive, Ez1_resistive = self.lin.compute_E_resistive()
        Er0_hall, Er1_hall, Etheta_hall, Ez_hall = self.lin.compute_E_hall()
        Er0_pressure, Er1_pressure, Etheta_pressure, Ez_pressure = self.lin.compute_E_pressure()
        
        # 1D perturbations of J and E
        ax = plt.subplot(1,2,1)
        ax.set_title('J_r')
        ax.plot(r[1: -1], f1(Jr[1: -1]),
                r[1: -1], f2(Jr[1: -1]))  
              
#         ax = plt.subplot(1,3,2)
#         ax.set_title('J_theta')
#         ax.plot(r[1: -1], f1(Jtheta[1: -1]),
#                 r[1: -1], f2(Jtheta[1: -1]))
            
        ax = plt.subplot(1,2,2)
        ax.set_title('J_z')
        ax.plot(r[1: -1], f1(Jz1[1: -1]),
                r[1: -1], f2(Jz1[1: -1]))
                
        plt.show()    
                    
        ax = plt.subplot(2,4,1)
        ax.set_title('E_r_ideal')
        ax.plot(r[1: -1], f1(Er_ideal[1: -1]),
                r[1: -1], epsilon * f2(Er_ideal[1: -1]))
            
#         ax = plt.subplot(4,3,2)
#         ax.set_title('E_theta_ideal')
#         ax.plot(r[1: -1], f1(Etheta_ideal[1: -1]),
#                 r[1: -1], f2(Etheta_ideal[1: -1]))
           
        ax = plt.subplot(2,4,5)
        ax.set_title('E_z_ideal')	
        ax.plot(r[1: -1], f1(Ez_ideal[1: -1]),
                r[1: -1], f2(Ez_ideal[1: -1]))
            
        ax = plt.subplot(2,4,2)
        ax.set_title('E_r_resistive')
        ax.plot(r[1: -1], f1(Er_resistive[1: -1]),
                r[1: -1], f2(Er_resistive[1: -1]))
            
#         ax = plt.subplot(4,3,5)
#         ax.set_title('E_theta_resistive')
#         ax.plot(r[1: -1], f1(Etheta_resistive[1: -1]),
#                 r[1: -1], f2(Etheta_resistive[1: -1]))
           
        ax = plt.subplot(2,4,6)
        ax.set_title('E_z_resistive')	
        ax.plot(r[1: -1], f1(Ez1_resistive[1: -1]),
                r[1: -1], f2(Ez1_resistive[1: -1]))
                
        ax = plt.subplot(2,4,3)
        ax.set_title('E_r_hall')
        ax.plot(r[1: -1], f1(Er1_hall[1: -1]),
                r[1: -1], f2(Er1_hall[1: -1]))
            
#         ax = plt.subplot(4,3,8)
#         ax.set_title('E_theta_hall')
#         ax.plot(r[1: -1], f1(Etheta_hall[1: -1]),
#                 r[1: -1], f2(Etheta_hall[1: -1]))
           
        ax = plt.subplot(2,4,7)
        ax.set_title('E_z_hall')	
        ax.plot(r[1: -1], f1(Ez_hall[1: -1]),
                r[1: -1], f2(Ez_hall[1: -1]))
                
        ax = plt.subplot(2,4,4)
        ax.set_title('E_r_pressure')
        ax.plot(r[1: -1], f1(Er1_pressure[1: -1]),
                r[1: -1], f2(Er1_pressure[1: -1]))
            
#         ax = plt.subplot(4,3,11)
#         ax.set_title('E_theta_pressure')
#         ax.plot(r[1: -1], f1(Etheta_pressure[1: -1]),
#                 r[1: -1], f2(Etheta_pressure[1: -1]))
           
        ax = plt.subplot(2,4,8)
        ax.set_title('E_z_pressure')	
        ax.plot(r[1: -1], f1(Ez_pressure[1: -1]),
                r[1: -1], f2(Ez_pressure[1: -1]))

        plt.show()
        
        # 2D contour plots of J and E
        z_osc = np.exp(1j * k * zz)
        R, Z = np.meshgrid(r[1: -1], z[1: -1])
        
        Jr     = np.reshape(Jr, (nr, ))
        Jtheta = np.reshape(Jtheta, (nr, ))
        Jz     = np.reshape(Jz, (nr, ))
        Er_ideal         = np.reshape(Er_ideal, (nr, ))
        Etheta_ideal     = np.reshape(Etheta_ideal, (nr, ))
        Ez_ideal         = np.reshape(Ez_ideal, (nr, ))
        Er_resistive     = np.reshape(Er_resistive, (nr, ))
        Etheta_resistive = np.reshape(Etheta_resistive, (nr, ))
        Ez0_resistive    = np.reshape(Ez0_resistive, (nr, ))
        Ez1_resistive    = np.reshape(Ez1_resistive, (nr, ))
        Er0_hall         = np.reshape(Er0_hall, (nr, ))
        Er1_hall         = np.reshape(Er1_hall, (nr, ))
        Etheta_hall      = np.reshape(Etheta_hall, (nr, ))
        Ez_hall          = np.reshape(Ez_hall, (nr, ))
        Er0_pressure     = np.reshape(Er0_pressure, (nr, ))
        Er1_pressure     = np.reshape(Er1_pressure, (nr, ))
        Ez_pressure      = np.reshape(Ez_pressure, (nr, ))
        
        Jr_contour     = f1(z_osc[1: -1] * Jr[1: -1])
        Jtheta_contour = f1(z_osc[1: -1] * Jtheta[1: -1])
        Jz_contour     = J_0[1: -1].T + f1(z_osc[1: -1] * Jz1[1: -1])
        Er_ideal_contour         = f1(z_osc[1: -1] * Er_ideal[1: -1])
        Etheta_ideal_contour     = f1(z_osc[1: -1] * Etheta_ideal[1: -1])
        Ez_ideal_contour         = f1(z_osc[1: -1] * Ez_ideal[1: -1])
        Er_resistive_contour     = f1(z_osc[1: -1] * Er_resistive[1: -1])
        Etheta_resistive_contour = f1(z_osc[1: -1] * Etheta_resistive[1: -1])
        Ez_resistive_contour     = Ez0_resistive[1: -1].T + f1(z_osc[1: -1] * Ez1_resistive[1: -1])
        Er_hall_contour          = Er0_hall[1: -1].T + f1(z_osc[1: -1] * Er1_hall[1: -1])
        Etheta_hall_contour      = f1(z_osc[1: -1] * Etheta_hall[1: -1])
        Ez_hall_contour          = f1(z_osc[1: -1] * Ez_hall[1: -1])
        Er_pressure_contour      = Er0_pressure[1: -1].T + f1(z_osc[1: -1] * Er1_pressure[1: -1])
        Etheta_pressure_contour  = f1(z_osc[1: -1] * Etheta_pressure[1: -1])
        Ez_pressure_contour      = f1(z_osc[1: -1] * Ez_pressure[1: -1])
        
        ax = plt.subplot(1,2,1)
        ax.set_title('J_r')
        plot_1 = ax.contourf(R, Z, Jr_contour, 20)
        plt.colorbar(plot_1)
    
#         ax = plt.subplot(1,3,2)
#         ax.set_title('J_theta')
#         plot_2 = ax.contourf(R, Z, J_theta_contour, 20)
#         plt.colorbar(plot_2)
    
        ax = plt.subplot(1,2,2)
        ax.set_title('J_z')
        plot_3 = ax.contourf(R, Z, Jz_contour, 20)
        plt.colorbar(plot_3)
        
        plt.show()
    
        ax = plt.subplot(2,4,1)
        ax.set_title('E_r_ideal')
        plot_4 = ax.contourf(R, Z, Er_ideal_contour, 20)
        plt.colorbar(plot_4)
        
#         ax = plt.subplot(4,3,2)
#         ax.set_title('E_theta_ideal')
#         plot_5 = ax.contourf(R, Z, Etheta_ideal_contour, 20)
#         plt.colorbar(plot_5)
    
        ax = plt.subplot(2,4,5)
        ax.set_title('E_z_ideal')
        plot_6 = ax.contourf(R, Z, Ez_ideal_contour, 20)
        plt.colorbar(plot_6)
        
        ax = plt.subplot(2,4,2)
        ax.set_title('E_r_resistive')
        plot_7 = ax.contourf(R, Z, Er_resistive_contour, 20)
        plt.colorbar(plot_7)
        
#         ax = plt.subplot(4,3,5)
#         ax.set_title('E_theta_resistive')
#         plot_8 = ax.contourf(R, Z, Etheta_resistive_contour, 20)
#         plt.colorbar(plot_8)
    
        ax = plt.subplot(2,4,6)
        ax.set_title('E_z_resistive')
        plot_9 = ax.contourf(R, Z, Ez_resistive_contour, 20)
        plt.colorbar(plot_9)
        
        ax = plt.subplot(2,4,3)
        ax.set_title('E_r_hall')
        plot_10 = ax.contourf(R, Z, Er_hall_contour, 20)
        plt.colorbar(plot_10)
        
#         ax = plt.subplot(4,3,8)
#         ax.set_title('E_theta_hall')
#         plot_11 = ax.contourf(R, Z, Etheta_hall_contour, 20)
#         plt.colorbar(plot_11)
    
        ax = plt.subplot(2,4,7)
        ax.set_title('E_z_hall')
        plot_12 = ax.contourf(R, Z, Ez_hall_contour, 20)
        plt.colorbar(plot_12)
        
        ax = plt.subplot(2,4,4)
        ax.set_title('E_r_pressure')
        plot_13 = ax.contourf(R, Z, Er_pressure_contour, 20)
        plt.colorbar(plot_13)
        
#         ax = plt.subplot(4,3,11)
#         ax.set_title('E_theta_pressure')
#         plot_14 = ax.contourf(R, Z, Etheta_pressure_contour, 20)
#         plt.colorbar(plot_14)
    
        ax = plt.subplot(2,4,8)
        ax.set_title('E_z_pressure')
        plot_15 = ax.contourf(R, Z, Ez_pressure_contour, 20)
        plt.colorbar(plot_15)

        plt.show()
        
        # 2D quiver plots of J and E
        d_vec = 20
        
        ax = plt.subplot(2,2,1)
        ax.set_title('E_ideal')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Er_ideal_contour[::d_vec, ::d_vec], Ez_ideal_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=0.5)
    
        ax = plt.subplot(2,2,2)
        ax.set_title('E_resistive')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Er_resistive_contour[::d_vec, ::d_vec], Ez_resistive_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=10)
    
        ax = plt.subplot(2,2,3)
        ax.set_title('E_hall')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Er_hall_contour[::d_vec, ::d_vec], Ez_hall_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=40)
                   
        ax = plt.subplot(2,2,4)
        ax.set_title('E_pressure')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Er_pressure_contour[::d_vec, ::d_vec], Ez_pressure_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=40)
        plt.show()
        
        # Total electric field
        ax = plt.subplot(2, 2, 1)
        ax.set_title('E_r_total')
        plot_1 = ax.contourf(R, Z, Er_ideal_contour + Er_resistive_contour + Er_hall_contour + Er_pressure_contour, 20, cmap='plasma')
        plt.colorbar(plot_1)
        
#         ax = plt.subplot(2, 2, 2)
#         ax.set_title('E_theta_total')
#         plot_2 = ax.contourf(R, Z, Etheta_ideal_contour + Etheta_resistive_contour + Etheta_hall_contour + Etheta_pressure_contour, 20, cmap='plasma')
#         plt.colorbar(plot_2)
        
        ax = plt.subplot(2, 2, 2)
        ax.set_title('E_z_total')
        plot_3 = ax.contourf(R, Z, Ez_ideal_contour + Ez_resistive_contour + Ez_hall_contour + Ez_pressure_contour, 20, cmap='plasma')
        plt.colorbar(plot_3)
        
        ax = plt.subplot(2,2,3)
        ax.set_title('J')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Jr_contour[::d_vec, ::d_vec], Jz_contour[::d_vec, ::d_vec], 
                   pivot='mid', width=0.002, scale=20)
        
        ax = plt.subplot(2,2,4)
        ax.set_title('E_total')
        ax.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec],
                  Er_ideal_contour[::d_vec, ::d_vec] + Er_resistive_contour[::d_vec, ::d_vec] +
                  Er_hall_contour[::d_vec, ::d_vec] + Er_pressure_contour[::d_vec, ::d_vec],
                  Ez_ideal_contour[::d_vec, ::d_vec] + Ez_resistive_contour[::d_vec, ::d_vec] +
                  Ez_hall_contour[::d_vec, ::d_vec] + Ez_pressure_contour[::d_vec, ::d_vec],
                   pivot='mid', width=0.002, scale=30)
        plt.show()
        
    def plot_evolution(self):
        r   = self.sys.grid_r.r
        z   = self.sys.grid_z.r
        B   = self.evo.B
        rho = self.evo.rho
        Vr  = self.evo.Vr
        Vz  = self.evo.Vz
        p   = self.evo.p
        T   = self.evo.T
            
        plt.plot(r[1: -1], B[1: -1, 1], r[1: -1], rho[1: -1, 1], r[1: -1], Vr[1: -1, 1], r[1: -1], p[1: -1, 1], r[1: -1], T[1: -1, 1])
        plt.legend(['B', 'rho', 'V_r', 'p', 'T'])
        plt.xlabel('r')
        plt.title('Time evolution')
        plt.show()
        
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
        ax.set_title('V_r')
        plot_5 = ax.contourf(R, Z, Vr[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_5)
    
        ax = plt.subplot(2,3,6)
        ax.set_title('V_z')
        plot_6 = ax.contourf(R, Z, Vz[1: -1, 1: -1], 20, cmap='plasma')
        plt.colorbar(plot_6)     
        
        plt.show()    
        
        d_vec=5
        plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
                   Vr[::d_vec, ::d_vec].T, Vz[::d_vec, ::d_vec].T, 
                   pivot='mid', width=0.002, scale=100)
        plt.show() 
