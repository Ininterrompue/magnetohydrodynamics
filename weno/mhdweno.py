import numpy as np
from aaa_source import apply_along_axis as aaa
from scipy.special import gamma, gammainc
import matplotlib.pyplot as plt
import time

class Const:
    c   = 30
    m_i = 1.04e6
    I   = 3e6
    T_0 = 1e4 * 8.62e-5
    r_0 = 1
    
    # Normalized constants
#     c   = 1
#     m_i = 1
#     I   = np.sqrt(np.pi)
#     T_0 = 1
#     r_0 = 1


class MHDGrid:
    def __init__(self, res=64, n_ghost=3, r_max=2*np.pi):
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

        
class MHDEquilibrium:
    def __init__(self, grid, p_exp):
        self.grid = grid
        self.p_exp = p_exp
        self.p = self.compute_p()
        self.rho = self.compute_rho()
        self.B = self.compute_B()
        self.J = self.compute_J()

    def compute_p(self):
        # Initial pressure profile
        p_0 = Const.I**2 / (np.pi * Const.r_0**2 * Const.c**2)
        p = p_0 * (0.05 + np.exp(-self.grid.rr**self.p_exp))
        return p

    def compute_rho(self):
        # Equation of state. Initial condition of T=1 uniform
        rho = Const.m_i * self.p / (2 * Const.T_0)
        return rho

    def compute_B(self):
        # Magnetic pressure B^2/8pi
        p_0 = Const.I**2 / (np.pi * Const.r_0**2 * Const.c**2)
        b_pressure = p_0 * (2 / (self.p_exp * self.grid.rr**2) * gamma(2 / self.p_exp) * gammainc(2 / self.p_exp, self.grid.rr**self.p_exp)
                            - np.exp(-self.grid.rr**self.p_exp))
        b = np.sqrt(8 * np.pi) * np.sign(b_pressure) * np.sqrt(np.abs(b_pressure))
        return b

    def compute_J(self):
        num = np.exp(-self.grid.rr**self.p_exp) * np.sqrt(2 * np.pi) * self.p_exp * self.grid.rr**(self.p_exp - 1)
        insqrt = 2 / (self.p_exp * self.grid.rr**2) * gamma(2 / self.p_exp) * gammainc(2 / self.p_exp, self.grid.rr**self.p_exp) - np.exp(-self.grid.rr**self.p_exp)
        denom = np.sign(insqrt) * np.sqrt(np.abs(insqrt))
        j = Const.c / (4 * np.pi) * num / denom
        return j

    def plot_equilibrium(self):
        plt.plot(self.grid.r[3:-3], self.p[3:-3], self.grid.r[3:-3], self.B[3:-3], self.grid.r[3:-3], self.J[3:-3])
        plt.title('Initial profiles')
        plt.xlabel('r')
        plt.legend(['P', 'B', 'J'])
        plt.show()     


class MHDEvolution:
    def __init__(self, grid_r, grid_z, equ, k=1, rosh=2, D_eta=0, D_nu=0, courant=0.8, t_max=0):
        self.grid_r = grid_r
        self.grid_z = grid_z
        self.equ = equ
        self.k = k
        self.rosh = rosh
        self.D_eta = D_eta
        self.D_nu = D_nu
        self.courant = courant
        self.t_max = t_max
        self.B, self.Vr, self.Vz, self.rho, self.p, self.T = self.seed()

    # Seed simulation with perturbation in density and pressure
    # Convert variables to 2D: [r, z]
    def seed(self):
        nr = self.grid_r.nr
        nz = self.grid_z.nr
        zz = self.grid_z.rr

        pert = 1 - 0.01 * np.cos(self.k * zz)
        B = self.equ.B * np.ones(nz).T
        Vr = np.zeros((nr, nz))
        Vz = np.zeros((nr, nz))
        rho = self.equ.rho * pert.T
        p = self.equ.p * pert.T
        T = Const.T_0 * np.ones((nr, nz))
        
        print('Simulation seeded.')
        return B, Vr, Vz, rho, p, T 

    def plot_lineouts(self):      
        plt.plot(self.grid_r.r[3:-3], self.B[3:-3,3],
                 self.grid_r.r[3:-3], self.Vr[3:-3,3],
                 self.grid_r.r[3:-3], self.Vz[3:-3,3],
                 self.grid_r.r[3:-3], self.rho[3:-3,3],
                 self.grid_r.r[3:-3], self.p[3:-3,3],
                 self.grid_r.r[3:-3], self.T[3:-3,3])
        plt.title('Lineouts at z=0')
        plt.xlabel('r')
        plt.legend(['B', 'Vr', 'Vz', 'rho', 'p', 'T'])
        plt.show()

    def plot_evolved(self):
        R, Z = np.meshgrid(self.grid_r.r[3:-3], self.grid_z.r[3:-3])
        
        ax = plt.subplot(2,3,1)
        ax.set_title('B_theta')
        plot_1 = ax.contourf(R, Z, self.B[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_1)
    
        ax = plt.subplot(2,3,2)
        ax.set_title('Vr')
        plot_2 = ax.contourf(R, Z, self.Vr[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_2)
    
        ax = plt.subplot(2,3,3)
        ax.set_title('Vz')
        plot_3 = ax.contourf(R, Z, self.Vz[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_3)
    
        ax = plt.subplot(2,3,4)
        ax.set_title('rho')
        plot_4 = ax.contourf(R, Z, self.rho[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_4)
        
        ax = plt.subplot(2,3,5)
        ax.set_title('p')
        plot_5 = ax.contourf(R, Z, self.p[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_5)
    
        ax = plt.subplot(2,3,6)
        ax.set_title('T')
        plot_6 = ax.contourf(R, Z, self.T[3:-3,3:-3].T, 20, cmap='plasma')
        plt.colorbar(plot_6)     
        
        plt.show()

    def evolve(self, courant=0.8, t_max=0):
        B = self.B
        Vr = self.Vr
        Vz = self.Vz
        rho = self.rho
        p = self.p
        T = self.T
        
        t = 0
        cycle = 0
        # counter = 0

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
            if cycle == 0:
                # Initial energy
                e_temp = self.compute_e(B_temp, Vr_temp, Vz_temp, rho_temp, p_temp)
            else:
                # Once time-evolved, just copy; no need to compute it again
                e_temp = e.copy()

            # Time step from CFL condition
            self.dt = self.CFL(courant, B_temp, Vr_temp, Vz_temp, rho_temp, T_temp)
            if self.dt < 1e-7:
                print('Solution has not converged. Break')
                break
                
            t += self.dt
            cycle += 1
            # counter += 1
#             if counter == 10:
#                 print(cycle, t)
#                 counter = 0
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
            source_Ur  = -1 / self.grid_r.rr * (rho_temp * Vr_temp**2 + B_temp**2 / (4 * np.pi))
            source_Uz  = -1 / self.grid_r.rr * rho_temp * Vr_temp * Vz_temp
            source_rho = -1 / self.grid_r.rr * rho_temp * Vr_temp
            source_e   = -1 / self.grid_r.rr * Vr_temp * (self.rosh / (self.rosh - 1) * p_temp + 1/2 * rho_temp * (Vr_temp**2 + Vz_temp**2) + B_temp**2 / (4 * np.pi))

            # Updates largest eigenvalues of the Jacobians of the fluxes
            # eval = V + sqrt(v_s^2 + v_A^2)
            self.evalR = np.amax(np.abs(Vr_temp) + np.sqrt(self.rosh * p_temp / rho_temp + np.abs(B_temp)**2 / (4 * np.pi * np.abs(rho_temp))))
            self.evalZ = np.amax(np.abs(Vz_temp) + np.sqrt(self.rosh * p_temp / rho_temp + np.abs(B_temp)**2 / (4 * np.pi * np.abs(rho_temp))))

            # Updates conserved variables.
            B   = PDESolver(B_temp,   fluxR_B,   fluxZ_B,   source_B,   self).time_step()
            Ur  = PDESolver(Ur_temp,  fluxR_Ur,  fluxZ_Ur,  source_Ur,  self).time_step()
            Uz  = PDESolver(Uz_temp,  fluxR_Uz,  fluxZ_Uz,  source_Uz,  self).time_step()
            rho = PDESolver(rho_temp, fluxR_rho, fluxZ_rho, source_rho, self).time_step()
            e   = PDESolver(e_temp,   fluxR_e,   fluxZ_e,   source_e,   self).time_step()
            
            # Imposes boundary conditions on conserved variables
            B   = self.boundary_conditions(B,   'z', 'periodic')
            Ur  = self.boundary_conditions(Ur,  'z', 'periodic')
            Uz  = self.boundary_conditions(Uz,  'z', 'periodic')
            rho = self.boundary_conditions(rho, 'z', 'periodic')
            e   = self.boundary_conditions(e,   'z', 'periodic')

            # Updates primitive and post-processed variables
            Vr = Ur / rho
            Vz = Uz / rho
            p  = self.compute_p(B, Vr, Vz, rho, e)
            T  = self.compute_T(rho, p)
            
            # Imposes boundary conditions on post-processed variables
            p = self.boundary_conditions(p, 'z', 'periodic')
            T = self.boundary_conditions(T, 'z', 'periodic')
        
        t1 = time.time()
        print('Simulation finished.')
        print('Time elapsed:', t1-t0)
        
        self.B = 1e-5 * B
        self.Vr = 1e3 * Vr
        self.Vz = 1e3 * Vz
        self.rho = 1e-15 * rho
        self.p = 1e-9 * p
        self.T = 1e0 * T
        return self.B, self.Vr, self.Vz, self.rho, self.p, self.T

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

    def CFL(self, courant, B, Vr, Vz, rho, T):
        # This CFL condition isn't correct, need to adjust order of magnitude
        v_fluid = np.sqrt(np.abs(Vr)**2 + np.abs(Vz)**2)
        v_alfven2 = np.abs(B)**2 / (4 * np.pi * np.abs(rho))
        v_sound2 = 2 * self.rosh * np.abs(T) / Const.m_i
        v_magnetosonic = np.sqrt(v_alfven2 + v_sound2)
        v_courant = np.amax(v_fluid + v_magnetosonic)
        dt = self.grid_r.dr / v_courant * 1e-1 * courant
        
        return dt

    def boundary_conditions(self, u, dimension, type_of):
        u_temp = u
        
        if dimension == 'z' and type_of == 'periodic':
            u_temp[:, 0] = u_temp[:, -6]
            u_temp[:, 1] = u_temp[:, -5]
            u_temp[:, 2] = u_temp[:, -4]
            u_temp[:, -3] = u_temp[:, 3]
            u_temp[:, -2] = u_temp[:, 4]
            u_temp[:, -1] = u_temp[:, 5]
        elif dimension == 'r' and type_of == 'dirichlet':
            pass
        elif dimension == 'r' and type_of == 'neumann':
            pass
        elif dimension == 'r' and type_of == '1/r':
            pass
        
        return u_temp
        

class PDESolver:
    def __init__(self, u, fluxR, fluxZ, source, evo):
        self.u = u
        self.fluxR = fluxR
        self.fluxZ = fluxZ
        self.source = source
        self.evo = evo

    # 3rd order Runge-Kutta time discretization
    def time_step(self):
        nr = self.evo.grid_r.nr
        nz = self.evo.grid_z.nr
        dt = self.evo.dt
        
        u1 = np.zeros((nr, nz))
        u2 = np.zeros((nr, nz))
        u3 = np.zeros((nr, nz))
        
        # aaa is the modified np.apply_along_axis, see aaa_source.py
        u1 = self.u + dt * (aaa(self.derivativeR, 0, self.u, self.fluxR) + aaa(self.derivativeZ, 1, self.u, self.fluxZ) - self.source)
        u2 = 3/4 * self.u + 1/4 * u1 + 1/4 * dt * (aaa(self.derivativeR, 0, u1, self.fluxR) + aaa(self.derivativeZ, 1, u1, self.fluxZ) - self.source)
        u3 = 1/3 * self.u + 2/3 * u2 + 2/3 * dt * (aaa(self.derivativeR, 0, u2, self.fluxR) + aaa(self.derivativeZ, 1, u2, self.fluxZ) - self.source)

        return u3
        
    # 5th order WENO-Z FD scheme, component-wise
    def derivativeR(self, u, flux):
        self.a = self.evo.evalR
        self.flux = flux
        nr = self.evo.grid_r.nr
        dr = self.evo.grid_r.dr

        dv = np.zeros(nr)
        dv[3: -3] = -1 / dr * (self.fhat(u, 'plus', 0) + self.fhat(u, 'minus', 0) - self.fhat(u, 'plus', -1) - self.fhat(u, 'minus', -1))
        return dv
        
    def derivativeZ(self, u, flux):
        self.a = self.evo.evalZ
        self.flux = flux
        nz = self.evo.grid_z.nr
        dz = self.evo.grid_z.dr
        
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

            # Borges, Costa, and Don (2006): WENO-Z, higher order smoothness indicators for 5th order convergence
            tau5 = np.max(IS2 - IS0)
            if tau5 < epsilon:
                tau5 = epsilon

            IS0_Z = (IS0 + epsilon) / (IS0 + tau5)
            IS1_Z = (IS1 + epsilon) / (IS1 + tau5)
            IS2_Z = (IS2 + epsilon) / (IS2 + tau5)

            # alpha = d / IS
            alpha0 = 1/10 / IS0_Z
            alpha1 = 6/10 / IS1_Z
            alpha2 = 3/10 / IS2_Z
        
        elif pm == 'minus':
            IS0 = 13/12 * (fminus[2+g: -4+g] - 2 * fminus[3+g: -3+g] + fminus[4+g: -2+g])**2 + 1/4 * (fminus[2+g: -4+g] - 4 * fminus[3+g: -3+g] + 3 * fminus[4+g: -2+g])**2
            IS1 = 13/12 * (fminus[3+g: -3+g] - 2 * fminus[4+g: -2+g] + fminus[5+g: -1+g])**2 + 1/4 * (fminus[3+g: -3+g] - fminus[5+g: -1+g])**2

            if g == 0:
                IS2 = 13/12 * (fminus[4+g: -2+g] - 2 * fminus[5+g: -1+g] + fminus[6+g:  ])**2 + 1/4 * (3 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g:  ])**2
            elif g == -1:
                IS2 = 13/12 * (fminus[4+g: -2+g] - 2 * fminus[5+g: -1+g] + fminus[6+g: g])**2 + 1/4 * (3 * fminus[4+g: -2+g] - 4 * fminus[5+g: -1+g] + fminus[6+g: g])**2

            tau5 = np.max(IS2 - IS0)
            if tau5 < epsilon:
                tau5 = epsilon

            IS0_Z = (IS0 + epsilon) / (IS0 + tau5)
            IS1_Z = (IS1 + epsilon) / (IS1 + tau5)
            IS2_Z = (IS2 + epsilon) / (IS2 + tau5)

            alpha0 = 3/10 / IS0_Z
            alpha1 = 6/10 / IS1_Z
            alpha2 = 1/10 / IS2_Z
        
        w0 = alpha0 / (alpha0 + alpha1 + alpha2)
        w1 = alpha1 / (alpha0 + alpha1 + alpha2)
        w2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
        return w0, w1, w2

        
        
        
        
        
        
