import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def grid(size=10, max=1, ghost=1):
    global nx, dx, x, xx, gh
    nx = 2 * gh + size
    dx = max/size
    x = np.linspace(-(2*gh - 1)/2 * dx, max + (2*gh - 1)/2 * dx, nx)
    xx = np.reshape(x, (nx, 1))
    gh = ghost
    
    return nx, dx, x, xx


def FD_matrix(nr, dr, order):
    one = np.ones(nr)

    if order == 1:
        dv = (dia_matrix((one, 1), shape=(nr, nr)) 
        - dia_matrix((one, -1), shape=(nr, nr))).toarray() / (2 * dr)
    elif order == 2:
        dv = (dia_matrix((one, 1), shape=(nr, nr)) 
        + dia_matrix((one, -1), shape=(nr, nr)) 
        - 2 * dia_matrix((one, 0), shape=(nr, nr))).toarray() / dr**2
    
    return dv
    

def equilibrium(FD_matrix, zero_out):
    P = np.exp(-rr**4) + 0.05
    op = zero_out(FD_matrix(nr, dr, 1) / 2 + dia_matrix((1.0 / r, 0), shape=(nr, nr)).toarray())
    
    op[0, 0] = 1
    op[0, 1] = 1
    op[nr - 1, nr - 1] = -r[-1] / r[-2]
    op[nr - 1, nr - 2] = 1

    rhs = -FD_matrix(nr, dr, 1) @ P
    rhs[-1] = 0
    rhs[0] = 0
    
    # Equilibrium rho and B. Mass of ion set to unity.
    rho_0 = P / 2.0
    B2 = np.linalg.solve(op, rhs)
    B_0 = np.sqrt(4 * np.pi) * np.sign(B2) * np.sqrt(np.abs(B2))
    J_0 = (FD_matrix(nr, dr, 1) @ (rr * B_0)) / rr
    return rho_0, B_0, J_0


def DV_product(vec, dr):
    return (np.diagflat(vec[1: ], 1) - np.diagflat(vec[: -1], -1)) / (2 * dr)


# Generalized eigenvalue problem matrix
def create_G(nr):
    G = np.identity(7*nr)
    G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
    G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = 0
    G[4*nr - 1, 4*nr - 1] = G[4*nr, 4*nr] = G[5*nr - 1, 5*nr - 1] = 0
    G[5*nr, 5*nr] = G[6*nr - 1, 6*nr - 1] = G[6*nr - 6*nr] = 0
    G[7*nr - 1, 7*nr - 1] = G[-1, -1] = 0

#     G = np.identity(4*nr)
#     G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
#     G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = 0
#     G[4*nr - 1, 4*nr - 1] = 0

    return G


def zero_out(M):
    M[0, :] = 0
    M[-1, :] = 0
    return M


# 0: f = 0. 1: f' = 0.
def BC(m, nr, bc_begin, bc_end):
    if bc_begin == 0:
        m[0, 0] = 1
        m[0, 1] = 1
    elif bc_begin == 1:
        m[0, 0] = 1
        m[0, 1] = -1

    if bc_end == 0:
        m[nr - 1, nr - 1] = 1
        m[nr - 1, nr - 2] = 1
    elif bc_end == 1:
        m[nr - 1, nr - 1] = 1
        m[nr - 1, nr - 2] = -1
        
    return m
    

# State vector X = (rho, B, V)
def create_M(rr, nr, dr, rho_0, B_0, FD_matrix, k, zero_out, BC, DV_product):
    m0 = np.zeros((nr, nr))
    
    m_rho_Vr = zero_out(-1j / rr * DV_product(rr * rho_0, dr))
    m_rho_Vtheta = zero_out(np.diagflat(2 * np.pi * m * rho_0 / rr, 0))
    m_rho_Vz = zero_out(np.diagflat(k * rho_0, 0))
    
    m_Br_Vr = zero_out(np.diagflat(-2 * np.pi * m * B_0 / rr, 0))
    
    m_Btheta_Vr = zero_out(-1j * DV_product(B_0, dr))
    m_Btheta_Vz = zero_out(np.diagflat(k * B_0, 0))
    
    m_Bz_Vz = zero_out(np.diagflat(-2 * np.pi * m * B_0 / rr, 0))
    
    m_Vr_rho = zero_out(-2j / rho_0 * FD_matrix(nr, dr, 1))
    m_Vr_Br = zero_out(np.diagflat(-m * B_0 / (2 * rho_0 * rr), 0))
    m_Vr_Btheta = zero_out(-1j / (4 * np.pi * rho_0 * rr**2) * DV_product(rr**2 * B_0, dr))
    
    m_Vtheta_rho = zero_out(np.diagflat(4 * np.pi * m / (rr * rho_0), 0)) 
    
    m_Vz_rho = zero_out(np.diagflat(2.0 * k / rho_0, 0))
    m_Vz_Btheta = zero_out(np.diagflat(k * B_0 / (4.0 * np.pi * rho_0), 0))
    m_Vz_Bz = zero_out(np.diagflat(-B_0 * m / (2 * rho_0 * rr), 0))
    
    m_Br_Br = np.zeros((nr, nr))
    m_Btheta_rho = np.zeros((nr, nr))
    m_Btheta_Btheta = np.zeros((nr, nr))
    m_Bz_Br = np.zeros((nr, nr))
    m_Bz_Bz = np.zeros((nr, nr))
    
    m_rho_rho = np.zeros((nr, nr))
    m_Vr_Vr = np.zeros((nr, nr))
    m_Vtheta_Vtheta = np.zeros((nr, nr))
    m_Vz_Vz = np.zeros((nr, nr))
    
    # Resistive term
    m_Br_Br = m_Br_Br + 1j * D_eta * zero_out((1 / rr * FD_matrix(nr, dr, 1) + FD_matrix(nr, dr, 2)) - np.diagflat(4 * np.pi**2 * m**2 / rr**2 + k**2 + 1 / rr**2, 0))
    m_Btheta_Btheta = m_Btheta_Btheta + 1j * D_eta * zero_out((1 / rr * FD_matrix(nr, dr, 1) + FD_matrix(nr, dr, 2)) - np.diagflat(4 * np.pi**2 * m**2 / rr**2 + k**2 + 1 / rr**2, 0))
    m_Bz_Bz = m_Bz_Bz + 1j * D_eta * zero_out((1 / rr * FD_matrix(nr, dr, 1) + FD_matrix(nr, dr, 2)) - np.diagflat(4 * np.pi**2 * m**2 / rr**2 + k**2, 0))
    
    # Hall term
    m_Br_Br = m_Br_Br + D_H * zero_out(np.diagflat(-k / (rr * rho_0) * (FD_matrix(nr, dr, 1) @ (rr * B_0)), 0))
    m_Btheta_rho = m_Btheta_rho + D_H * zero_out(np.diagflat(k * B_0 / (rr * rho_0**2) * FD_matrix(nr, dr, 1) @ (rr * B_0), 0))
    m_Btheta_Btheta = m_Btheta_Btheta + D_H * zero_out(np.diagflat(-2 * k * B_0 / (rr * rho_0) + (FD_matrix(nr, dr, 1) @ (1 / rho_0)) * B_0 * k, 0))
    m_Bz_Br = m_Bz_Br + D_H * zero_out(-1j / (rr * rho_0) * ((FD_matrix(nr, dr, 1) @ (rr * B_0)) * FD_matrix(nr, dr, 1) + np.diagflat(FD_matrix(nr, dr, 2) @ (rr * B_0), 0))
                                       - np.diagflat(1j * FD_matrix(nr, dr, 1) @ (1 / rho_0) * 1 / rr * FD_matrix(nr, dr, 1) @ (rr * B_0), 0))
    
    # Electron pressure term
    m_Btheta_rho = m_Btheta_rho + D_P * zero_out(np.diagflat(k * (1 / rho_0**2 * (FD_matrix(nr, dr, 1) @ rho_0) + FD_matrix(nr, dr, 1) @ (1 / rho_0)), 0))
    
    # Boundary conditions
    m_rho_rho = BC(m_rho_rho, nr, 1, 0)
    m_Br_Br = BC(m_Br_Br, nr, 0, 0)
    m_Btheta_Btheta = BC(m_Btheta_Btheta, nr, 0, 0)
    m_Bz_Bz = BC(m_Bz_Bz, nr, 0, 0)
    m_Vr_Vr = BC(m_Vr_Vr, nr, 0, 1)
    m_Vtheta_Vtheta = BC(m_Vtheta_Vtheta, nr, 0, 1)
    m_Vz_Vz = BC(m_Vz_Vz, nr, 1, 1)
    
    M = np.block([[m_rho_rho, m0, m0, m0, m_rho_Vr, m_rho_Vtheta, m_rho_Vz], 
				[m0, m_Br_Br, m0, m0, m_Br_Vr, m0, m0], 
				[m_Btheta_rho, m0, m_Btheta_Btheta, m0, m_Btheta_Vr, m0, m_Btheta_Vz], 
				[m0, m_Bz_Br, m0, m_Bz_Bz, m0, m0, m_Bz_Vz],
				[m_Vr_rho, m_Vr_Br, m_Vr_Btheta, m0, m_Vr_Vr, m0, m0],
				[m_Vtheta_rho, m0, m0, m0, m0, m_Vtheta_Vtheta, m0],
				[m_Vz_rho, m0, m_Vz_Btheta, m_Vz_Bz, m0, m0, m_Vz_Vz]])

    return M


def gamma_vs_k(G, rr, nr, dr, rho_0, B_0, FD_matrix, kk, zero_out, BC, DV_product, create_M):
    gamma = []
    for K in kk: 
	    M = create_M(rr, nr, dr, rho_0, B_0, FD_matrix, K, zero_out, BC, DV_product)
	    eval = eigs(M, k=1, M=G, sigma=20j, which='LI', return_eigenvectors=False)
	    gamma.append(eval.imag)

    plt.plot(kk, gamma)
    plt.title('Largest mode')
    plt.xlabel('k')
    plt.ylabel('gamma')
    plt.show()


def convergence(grid, res, FD_matrix, equilibrium, zero_out, BC, DV_product, create_G):
    gamma = []
    for i_res in res:
        nr2, r_max2, dr2, r2, rr2 = grid(size=int(i_res), max=5.0)
        P2 = np.exp(-rr2**4) + 0.05
        rho_0, B_0, J_0 = equilibrium(FD_matrix, zero_out, r2, rr2, nr2, dr2)
        M = create_M(rr2, nr2, dr2, rho_0, B_0, FD_matrix, 1, zero_out, BC, DV_product)
        G = create_G(nr2)
        eval = eigs(M, k=1, M=G, sigma=10j, which='LI', return_eigenvectors=False)
        gamma.append(eval.imag)
#         eval, evec = eigs(M, k=1, M=G, sigma=2j, which='LI', return_eigenvectors=True)
#         rho_test = evec[nr2: 2*nr2]
#         rho_test = rho_test * np.exp(-1j * np.angle(rho_test[0]))
#         integral = dr2 * sum(np.abs(rho_test.imag))
#         gamma.append(integral)
	
#     plt.loglog(res, gamma, basex=2, basey=2)
    plt.plot(res, gamma)
    plt.title('Resolution convergence of gamma')
    plt.xlabel('Resolution')
    plt.ylabel('gamma')
#     plt.title('Integral of imaginary part of rho')
#     plt.xlabel('nr')
    plt.show()


# def L(WENO, f, fp, fm, phi_N, u):
#     udv2 = np.reshape(np.zeros(nr), (nr, 1))
#     udv2[1: -1] = 1 / dr**2 * (u[0: -2] - 2 * u[1: -1] + u[2: ])
#     return WENO(f, fp, fm, phi_N, u)


def runge_kutta(WENO, f, fp, fm, phi_N, u, dt):
    u1 = np.reshape(np.zeros(nr), (nr, 1))
    u2 = np.reshape(np.zeros(nr), (nr, 1))
    u3 = np.reshape(np.zeros(nr), (nr, 1))
    u1 = u + dt * WENO(f, fp, fm, phi_N, u)
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * WENO(f, fp, fm, phi_N, u1)
    u3 = 1/3 * u + 2/3 * u2 + 2/3 * dt * WENO(f, fp, fm, phi_N, u2)
    
    return u3


# Testing inviscid 1D Burger's equation
def test(B_0, t_max, WENO, f, fp, fm, phi_N):
    B = B_0
    dt = dr**2 / 2
    t = 0
    while t < t_max:
        t = t + dt
        B_temp = B.copy()
        B = runge_kutta(WENO, f, fp, fm, phi_N, B_temp, dt)
        B[0] = 0
        B[-1] = B[-2] * r[-2] / r[-1]
        
    plt.plot(r[1: -1], B[1: -1])
    plt.legend(['B'])
    plt.xlabel('r/r0')
    plt.title('Time evolution')
    plt.show()      


def phi_N(a, b, c, d):
    # Small parameter to avoid division by 0 in Weights
    epsilon = 1e-6
    
    # Smoothness indicators
    IS0 = 13 * (a - b)**2 + 3 * (a - 3*b)**2
    IS1 = 13 * (b - c)**2 + 3 * (b + c)**2
    IS2 = 13 * (c - d)**2 + 3 * (3*c - d)**2
    
    # Weights
    alpha0 = 1 / (epsilon + IS0)**2
    alpha1 = 6 / (epsilon + IS1)**2
    alpha2 = 3 / (epsilon + IS2)**2
    omega0 = alpha0 / (alpha0 + alpha1 + alpha2)
    omega2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
    return 1/3 * omega0 * (a - 2*b + c) + 1/6 * (omega2 - 1/2) * (b - 2*c + d)
    
    
def f(u):
    return 1/2 * u**2
    
    
def fp(f, u):
    alpha = max(u)
    return 1/2 * (f(u) + alpha * u)
def fm(f, u):
    alpha = max(u)
    return 1/2 * (f(u) - alpha * u)    
 
   
def WENO(f, fp, fm, phi_N, u):
    
    # CHECK NEGATIVE FLUX COMPONENTS
    fhat_right = np.reshape(np.zeros(nr), (nr, 1))
    fhat_right[3: -3] = (1/12 * (-f(u[2: -4]) + 7 * f(u[3: -3]) + 7 * f(u[4: -2]) - f(u[5: -1]))
                  - phi_N(fp(u[2: -4]) - fp(u[1: -5]), fp(u[3: -3]) - fp(u[2: -4]), fp(u[4: -2]) - fp(u[3: -3]), fp(u[5: -1]) - fp(u[4: -2]))
                  + phi_N(fm(u[6:   ]) - fm(u[5: -1]), fm(u[5: -1]) - fm(u[4: -2]), fm(u[4: -2]) - fm(u[3: -3]), fm(u[3: -3]) - fm(u[2: -4])))
    
    fhat_left  = np.reshape(np.zeros(nr), (nr, 1))
    fhat_left[3: -3]  = (1/12 * (-f(u[1: -5]) + 7 * f(u[2: -4]) + 7 * f(u[3: -3]) - f(u[4: -2]))
                  - phi_N(fp(u[1: -5]) - fp(u[0: -6]), fp(u[2: -4]) - fp(u[1: -5]), fp(u[3: -3]) - fp(u[2: -4]), fp(u[4: -2]) - fp(u[3: -3]))
                  + phi_N(fm(u[5: -1]) - fm(u[4: -2]), fm(u[4: -2]) - fm(u[3: -3]), fm(u[3: -3]) - fm(u[2: -4]), fm(u[2: -4]) - fm(u[1: -5])))
    
    df_dr = np.reshape(np.zeros(nr), (nr, 1))
    df_dr[3: -3] = 1/dr * (fhat_right[3: -3] - fhat_left[3: -3])
    
    return df_dr


def pinch(rho, B, rr, nr, dr, t_max):
#     nt = int(t_max / dt)
    t = 0
    Vr = np.reshape(np.zeros(nr), (nr, 1))
#     dt = 0.000001
  
    while t < t_max:
        rho_temp = rho.copy()
        B_temp = B.copy()
        Vr_temp = Vr.copy()       

        v_A = max(B / np.sqrt(4 * np.pi * np.abs(rho)))
        dt = dr / (15 * v_A)
        t = t + dt
        print(v_A, t)
                
        rho[1: -1] = rho_temp[1: -1] - dt / (rr[1: -1] * 2 * dr) * (rr[2: ] * rho_temp[2: ] * Vr_temp[2: ] - rr[: -2] * rho_temp[: -2] * Vr_temp[: -2])
        B[1: -1] = (B_temp[1: -1] - dt / (2 * dr) * (Vr_temp[2: ] * B_temp[2: ] - Vr_temp[: -2] * B_temp[: -2]) 
                    + dt * D_eta * ((B_temp[2: ] - 2 * B_temp[1: -1] + B_temp[: -2]) / dr**2 + (B_temp[2: ] - B_temp[: -2]) / (rr[1: -1] * 2 * dr) - B_temp[1: -1] / rr[1: -1]**2))
        Vr[1: -1] = (Vr_temp[1: -1] - 2 * dt / (rho_temp[1: -1] * 2 * dr) * (rho_temp[2: ] - rho_temp[: -2]) 
                     - B_temp[1: -1] * dt / (4 * np.pi * rho_temp[1: -1] * rr[1: -1] * 2 * dr) * (rr[2: ] * B_temp[2: ] - rr[: -2] * B_temp[: -2]))
        
        rho[0] = rho[1]
        rho[-1] = rho[-2]
        B[0] = -B[1]
        B[-1] = rr[-2] * B[-2] / rr[-1]
        Vr[0] = -Vr[1]
        Vr[-1] = -Vr[-2]

    plt.plot(r[1: -1], B[1: -1], r[1: -1], rho[1: -1], r[1: -1], Vr[1: -1])
    plt.legend(['B', 'rho', 'V_r'])
    plt.xlabel('r/r0')
    plt.title('Time evolution')
    plt.show()      

def plot_eigenvalues(M, G):
    evals, evecs = eig(M, G)
    plt.scatter(evals.real, evals.imag, s=1)
    plt.title('Omega')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

# def f1(x): return np.abs(x)
# def f2(x): return np.unwrap(np.angle(x)) / (2 * np.pi)
def f1(x): return np.real(x)
def f2(x): return np.imag(x) 


def plot_mode(i):
    evals, evecs = eig(M, G)
    i_0 = -i
    index = np.argsort(evals.imag)
    omega = evals[index[i_0]]
    v_omega = evecs[:, index[i_0]]
    gamma = omega.imag
    
    rho = v_omega[0: nr]
    B_r = v_omega[nr: 2*nr]
    B_theta = v_omega[2*nr: 3*nr]
    B_z = v_omega[3*nr: 4*nr]
    V_r = v_omega[4*nr: 5*nr]
    V_theta = v_omega[5*nr: 6*nr]
    V_z = v_omega[6*nr: 7*nr]
    phase = np.exp(-1j * np.angle(rho[0]))
#     print(f1(phase * B_theta)[0:10])
#     print(f1(phase * B_theta)[-10:])
#   
#     print(np.angle(rho))
	
	# 1D plots of real and imaginary parts 
    f = plt.figure()
    f.suptitle(omega.imag)
            
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
    
    epsilon = 1
    t = 0
    rho_contour = rho_0[1: -1].T + epsilon * f1(z_osc[1: -1] * phase * rho[1: -1]) * np.exp(gamma * t)
    B_r_contour = epsilon * f1(z_osc[1: -1] * phase * B_r[1: -1]) * np.exp(gamma * t)
    B_theta_contour = B_0[1: -1].T + epsilon * f1(z_osc[1: -1] * phase * B_theta[1: -1]) * np.exp(gamma * t)
    B_z_contour = epsilon * f1(z_osc[1: -1] * phase * B_z[1: -1]) * np.exp(gamma * t)
    V_r_contour = epsilon * f1(z_osc[1: -1] * phase * V_r[1: -1]) * np.exp(gamma * t)
    V_theta_contour = epsilon * f1(z_osc[1: -1] * phase * V_theta[1: -1]) * np.exp(gamma * t)
    V_z_contour = epsilon * f1(z_osc[1: -1] * phase * V_z[1: -1]) * np.exp(gamma * t)
    
    # 2D contour plots
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
    
    #2D quiver plot
    R, Z = np.meshgrid(r[1: -1], z[1: -1])
    d_vec = 10
    plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], 
               V_r_contour[::d_vec, ::d_vec], V_z_contour[::d_vec, ::d_vec], 
               pivot='mid', width=0.004, scale=3.5)
    plt.title('Flow velocity')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.show()


## GRID SIZE, EQUILIBRIUM
nr, dr, r, rr = grid(size=400, max=5.0, ghost=1)
rho_0, B_0, J_0 = equilibrium(FD_matrix, zero_out)

# plt.plot(r[gh: -gh], B_0[gh: -gh], r[gh: -gh], rho_0[gh: -gh], r[gh: -gh], J_0[gh: -gh])
# plt.show()

# test(B_0, t_max, WENO, f, fp, fm, phi_N)

## PARAMETERS
# k = 1
# m = 0
# D_eta = 1e-4
# D_H = 0.3
# D_P = 0
# nz, dz, z, zz = grid(size=400, max=2*np.pi/k)
# z_osc = np.exp(1j * k * zz)
# G = create_G(nr)

## TIME EVOLUTION
# B_1 = 2 * B_0.copy()
# pinch(rho_0, B_1, rr, nr, dr, 0.3)

## CONVERGENCE TEST
# res_min = 20
# res_max = 200
# d_res = 10
# n_res = 1 + (res_max - res_min)/d_res
# res = np.linspace(res_min, res_max, n_res)
# convergence(grid, res, FD_matrix, equilibrium, zero_out, BC, DV_product, create_G)

## DISPERSION
# k_min = 0.5
# k_max = 10
# dk = 0.5
# nk = 1 + (k_max - k_min)/dk
# kk = np.linspace(k_min, k_max, nk)
# gamma_vs_k(G, rr, nr, dr, rho_0, B_0, FD_matrix, kk, zero_out, BC, DV_product, create_M)

## EIGENSYSTEM
# M = create_M(rr, nr, dr, rho_0, B_0, FD_matrix, k, zero_out, BC, DV_product)
# plot_eigenvalues(M, G)
# plot_mode(1)




