import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def grid(size=10, max=1):
	nx = 2 + size
	dx = max/(nx - 2)
	x = np.linspace(-dx/2, max + dx/2, nx)
	xx = np.reshape(x, (nx, 1))
	return nx, max, dx, x, xx


def FD_matrix(nr, dr):
    one = np.ones(nr)
    dv = (dia_matrix((one, 1), shape=(nr, nr)) - dia_matrix((one, -1), shape=(nr, nr))).toarray()/(2*dr)
    return dv


def equilibrium(dv, r, rr, nr):
	P = np.exp(-rr**4) + 0.05
	op = dv/2 + dia_matrix((1.0/r, 0), shape=(nr, nr)).toarray()
	op[0, 1] = -op[0, 0]
	op[nr - 1, nr - 2] = -op[nr - 1, nr - 1]

	rhs = -dv @ P
	rhs[-1] = 0
	rhs[0] = 0

	# Equilibrium rho and B. Mass of ion set to unity.
	rho_0 = P/2.0
	B2 = np.linalg.solve(op, rhs)
	B_0 = np.sign(B2)*np.sqrt(np.abs(B2))
	J_0 = (dv @ (rr*B_0))/rr
	return rho_0, B_0, J_0


def DV_product(vec, dr):
    return (np.diagflat(vec[1:], 1) - np.diagflat(vec[:-1], -1))/(2*dr)


# Generalized eigenvalue problem matrix
def create_G(nr):
	G = np.identity(4*nr)
	G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
	G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = G[-1, -1] = 0
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


# Elements of the block matrix, of which 8 are zero.
def create_M(rr, nr, dr, rho_0, B_0, dv, k, zero_out, BC, DV_product):
	m0 = np.zeros((nr, nr))
	m3 = zero_out(-1j/rr*DV_product(rr*rho_0, dr))
	m4 = zero_out(np.diagflat(k*rho_0, 0))	
	m7 = zero_out(-1j*DV_product(B_0, dr))
	m8 = zero_out(np.diagflat(k*B_0, 0))
	m9 = zero_out(-2j/(rho_0)*dv)
	m10 = zero_out(-1j/(4*np.pi*rho_0*rr**2)*DV_product(rr**2*B_0, dr))
	# m10 = zero_out(-1j/(4*np.pi*rho_0*rr))*(B_0*DV_product(rr, dr) + dv @ (rr * B_0))
	m13 = zero_out(np.diagflat(2.0*k/rho_0, 0))
	m14 = zero_out(np.diagflat(k*B_0/(4.0*np.pi*rho_0), 0))
	
	# Resistive MHD term
	m6 = zero_out(-1j*k**2*D_eta * np.identity(nr))
	
	m1 = np.zeros((nr, nr))
	# m6 = np.zeros((nr, nr))
	m11 = np.zeros((nr, nr))
	m16 = np.zeros((nr, nr))
	
	# BOUNDARY CONDITIONS 
	m1 = BC(m1, nr, 1, 0)
	m6 = BC(m6, nr, 0, 0)
	m11 = BC(m11, nr, 0, 1)
	m16 = BC(m16, nr, 1, 1)
	
	M = np.block([[m1, m0, m3, m4], 
				[m0, m6, m7, m8], 
				[m9, m10, m11, m0], 
				[m13, m14, m0, m16]])
	return M


def gamma_vs_k(G, rr, nr, dr, rho_0, B_0, dv, kk, zero_out, BC, DV_product):
	gamma = []
	for K in kk: 
		M = create_M(rr, nr, dr, rho_0, B_0, dv, K, zero_out, BC, DV_product)
		eval = eigs(M, k=1, M=G, sigma=10j, which='LI', return_eigenvectors=False)
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
		dv = FD_matrix(nr2, dr2)
		rho_0, B_0, J_0 = equilibrium(dv, r2, rr2, nr2)
		M = create_M(rr2, nr2, dr2, rho_0, B_0, dv, 1, zero_out, BC, DV_product)
		G = create_G(nr2)
		eval = eigs(M, k=1, M=G, sigma=10j, which='LI', return_eigenvectors=False)
		gamma.append(eval.imag)
	
	plt.plot(res, gamma)
	plt.title('Resolution convergence of gamma')
	plt.xlabel('Resolution')
	plt.ylabel('gamma')
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


# ith mode by magnitude of imaginary part
def plot_mode(i):
	evals, evecs = eig(M, G)

	i_0 = -i
	index = np.argsort(evals.imag)
	omega = evals[index[i_0]]
	v_omega = evecs[:, index[i_0]]
	gamma = omega.imag
	
	rho = v_omega[0: nr]
	B_theta = v_omega[nr: 2*nr]
	V_r = v_omega[2*nr: 3*nr]
	V_z = v_omega[3*nr: 4*nr]
	phase = np.exp(-1j * np.angle(rho[0]))
# 	print(f1(phase * V_r)[0:10])
# 	print(f1(phase * V_r)[-10:])
# 	
# 	print(f2(phase * V_z)[0:10])
# 	print(f2(phase * V_z)[-10:])
	
	# 1D plots of real and imaginary parts 
	f = plt.figure()
	f.suptitle(omega.imag)
	
	ax = plt.subplot(2,2,1)
	ax.set_title('rho')
	ax.plot(r[1: -1], f1(phase * rho[1: -1]),
            r[1: -1], f2(phase * rho[1: -1])) 
              
	ax = plt.subplot(2,2,2)
	ax.set_title('B_theta')
	ax.plot(r[1: -1], f1(phase * B_theta[1: -1]),
            r[1: -1], f2(phase * B_theta[1: -1]) )
            
	ax = plt.subplot(2,2,3)
	ax.set_title('V_r')
	ax.plot(r[1: -1], f1(phase * V_r[1: -1]),
            r[1: -1], f2(phase * V_r[1: -1]) )
            
	ax = plt.subplot(2,2,4)
	ax.set_title('V_z')	
	ax.plot(r[1: -1], f1(phase * V_z[1: -1]),
            r[1: -1], f2(phase * V_z[1: -1]) )

	plt.show()
	
	epsilon = 1
	t = 0
	rho_contour = rho_0[1: -1].T + epsilon * f1(z_osc[1: -1] * phase * rho[1: -1]) * np.exp(gamma * t)
	B_theta_contour = B_0[1: -1].T + epsilon * f1(z_osc[1: -1] * phase * B_theta[1: -1]) * np.exp(gamma * t)
	V_r_contour = epsilon * f1(z_osc[1: -1] * phase * V_r[1: -1]) * np.exp(gamma * t)
	V_z_contour = epsilon * f1(z_osc[1: -1] * phase * V_z[1: -1]) * np.exp(gamma * t)

	# 2D contour plots
	f = plt.figure()
	f.suptitle(omega.imag)
	R, Z = np.meshgrid(r[1: -1], z[1: -1])
	
	ax = plt.subplot(2,2,1)
	ax.set_title('rho')
	plot_1 = ax.contourf(R, Z, rho_contour, 100)
	plt.colorbar(plot_1)
	
	ax = plt.subplot(2,2,2)
	ax.set_title('B_theta')
	plot_2 = ax.contourf(R, Z, B_theta_contour, 100)
	plt.colorbar(plot_2)
	
	ax = plt.subplot(2,2,3)
	ax.set_title('V_r')
	plot_3 = ax.contourf(R, Z, V_r_contour, 100)
	plt.colorbar(plot_3)
	
	ax = plt.subplot(2,2,4)
	ax.set_title('V_z')
	plot_4 = ax.contourf(R, Z, V_z_contour, 100)
	plt.colorbar(plot_4)
	
	plt.show()
	
	#2D quiver plot
# 	R, Z = np.meshgrid(r[1: -1], z[1: -1])
# 	d_vec = 10
# 	plt.quiver(R[::d_vec, ::d_vec], Z[::d_vec, ::d_vec], V_r_contour[::d_vec, ::d_vec], V_z_contour[::d_vec, ::d_vec], pivot='mid', width=0.004)
# 	plt.show()

nr, r_max, dr, r, rr = grid(size=100, max=5.0)
dv = FD_matrix(nr, dr)
rho_0, B_0, J_0 = equilibrium(dv, r, rr, nr)

# plt.plot(r[1: -1], B_0[1: -1], r[1: -1], rho_0[1: -1], r[1: -1], J_0[1: -1])
# plt.show()

k = 4
D_eta = 0.1
nz, z_max, dz, z, zz = grid(size=200, max=2*np.pi/k)
z_osc = np.exp(1j * k * zz)
G = create_G(nr)

res_min = 20
res_max = 300
d_res = 20
n_res = 1 + (res_max - res_min)/d_res
res = np.linspace(res_min, res_max, n_res)
# k = 1 for convergence
# convergence(grid, res, FD_matrix, equilibrium, zero_out, BC, DV_product, create_G)

k_min = 0
k_max = 10
dk = 0.25
nk = 1 + (k_max - k_min)/dk
kk = np.linspace(k_min, k_max, nk)
# gamma_vs_k(G, rr, nr, dr, rho_0, B_0, dv, kk, zero_out, BC, DV_product)

M = create_M(rr, nr, dr, rho_0, B_0, dv, k, zero_out, BC, DV_product)

plot_eigenvalues(M, G)
plot_mode(1)




