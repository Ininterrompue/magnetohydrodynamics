import numpy
from matplotlib import pyplot
from scipy.sparse import dia_matrix
from scipy.linalg import eig

nr = 2 + 200
r_max = 10.0
dr = r_max/(nr - 2)
r = numpy.linspace(-dr/2, r_max + dr/2, nr)
rr = numpy.reshape(numpy.linspace(-dr/2, r_max + dr/2, nr), (nr, 1))

P = numpy.exp(-rr**4) + 0.05

def FD_matrix():
    one = numpy.ones(nr)
    dv = (dia_matrix((one, 1), shape=(nr, nr)) - dia_matrix((one, -1), shape=(nr, nr))).toarray()/(2*dr)
    return dv

dv = FD_matrix()
op = dv/2 + dia_matrix((1.0/r, 0), shape=(nr, nr)).toarray()
op[0, 1] = -op[0, 0]
op[nr - 1, nr - 2] = -op[nr - 1, nr - 1]

rhs = -dv @ P
rhs[-1] = 0
rhs[0] = 0

# Equilibrium rho and B. Mass of ion set to unity.
rho_0 = P/2.0
B2 = numpy.linalg.solve(op, rhs)
B_0 = numpy.sign(B2)*numpy.sqrt(numpy.abs(B2))
J_0 = (dv @ (rr*B_0))/rr

pyplot.plot(r[1: -1], B_0[1: -1], 
            r[1: -1], rho_0[1: -1],
            r[1: -1], J_0[1: -1])
pyplot.show()

##
m0 = numpy.zeros((nr, nr))
k = 1

def DV_product(vec):
    return (numpy.diagflat(vec[1:], 1) - numpy.diagflat(vec[:-1], -1))/(2*dr)

def zero_out(M):
    M[0, :] = 0
    M[-1, :] = 0
    return M

# Generalized eigenvalue problem matrix
G = -numpy.identity(4*nr)
G[0, 0] = G[nr - 1, nr - 1] = G[nr, nr] = G[2*nr - 1, 2*nr - 1] = 0
G[2*nr, 2*nr] = G[3*nr - 1, 3*nr - 1] = G[3*nr, 3*nr] = G[-1, -1] = 0

# Elements of the block matrix, of which 8 are zero.
m3 = 1j/rr*DV_product(rr*rho_0)
zero_out(m3)
m4 = numpy.diagflat(-k*rho_0, 0)
zero_out(m4)
m7 = 1j*DV_product(B_0)
zero_out(m7)
m8 = numpy.diagflat(-k*B_0, 0)
zero_out(m8)
m9 = 2j/(rho_0)*dv
zero_out(m9)
m10 = 1j/(4*numpy.pi*rho_0*rr**2)*DV_product(rr**2*B_0)
zero_out(m10)
m13 = numpy.diagflat(-2.0*k/rho_0, 0)
zero_out(m13)
m14 = numpy.diagflat(-k*B_0/(4.0*numpy.pi*rho_0), 0)
zero_out(m14)

# 0: f = 0. 1: f' = 0.
def BC(M, bc_begin, bc_end):
    if bc_begin == 0:
        M[0, 0] = 1
        M[0, 1] = 1
    elif bc_begin == 1:
        M[0, 0] = 1
        M[0, 1] = -1

    if bc_end == 0:
        M[nr - 1, nr - 1] = 1
        M[nr - 1, nr - 2] = 1
    elif bc_end == 1:
        M[nr - 1, nr - 1] = 1
        M[nr - 1, nr - 2] = -1

    return M

m1 = numpy.zeros((nr, nr))
m6 = numpy.zeros((nr, nr))
m11 = numpy.zeros((nr, nr))
m16 = numpy.zeros((nr, nr))
m1 = BC(m1, 1, 0)
m6 = BC(m6, 0, 1)
m11 = BC(m11, 0, 1)
m16 = BC(m16, 1, 0)

M = numpy.block([[m1, m0, m3, m4], [m0, m6, m7, m8], [m9, m10, m11, m0], [m13, m14, m0, m16]])

evals, evects = eig(M, G)

def plot_eigenvalues(evals, evects):
    pyplot.scatter(evals.real, evals.imag, s=1)
    pyplot.title('Omega')
    pyplot.xlabel('Re')
    pyplot.ylabel('Im')
    pyplot.show()

plot_eigenvalues(evals, evects)

# ith mode by magnitude of imaginary part
def plot_mode(i, evals, evects):
    index = numpy.argsort(evals.imag)

    omega = evals[index[i]]
    v_omega = evects[:, index[i]]

    f = pyplot.figure()
    f.suptitle(omega)

    # def f1(x): return numpy.abs(x)
    # def f2(x): return numpy.unwrap(numpy.angle(x)) / (2*3.14159)
    def f1(x): return numpy.real(x)
    def f2(x): return numpy.imag(x)

    ax = pyplot.subplot(2,2,1)
    ax.set_title("rho")
    rho = v_omega[0: nr]
    t = numpy.exp(-1j*numpy.angle(rho[0]))
    ax.plot(r[1: -1], f1(t*rho[1: -1]),
            r[1: -1], f2(t*rho[1: -1]) )
              
    ax = pyplot.subplot(2,2,2)
    ax.set_title("B_theta")
    B = v_omega[nr: 2*nr]
    ax.plot(r[1: -1], f1(t*B[1: -1]),
            r[1: -1], f2(t*B[1: -1]) )
            
    ax = pyplot.subplot(2,2,3)
    ax.set_title("V_r")
    V_r = v_omega[2*nr: 3*nr]
    ax.plot(r[1: -1], f1(t*V_r[1: -1]),
            r[1: -1], f2(t*V_r[1: -1]) )
            
    ax = pyplot.subplot(2,2,4)
    ax.set_title("V_z")
    V_z = v_omega[3*nr: 4*nr]
    ax.plot(r[1: -1], f1(t*V_z[1: -1]),
            r[1: -1], f2(t*V_z[1: -1]) )

    pyplot.show()

# for jj in range(4*nr):
#    plot_mode(jj, evals, evects)

plot_mode(-1, evals, evects)

# ##
# pyplot.figure()
# pyplot.imshow(numpy.real(evects))
# pyplot.show()

