import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dia_matrix
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def grid(size=10, max=1, ghost=1):
    global nx, dx, x, xx, gh
    gh = ghost
    nx = 2 * gh + size
    dx = max/size
    x = np.linspace(-(2*gh - 1)/2 * dx, max + (2*gh - 1)/2 * dx, nx)
    xx = np.reshape(x, (nx, 1))
    
    return nx, dx, x, xx


# Testing inviscid 1D Burger's equation
def test(u, t_max, WENO, fhat, f, fp, fm, SI_weights):
    dt = dr / 2
    t = 0
    
    while t < t_max:
        t = t + dt
        u_temp = u.copy()
        u = runge_kutta(WENO, fhat, f, fp, fm, SI_weights, u_temp, dt)
#         u[-3] = u[0]
#         u[-2] = u[1]
#         u[-1] = u[2]
        
    plt.plot(r[3: -3], u[3: -3], r[3: -3], u_0[3: -3])
    plt.legend(['u', 'u_0'])
    plt.xlabel('x')
    plt.title('Time evolution')
    plt.show()

    
def runge_kutta(WENO, fhat, f, fp, fm, SI_weights, u, dt):
    u1 = np.reshape(np.zeros(nr), (nr, 1))
    u2 = np.reshape(np.zeros(nr), (nr, 1))
    u3 = np.reshape(np.zeros(nr), (nr, 1))
    u1 = u + dt * WENO(fhat, f, fp, fm, SI_weights, u)
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * WENO(fhat, f, fp, fm, SI_weights, u1)
    u3 = 1/3 * u + 2/3 * u2 + 2/3 * dt * WENO(fhat, f, fp, fm, SI_weights, u2)
    
    return u3


# FLUX
def f(u):
    return 1/2 * u**2
    
# max(abs(u))    
def fp(f, u):
    a = max(abs(u))
    return 1/2 * (f(u) + a * u)
def fm(f, u):
    a = max(abs(u))
    return 1/2 * (f(u) - a * u)


# Output is a spatial derivative only
def WENO(fhat, f, fp, fm, SI_weights, u):
    df_dr = np.reshape(np.zeros(nr), (nr, 1))
    df_dr[3: -3] = -1/dr * (fhat(1, 0, f, fp, fm, SI_weights, u) + fhat(-1, 0, f, fp, fm, SI_weights, u) - fhat(1, -1, f, fp, fm, SI_weights, u) - fhat(-1, -1, f, fp, fm, SI_weights, u))
    
    return df_dr


def fhat(pm, g, f, fp, fm, SI_weights, u):
    if pm == 1:
        fhat0 = 1/3 * fp(f, u[1+g: -5+g]) - 7/6 * fp(f, u[2+g: -4+g]) + 11/6 * fp(f, u[3+g: -3+g])
        fhat1 = -1/6 * fp(f, u[2+g: -4+g]) + 5/6 * fp(f, u[3+g: -3+g]) + 1/3 * fp(f, u[4+g: -2+g])
        fhat2 = 1/3 * fp(f, u[3+g: -3+g]) + 5/6 * fp(f, u[4+g: -2+g]) - 1/6 * fp(f, u[5+g: -1+g])
        
        w0, w1, w2 = SI_weights(+1, g, f, fp, fm, u)
        
    elif pm == -1:
        fhat0 = 1/3 * fm(f, u[4+g: -2+g]) + 5/6 * fm(f, u[3+g: -3+g]) - 1/6 * fm(f, u[2+g: -4+g])
        fhat1 = -1/6 * fm(f, u[5+g: -1+g]) + 5/6 * fm(f, u[4+g: -2+g]) + 1/3 * fm(f, u[3+g: -3+g])

        if g == 0:
            fhat2 = 1/3 * fm(f, u[6+g: ]) - 7/6 * fm(f, u[5+g: -1+g]) + 11/6 * fm(f, u[4+g: -2+g])
        elif g == -1:
            fhat2 = 1/3 * fm(f, u[6+g: g]) - 7/6 * fm(f, u[5+g: -1+g]) + 11/6 * fm(f, u[4+g: -2+g])
            
        w0, w1, w2 = SI_weights(-1, g, f, fp, fm, u)

    return w0 * fhat0 + w1 * fhat1 + w2 * fhat2

 
def SI_weights(pm, g, f, fp, fm, u):
    # Small parameter to avoid division by 0 in Weights
    epsilon = 1e-6
    
    # pm: positive/negative flux.
    if pm == 1:
        IS0 = 13/12 * (fp(f, u[1+g: -5+g]) - 2 * fp(f, u[2+g: -4+g]) + fp(f, u[3+g: -3+g]))**2 + 1/4 * (fp(f, u[1+g: -5+g]) - 4 * fp(f, u[2+g: -4+g]) + 3 * fp(f, u[3+g: -3+g]))**2
        IS1 = 13/12 * (fp(f, u[2+g: -4+g]) - 2 * fp(f, u[3+g: -3+g]) + fp(f, u[4+g: -2+g]))**2 + 1/4 * (fp(f, u[2+g: -4+g]) - fp(f, u[4+g: -2+g]))**2
        IS2 = 13/12 * (fp(f, u[3+g: -3+g]) - 2 * fp(f, u[4+g: -2+g]) + fp(f, u[5+g: -1+g]))**2 + 1/4 * (3 * fp(f, u[3+g: -3+g]) - 4 * fp(f, u[4+g: -2+g]) + fp(f, u[5+g: -1+g]))**2
        
        alpha0 = 1/10 / (epsilon + IS0)**2
        alpha1 = 6/10 / (epsilon + IS1)**2
        alpha2 = 3/10 / (epsilon + IS2)**2
        
    elif pm == -1:
        IS0 = 13/12 * (fm(f, u[2+g: -4+g]) - 2 * fm(f, u[3+g: -3+g]) + fm(f, u[4+g: -2+g]))**2 + 1/4 * (fm(f, u[2+g: -4+g]) - 4 * fm(f, u[3+g: -3+g]) + 3 * fm(f, u[4+g: -2+g]))**2
        IS1 = 13/12 * (fm(f, u[3+g: -3+g]) - 2 * fm(f, u[4+g: -2+g]) + fm(f, u[5+g: -1+g]))**2 + 1/4 * (fm(f, u[3+g: -3+g]) - fm(f, u[5+g: -1+g]))**2

        if g == 0:
            IS2 = 13/12 * (fm(f, u[4+g: -2+g]) - 2 * fm(f, u[5+g: -1+g]) + fm(f, u[6+g: ]))**2 + 1/4 * (3 * fm(f, u[4+g: -2+g]) - 4 * fm(f, u[5+g: -1+g]) + fm(f, u[6+g: ]))**2
        elif g == -1:
            IS2 = 13/12 * (fm(f, u[4+g: -2+g]) - 2 * fm(f, u[5+g: -1+g]) + fm(f, u[6+g: g]))**2 + 1/4 * (3 * fm(f, u[4+g: -2+g]) - 4 * fm(f, u[5+g: -1+g]) + fm(f, u[6+g: g]))**2
        
        alpha0 = 3/10 / (epsilon + IS0)**2
        alpha1 = 6/10 / (epsilon + IS1)**2
        alpha2 = 1/10 / (epsilon + IS2)**2
        
    w0 = alpha0 / (alpha0 + alpha1 + alpha2)
    w1 = alpha1 / (alpha0 + alpha1 + alpha2)
    w2 = alpha2 / (alpha0 + alpha1 + alpha2)
    
    return w0, w1, w2


## GRID SIZE
nr, dr, r, rr = grid(size=100, max=10, ghost=3)

global u_0
u_0 = np.exp(-(rr-5)**2)
u = u_0.copy()
test(u, 2, WENO, fhat, f, fp, fm, SI_weights)



