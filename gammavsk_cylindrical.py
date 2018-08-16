import numpy as np
from mhdsystem import System, Plot
from mhdcylindrical import AnalyticalEquilibriumCyl, NumericalEquilibriumCyl, LinearCyl


k_vals = np.reshape(np.linspace(0.1, 10, 50), (50, 1))
gammas0 = []
gammas1 = []
# gammas2 = []

sys   = System(n_ghost=1, resR=400, r_max=2*np.pi, resZ=400, z_max=2*np.pi, D_eta=0, D_H=0.05, D_P=0, B_Z0=0)
equ_a = AnalyticalEquilibriumCyl(sys, p_exp=4)
lin = LinearCyl(equ_a, k=1, m=0, rosh=2)
for k in k_vals:
    print(k)
    lin.set_z_mode(k, m=0)
    lin.solve(num_modes=None)
    gammas0.append(lin.remove_baddies())

sys   = System(n_ghost=1, resR=400, r_max=2*np.pi, resZ=400, z_max=2*np.pi, D_eta=0, D_H=0.05, D_P=0, B_Z0=1)
equ_a = AnalyticalEquilibriumCyl(sys, p_exp=4)
lin = LinearCyl(equ_a, k=1, m=0, rosh=2)
for k in k_vals:
    print(k)
    lin.set_z_mode(k, m=0)
    lin.solve(num_modes=None)
    gammas0.append(lin.remove_baddies())
   
# sys   = System(n_ghost=1, resR=400, r_max=2*np.pi, resZ=400, z_max=2*np.pi, D_eta=0, D_H=0.05, D_P=0, B_Z0=1)
# equ_a = AnalyticalEquilibriumCyl(sys, p_exp=4)
# lin = LinearCyl(equ_a, k=1, m=0, rosh=2)
# for k in k_vals:
#     print(k)
#     lin.set_z_mode(k, m=0)
#     lin.solve(num_modes=None)
#     gammas0.append(lin.remove_baddies())
    
plt.plot(k_vals, np.square(gammas0), k_vals, np.square(gammas1))
plt.title('Fastest growing mode, D_H = 0.05')
plt.xlabel('k')
plt.ylabel('gamma^2')
plt.legend(['B_Z0 = 0', 'B_Z0 = 1'])
plt.show()