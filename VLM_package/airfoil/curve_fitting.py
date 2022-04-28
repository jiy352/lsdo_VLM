import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr

data = np.loadtxt('xf-a18-il-200000.txt')
Re = 200000
Mach = 0.000

alpha = data[:, 0]
CL = data[:, 1]
CD = data[:, 2]

# plt.plot(alpha, CL)
# plt.plot(alpha, CD)
# plt.show()


def aoa_Cl(x, a, b):
    return a * x + b


def cd_aoa(x, a, b, c):
    return a * x**2 + b * x + c


# aoa_Cl
x_data = CL[:-2]
y_data = alpha[:-2]
popt, pcov = curve_fit(aoa_Cl, x_data, y_data)
np.savetxt('cl_aoa_coeff.txt', popt)
plt.plot(x_data,
         aoa_Cl(x_data, *popt),
         'r-',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.plot(CL[:-2], alpha[:-2])
plt.show()

# aoa_Cd

x_data = alpha[:-2]
y_data = CD[:-2]
popt, pcov = curve_fit(cd_aoa, x_data, y_data)
np.savetxt('cd_aoa_coeff.txt', popt)
plt.plot(x_data,
         cd_aoa(x_data, *popt),
         'r-',
         label='fit: a=%5.3f, b=%5.3f,c=%5.3f' % tuple(popt))
plt.plot(alpha[:-2], CD[:-2])
plt.show()
