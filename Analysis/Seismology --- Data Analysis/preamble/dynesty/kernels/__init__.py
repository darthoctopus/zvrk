import numpy as np
from tqdm.auto import tqdm
from scipy.integrate import cumtrapz
from os.path import isfile
from mesa_tricks.io.eigensystem import Eigensystem

def rot_kernel(ξr1, ξh1, ξr2, ξh2, l, m):
    Λ2 = l * (l + 1)
    return 2*m * (ξr1 * ξr2 + (Λ2 - 1) * (ξh1 * ξh2) - ξr1 * ξh2 - ξh1 * ξr2)

def K(eig, i, l):
    M = float(eig._info[0]['M_star'])
    R = float(eig._info[0]['R_star'])
    return 4 * np.pi * eig.r**2 * eig.ρ0 * rot_kernel(eig.ξ[i], eig.ξh[i], eig.ξ[i],
                                                      eig.ξh[i], l, 1) / (M * R**2) / 2
# evaluate actual rotational kernels

eig = {}
KK = {}

eig[1] = Eigensystem("1099_90.data-1/π.*.txt")
KK[1] = np.array([K(eig[1], i, 1) for i in range(2, 7)])

eig[2] = Eigensystem("1099_90.data-2/π.*.txt")
KK[2] = np.array([K(eig[2], i, 2) for i in range(3, 8)])

def get_t(eig):
    cs = np.sqrt(eig.P0 * eig.Γ1 / eig.ρ0)
    return cumtrapz(1/cs, eig.r, initial=0)

t = {}

t[1] = get_t(eig[1])
t[2] = get_t(eig[2])

T = max(t[1][-1], t[2][-1])
t[1] /= T
t[2] /= T