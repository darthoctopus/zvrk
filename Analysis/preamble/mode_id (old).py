import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ν, PS = np.loadtxt("diamonds_peakbagging/DIAMONDS/data/TIC350842552.txt").T
PS *= 1e6

prelim_id = pd.read_pickle("Seed/TIC350842552.pkl")

ll = prelim_id.ell.values
νν = prelim_id.nu_mad.values

ε_p = 0.75 # rough estimate from Yu et al.
Δν = np.median(np.diff(νν[ll == 0]))

n_p = np.round(νν / Δν - ll/2 / ε_p)

acc_ν = {}
acc_δν = {}
for l in (0, 1, 2):
    acc_ν[l] = []
    acc_δν[l] = []
    nn = np.unique(n_p[ll == l])
    for n in nn:
        nu = νν[(n_p == n) & (ll == l)]
        acc_ν[l].append(nu[len(nu) // 2])
        acc_δν[l].append((nu[-1] - nu[0]) / (2 * (len(nu) // 2)))
        
nu_guess = np.concatenate([acc_ν[l] for l in (0, 1, 2)])
δν_guess = np.concatenate([np.nan_to_num(acc_δν[l], np.nanmean(acc_δν[l])) for l in (0, 1, 2)])
ll_guess = np.concatenate([np.ones_like(acc_ν[l], dtype=int) * l for l in (0, 1, 2)])

# manual adjustment
nu_guess[5] -= .07
nu_guess[8] -= .05
nu_guess[9] -= .08
nu_guess[12] += .06
nu_guess = np.array([*nu_guess, nu_guess[3]-.05])

δν_guess[5] = 0.07
δν_guess[10] = 0.05
δν_guess[12] = 0.095
δν_guess = np.array([*δν_guess, .09])

ll_guess = np.array([*ll_guess, 2])

N_total = len(nu_guess)
N_nonradial = sum(ll_guess != 0)
N_radial = N_total - N_nonradial