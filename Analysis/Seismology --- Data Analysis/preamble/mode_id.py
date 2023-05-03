import numpy as np
from .mode_id_2 import *

N_total = len(nu_guess)
N_nonradial = sum(ll_guess != 0)
N_radial = N_total - N_nonradial

Δν = np.polyfit(np.arange(N_radial), nu_guess[ll_guess == 0], 1)[0]