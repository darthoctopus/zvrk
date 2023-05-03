import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nu_guess = np.ones(15, dtype=float)
δν_guess = np.ones(15, dtype=float)
ll_guess = np.ones(15, dtype=int)

nu_guess[0] = 5.817365414861656
nu_guess[1] = 6.910838497833987
nu_guess[2] = 8.2
nu_guess[3] = 9.41
nu_guess[4] = 10.65
nu_guess[5] = 5.34392788541509
nu_guess[6] = 6.405412326669864
nu_guess[7] = 7.64
nu_guess[8] = 8.78
nu_guess[9] = 10.08
nu_guess[10] = 5.636
nu_guess[11] = 6.70
nu_guess[12] = 7.9
nu_guess[13] = 9.17
nu_guess[14] = 10.44

δν_guess[0] = 0.0
δν_guess[1] = 0.0
δν_guess[2] = 0.0
δν_guess[3] = 0.0
δν_guess[4] = 0.0
δν_guess[5] = 0.07
δν_guess[6] = 0.08611506150286295
δν_guess[7] = 0.0819886417175576
δν_guess[8] = 0.11
δν_guess[9] = 0.11
δν_guess[10] = 0.09
δν_guess[11] = 0.11
δν_guess[12] = 0.09
δν_guess[13] = 0.11
δν_guess[14] = 0.097

ll_guess[0] = 0
ll_guess[1] = 0
ll_guess[2] = 0
ll_guess[3] = 0
ll_guess[4] = 0
ll_guess[5] = 1
ll_guess[6] = 1
ll_guess[7] = 1
ll_guess[8] = 1
ll_guess[9] = 1
ll_guess[10] = 2
ll_guess[11] = 2
ll_guess[12] = 2
ll_guess[13] = 2
ll_guess[14] = 2
