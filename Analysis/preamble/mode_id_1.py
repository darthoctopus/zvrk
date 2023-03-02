import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nu_guess = np.ones(14, dtype=float)
δν_guess = np.ones(14, dtype=float)
ll_guess = np.ones(14, dtype=int)

nu_guess[0] = 5.817365414861656
nu_guess[1] = 6.910838497833987
nu_guess[2] = 8.176932321450401
nu_guess[3] = 9.416520465925595
nu_guess[4] = 10.666982190615483
nu_guess[5] = 5.34392788541509
nu_guess[6] = 6.405412326669864
nu_guess[7] = 7.627379730500973
nu_guess[8] = 8.811967874976165
nu_guess[9] = 10.108544661168917
nu_guess[10] = 5.762997513788182
nu_guess[11] = 6.844355535257651
nu_guess[12] = 8.09557577865937
nu_guess[13] = 9.356520465925595

δν_guess[0] = 0.0
δν_guess[1] = 0.0
δν_guess[2] = 0.0
δν_guess[3] = 0.0
δν_guess[4] = 0.0
δν_guess[5] = 0.07
δν_guess[6] = 0.08611506150286295
δν_guess[7] = 0.0819886417175576
δν_guess[8] = 0.10329901203959935
δν_guess[9] = 0.11873580214694645
δν_guess[10] = 0.05
δν_guess[11] = 0.06683345655653651
δν_guess[12] = 0.095
δν_guess[13] = 0.09

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