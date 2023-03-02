import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

plt.style.use('tableau-colorblind10')

from scipy.ndimage import gaussian_filter1d, gaussian_filter

def echelle_power(ν, PS, Δν, nmesh=1000, nmin=3, nmax=10):
    νmodΔν = np.linspace(0, Δν, nmesh)
#     smoothed = gaussian_filter1d(PS, nblur)
    ν0 = np.arange(nmin, nmax) * Δν
    mesh = ν0[:,None] + νmodΔν[None, :]
    p = np.interp(mesh, ν, PS)
    return νmodΔν, ν0, mesh, p

def echelle_power_plot(ν, PS, Δν, nmesh=1000, nblur=(0, 1), nmin=3, nmax=10, **kwargs):
    νmodΔν, ν0, mesh, p = echelle_power(ν, PS, Δν, nmesh=nmesh, nmin=nmin, nmax=nmax)
    plt.pcolormesh(νmodΔν, ν0, gaussian_filter(p, nblur), **kwargs)

from .mode_id import *