from ..data import *
from ..mode_id import *

from functools import partial

import jax
import jax.numpy as jnp

import dynesty

w_ν = .4 

# break power spectrum down in to chunks associated with w_ν

bounds = []
chunks = {}
idxen = np.argsort(nu_guess)
for i in idxen:
    nu = nu_guess[i]
    for j, (low, high) in enumerate(bounds):
        if nu - w_ν/2 < high:
            bounds[j] = (low, nu + w_ν/2)
            chunks[i] = j
            break
    if i not in chunks:
        chunks[i] = len(bounds)
        bounds.append((nu - w_ν/2, nu + w_ν/2))
        
chunk_ν = [ν[(ν >= low) & (ν <= high)] for low, high in bounds]
chunk_PS = [PS[(ν >= low) & (ν <= high)] for low, high in bounds]

@jax.jit
def Lorentzian(ν, ν0, A, Γ):
    '''
    Lorentzian: nothing to see here
    '''
    return A / (1 + (ν - ν0)**2 / (Γ/2)**2)

@jax.jit
def modes_0(ν, ν0, A, Γ, ν_s, i):
    return Lorentzian(ν, ν0, A, Γ)

@jax.jit
def modes_1(ν, ν0, A, Γ, ν_s, i):
    return (
        Lorentzian(ν, ν0, A, Γ) * jnp.cos(i)**2
        + Lorentzian(ν, ν0 + ν_s, A, Γ) * jnp.sin(i)**2/2
        + Lorentzian(ν, ν0 - ν_s, A, Γ) * jnp.sin(i)**2/2
    )

@jax.jit
def modes_2(ν, ν0, A, Γ, ν_s, i):
    return (
        Lorentzian(ν, ν0, A, Γ) * (3 *jnp.cos(i)**2 - 1)**2 / 4
        + Lorentzian(ν, ν0 + ν_s, A, Γ) * jnp.sin(2*i)**2 * 3/8
        + Lorentzian(ν, ν0 - ν_s, A, Γ) * jnp.sin(2*i)**2 * 3/8
        + Lorentzian(ν, ν0 + 2*ν_s, A, Γ) * jnp.sin(i)**4 * 3/8
        + Lorentzian(ν, ν0 - 2*ν_s, A, Γ) * jnp.sin(i)**4 * 3/8
    )

modes = {0: modes_0, 1: modes_1, 2: modes_2}

@jax.jit
def ln_like_chi2_2dof(model, obs, soften=1):
    '''
    Set soften to the oversampling factor
    '''
    return -jnp.sum(jnp.log(model / obs) + obs / model - 1) / soften

@jax.jit
def model(θ):
    return [chunk_model(ν, θ, only_chunk=chunk) for chunk, ν in enumerate(chunk_ν)]

def summarise(sampler):
    DYresult = sampler.results
    samples = DYresult.samples
    weights = np.exp(DYresult.logwt - DYresult.logz[-1])
    mean, cov = dynesty.utils.mean_and_cov(samples, weights)
    new_samples = dynesty.utils.resample_equal(samples, weights)

    return {
        'mean': mean,
        'cov': cov,
        'new_samples': new_samples,
        'results': sampler.results,
        'weights': weights
    }

shared_labels = {'i': r'$i$/rad', 'Γ': r'$\Gamma/\mu$Hz', 'BG1': 'Granulation amplitude/ppm',
                 'BG2': r'Granulation $\Gamma/\mu$Hz', 'BG3': 'Background/ppm',
                 'δν1': r'$\delta\nu_{\text{rot}, 1}$', 'δν2': r'$\delta\nu_{\text{rot}, 2}$'
                }
mode_labels = {'ν': r"$\nu_{{{}}}/\mu$Hz", 'A': r"$A_{{{}}}$/ppm", 'Γ': r'$\Gamma_{{{}}}/\mu$Hz', 'δν': r"$\delta\nu_{{rot,{}}}/\mu$Hz"}
quantity_labels = {'ν': r'$\nu/\mu$Hz', 'A': 'Amplitude/ppm', 'Γ': r'$\Gamma/\mu$Hz', 'δν': r'$\delta\nu_\text{rot}$'}