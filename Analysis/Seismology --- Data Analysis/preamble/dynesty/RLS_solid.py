from .common import *
from ..kernels import *

from astropy import units as u
SURF_RATE = (1/(98 * u.d)).to(u.uHz).value

w_ν = .2 # our MCMC will be bounded in a box this wide around the guess m = 0 frequency
A_min = .01
A_max = 4
# δν_min = 0.05
# δν_max = 0.14

δν_min = SURF_RATE - .05
δν_max = SURF_RATE + .05

Γ_min = 0.01
Γ_max = 0.4

# Dimensions for MCMC

N_shared = 6 # 6 shared parameters: inclination, Γ, granulation (x2), background, and RLS parameters
shared_quantities = ['i', 'Γ', 'BG1', 'BG2', 'BG3', 'ν_rot']

s_ν = slice(N_shared, N_total + N_shared)
s_A = slice(N_total + N_shared, 2*N_total + N_shared)
s_Γ = slice(None, 0)
s_δν = slice(None, 0)

slices = {'ν': s_ν, 'A': s_A}

Ndims = 2*N_total + N_shared
checkpoint = "checkpoints/RLS_solid.npy"

# which indices describe a mode?

def get_indices(i, shared=False):
    i_ν = N_shared + i
    i_A = N_shared + N_total + i
    res = [i_ν, i_A]
    return [*([0, 1, 2] if shared else []), *res]

@jax.jit
def trapz(y, x):
    return jnp.sum(1/2 * (y[1:] + y[:-1]) * jnp.diff(x))

@jax.jit
def prior_transform(x):
    '''
    x takes values between 0 and 1, and we want to turn these into
    our science coordinates.
    '''
    
    # We will treat all modes as having the same inclination and linewidth,
    # since by assumption we are dealing with only the most p-dominated modes.
    
    μ = x[0]
    Γ = Γ_min + (Γ_max - Γ_min) * x[1]
    BG1 = x[2] * 0.5
    BG2 = x[3] * 16
    BG3 = jnp.exp(jnp.log(1e-6) + jnp.log(1.5e-3/1e-6) * x[4])
    ν_rot = δν_min + (δν_max - δν_min) * x[5]
    
    ν0 = nu_guess - w_ν/2 + w_ν * x[s_ν]
    A = jnp.exp(jnp.log(A_min) + jnp.log(A_max / A_min) * x[s_A])

    i = jnp.arccos(μ)
    
    return jnp.array([i, Γ, BG1, BG2, BG3, ν_rot, *ν0, *A])

def rotational_profile(t, ν_rot):
    # if Δt_shear != 0:
        return ν_rot
    # return np.where(t < t_shear, ν0 - δν/2, ν0 + δν/2)

def split(ν_rot, l, n):
    return trapz(
            KK[l][n] * rotational_profile(t[l], ν_rot),
            eig[l].r
        )

@partial(jax.jit, static_argnums=(2,3))
def chunk_model(ν, θ, only_l=-1, only_chunk=None):
    i0 = θ[0]
    Γ = θ[1]
    BG1 = θ[2]
    BG2 = θ[3]
    BG3 = θ[4]
    ν_rot = θ[5]

    ν0 = θ[s_ν]
    A = θ[s_A]

    BG = Lorentzian(ν, 0, BG1, BG2) + BG3

    ps = 0 * ν + BG

    for i in range(len(ν0)):
        n = i // 5
        l = ll_guess[i]
        if only_l>=0 and l != only_l:
            continue
        if only_chunk is not None and chunks[i] != only_chunk:
            continue
        if l == 0:
            ps += modes[l](ν, ν0[i], A[i], Γ, 0, 0)
        else:
            ps += modes[l](ν, ν0[i], A[i], Γ, split(ν_rot, l, n), i0)

    return ps