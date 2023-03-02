from .common import *

w_ν = .2 # our MCMC will be bounded in a box this wide around the guess m = 0 frequency
A_min = .01
A_max = 4
δν_min = 0.05
δν_max = 0.13
Γ_min = 0.01
Γ_max = 0.4

# Dimensions for MCMC

N_shared = 5 # 5 shared parameters: inclination, Γ, granulation (x2) and background
shared_quantities = ['i', 'Γ', 'BG1', 'BG2', 'BG3']

s_ν = slice(N_shared, N_total + N_shared)
s_A = slice(N_total + N_shared, 2*N_total + N_shared)
s_Γ = slice(None, 0)
s_δν = slice(2*N_total + N_shared, 2*N_total + N_shared + N_nonradial)

slices = {'ν': s_ν, 'A': s_A, 'δν': s_δν}

Ndims = 2*N_total + N_shared + N_nonradial
checkpoint = "checkpoints/pooled_lw.npy"

# which indices describe a mode?

def get_indices(i, shared=False):
    i_ν = N_shared + i
    i_A = N_shared + N_total + i
    if int(ll_guess[i]) == 0:
        res = [i_ν, i_A]
    else:
        i_δν = N_shared + 2 * N_total + i - N_radial
        res = [i_ν, i_A, i_δν]
    return [*([0, 1, 2] if shared else []), *res]

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
    
    ν0 = nu_guess - w_ν/2 + w_ν * x[s_ν]
    A = jnp.exp(jnp.log(A_min) + jnp.log(A_max / A_min) * x[s_A])
    δν = δν_min + (δν_max - δν_min) * x[s_δν]
    
    i = jnp.arccos(μ)
    
    return jnp.array([i, Γ, BG1, BG2, BG3, *ν0, *A, *δν])

@partial(jax.jit, static_argnums=(2,3))
def chunk_model(ν, θ, only_l=-1, only_chunk=None):
    i0 = θ[0]
    Γ = θ[1]
    BG1 = θ[2]
    BG2 = θ[3]
    BG3 = θ[4]

    ν0 = θ[s_ν]
    A = θ[s_A]
    δν = θ[s_δν]

    BG = Lorentzian(ν, 0, BG1, BG2) + BG3
    
    ps = 0 * ν + BG
    
    for i in range(len(ν0)):
        l = ll_guess[i]
        if only_l>=0 and l != only_l:
            continue
        if only_chunk is not None and chunks[i] != only_chunk:
            continue
        if l == 0:
            ps += modes[l](ν, ν0[i], A[i], Γ, 0, 0)
        else:
            ps += modes[l](ν, ν0[i], A[i], Γ, δν[i - N_radial], i0)
    return ps