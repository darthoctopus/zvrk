from .common import *

w_ν = .2 # our MCMC will be bounded in a box this wide around the guess m = 0 frequency
A_min = .01
A_max = 4
δν_min = 0.05
δν_max = 0.14
Γ_min = 0.01
Γ_max = 0.4

# Dimensions for MCMC

N_shared = 6 # 6 shared parameters: inclination, granulation (x2), background, and l=1,2 rotational splitting
shared_quantities = ['i', 'BG1', 'BG2', 'BG3', 'δν1', 'δν2']

s_ν = slice(N_shared, N_total + N_shared)
s_A = slice(N_total + N_shared, 2*N_total + N_shared)
s_Γ = slice(2*N_total + N_shared, 3*N_total + N_shared)
s_δν = slice(None, 0)

slices = {'ν': s_ν, 'A': s_A, 'Γ': s_Γ}

Ndims = 3*N_total + N_shared
checkpoint = "checkpoints/pooled_rot.npy"

# which indices describe a mode?

def get_indices(i, shared=False):
    i_ν = N_shared + i
    i_A = N_shared + N_total + i
    i_Γ = N_shared + 2*N_total + i
    res = [i_ν, i_A, i_Γ]
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
    BG1 = x[1] * 1
    BG2 = x[2] * 5000
    BG3 = jnp.exp(jnp.log(1e-6) + jnp.log(2/1e-6) * x[3])
    δν1 = δν_min + (δν_max - δν_min) * x[4]
    δν2 = δν_min + (δν_max - δν_min) * x[5]
    
    ν0 = nu_guess - w_ν/2 + w_ν * x[s_ν]
    A = jnp.exp(jnp.log(A_min) + jnp.log(A_max / A_min) * x[s_A])
    Γ = Γ_min + (Γ_max - Γ_min) * x[s_Γ]

    i = jnp.arccos(μ)
    
    return jnp.array([i, BG1, BG2, BG3, δν1, δν2, *ν0, *A, *Γ])

@partial(jax.jit, static_argnums=(2,3))
def chunk_model(ν, θ, only_l=-1, only_chunk=None):
    i0 = θ[0]
    BG1 = θ[1]
    BG2 = θ[2]
    BG3 = θ[3]
    δν1 = θ[4]
    δν2 = θ[5]

    ν0 = θ[s_ν]
    A = θ[s_A]
    Γ = θ[s_Γ]

    BG = Lorentzian(ν, 0, BG1, BG2) + BG3

    ps = 0 * ν + BG

    for i in range(len(ν0)):
        l = ll_guess[i]
        if only_l>=0 and l != only_l:
            continue
        if only_chunk is not None and chunks[i] != only_chunk:
            continue
        if l == 0:
            ps += modes[l](ν, ν0[i], A[i], Γ[i], 0, 0)
        elif l == 1:
            ps += modes[l](ν, ν0[i], A[i], Γ[i], δν1, i0)
        elif l == 2:
            ps += modes[l](ν, ν0[i], A[i], Γ[i], δν2, i0)

    return ps