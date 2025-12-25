
using ITensors, ITensorMPS
using LinearAlgebra
using Printf
include("../src/main.jl")
let
    N = 20

    sites = siteinds("S=1/2", N; conserve_qns = true)

    os = OpSum()
    for j in 1:(N - 1)
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
        os += "Sz", j, "Sz", j + 1
    end
    for j in 1:(N - 2)
        os += 0.5, "S+", j, "S-", j + 2
        os += 0.5, "S-", j, "S+", j + 2
        os += "Sz", j, "Sz", j + 2
    end
    H = MPO(os, sites)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = random_mps(sites, state; linkdims = 1)

    # Plan to do 5 DMRG sweeps:
    nsweeps = 50
    # Set maximum MPS bond dimensions for each sweep
    mindim = [1,2]
    maxdim = [10, 20,40,80,100,200]
    # Set maximum truncation error allowed when adapting bond dimensions
    cutoff = [0.0]
    noise = [1E-10,0.0]
    # sweeps = _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
    
    # Run the DMRG algorithm, returning energy and optimized MPS
    # energy, psi = dmrg(H, psi; nsweeps, maxdim, cutoff)
    eng,psi=dmrg3S(H, psi0,; nsweeps, maxdim, cutoff, alpha=1e-4)
    return nothing
end
