using MKL
using LinearAlgebra
include(joinpath(homedir(),"module","itensor_dmrg","src","main.jl"))
function main(; Nx = 10, Ny = 4, U = 0.0, t = 1.0)
    N = Nx * Ny

    nsweeps = 50
    maxdim = [4000]
    cutoff = [1.0e-12]
    noise = [0.0]

    sites = siteinds("Electron", N; conserve_qns = true)

    lattice = square_lattice(Nx, Ny; yperiodic = true)

    os = OpSum()
    for b in lattice
        os -= t, "Cdagup", b.s1, "Cup", b.s2
        os -= t, "Cdagup", b.s2, "Cup", b.s1
        os -= t, "Cdagdn", b.s1, "Cdn", b.s2
        os -= t, "Cdagdn", b.s2, "Cdn", b.s1
    end
    for n in 1:N
        os += U, "Nupdn", n
    end
    H = MPO(os, sites)

    # Half filling
    state = ["Emp" for n in 1:N]
    for i in 1:Int(0.9*Nx*Ny)
        state[i] = isodd(i) ? "Up" : "Dn"
    end
    # state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

    # Initialize wavefunction to a random MPS
    # of bond-dimension 10 with same quantum
    # numbers as `state`
    psi0 = random_mps(sites, state,linkdims=5)

    energy, psi = dmrg3SRSVD(H, psi0; nsweeps, maxdim, cutoff, noise)
    # energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
    @show t, U
    @show flux(psi)
    @show maxlinkdim(psi)
    @show energy
    Ex = -63.37540927469395
    @show abs(energy/Ex - 1)
    return nothing
end

main()
