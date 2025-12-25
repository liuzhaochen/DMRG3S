using ITensors, ITensorMPS
using LinearAlgebra
using Printf
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
    psi0 = random_mps(sites, state; linkdims = 10)

    # Plan to do 5 DMRG sweeps:
    nsweeps = 20
    # Set maximum MPS bond dimensions for each sweep
    maxdim = [10, 20, 30, 40, 50,100,200]
    # Set maximum truncation error allowed when adapting bond dimensions
    cutoff = [0.0]

    # Run the DMRG algorithm, returning energy and optimized MPS
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    # H = permute(H, (linkind, siteinds, linkind))
    # PH = ProjMPO(0,length(H)+1, 1, H, Vector{ITensor}(undef, length(H)))
    # #move PH
    # PH = position!(PH, psi, 2)
    # #then check LWM
    # W = PH.H[2]
    # LR = lproj(PH)
    # P = LR * W * psi[2]
    # com_idx= commoninds(P,psi[2])
    # P = noprime(P)
    # uni= uniqueinds(P,psi[2])

    # C = combiner(com_idx[1],uni[1]; tags="c")
    # P = P * C
    # uni = uniqueind(P, psi[2])
    # psi_new, sa = directsum(P=>uni, psi[2]=>com_idx[1])
    # # @show inds(psi_new),s
    # #now construct a zero element tensor
    # in_idx = dag(uni) #in-index
    # uni_idx = uniqueinds(psi[3],psi[2]) #out-index
    # com_idx = commonind(psi[3],psi[2])
    # #construct zero tensor
    # zero_ten = ITensor(in_idx,uni_idx)
    # psi_B , sb= directsum(zero_ten=>in_idx, psi[3]=>com_idx)
    # @show sb, sa
    # replaceind!(psi_B, sb, dag(sa))
    # @show inds(psi_new)
    # @show inds(psi_B)
    # @show commoninds(psi_new,psi_B)    
    # @show inds(psi_new*psi_B)
    # # @show zero_ten
    # # @show inds(psi[2])#,psi[3])
    # # @show uni[1],dag(uni[1])
    return nothing
end
