using ITensors
using ITensorMPS
using Printf: @printf
using KrylovKit: eigsolve

function dmrg3S(
        H,
        psi0::MPS;
        nsweeps,
        maxdim = ITensorMPS.default_maxdim(),
        mindim = ITensorMPS.default_mindim(),
        cutoff = ITensorMPS.default_cutoff(Float64),
        noise = ITensorMPS.default_noise(),
        kwargs...,
    )
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjMPO(0,length(H)+1, 1, H, Vector{ITensor}(undef, length(H)))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim...)
    setmindim!(sweeps, mindim...)
    setcutoff!(sweeps, cutoff...)
    setnoise!(sweeps, noise...)    

    return dmrg3S(PH, psi0, sweeps; kwargs...)
end
function dmrg3S(
        PH,
        psi0::MPS,
        sweeps::Sweeps;
        which_decomp = nothing,
        svd_alg = nothing,
        observer = NoObserver(),
        outputlevel = 1,
        write_when_maxdim_exceeds = nothing,
        write_path = tempdir(),
        # eigsolve kwargs
        eigsolve_tol = 1.0e-14,
        eigsolve_krylovdim = 3,
        eigsolve_maxiter = 1,
        eigsolve_verbosity = 0,
        eigsolve_which_eigenvalue = :SR,
        ishermitian = true,
        alpha = 0.1,
    )
    psi = copy(psi0)
    N = length(psi)
    if !isortho(psi) || orthocenter(psi) != 1
        psi = orthogonalize!(PH, psi, 1)
    end
    @assert isortho(psi) && orthocenter(psi) == 1

    if !isnothing(write_when_maxdim_exceeds)
        if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
                (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
            PH = disk(PH; path = write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy_0 =0.0
    energy = 0.0
    energy_i = 0.0
    for sw in 1:nsweep(sweeps)
        sw_time = @elapsed begin
            maxtruncerr = 0.0

            if !isnothing(write_when_maxdim_exceeds) &&
                    maxdim(sweeps, sw) > write_when_maxdim_exceeds
                if outputlevel >= 2
                    println(
                        "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
                    )
                end
                PH = disk(PH; path = write_path)
            end
            #from left to right
            left_to_right = true
            for (b,ha) in sweepnext(N, ncenter=1) #single site 
                PH = position!(PH, psi, b)
                phi = psi[b]
                if b==N && left_to_right
                    @goto Boundary
                end
                if b==1 && !left_to_right
                    @goto Boundary
                end
                vals, vecs = eigsolve(
                        PH,
                        phi,
                        1,
                        eigsolve_which_eigenvalue;
                        ishermitian,
                        tol = eigsolve_tol,
                        krylovdim = eigsolve_krylovdim,
                        maxiter = eigsolve_maxiter,
                        verbosity = eigsolve_verbosity,
                    )

                eig_0 = (PH(psi[b])*dag(psi[b]))[]
                phi = vecs[1]
                if sw == 1
                    energy_0 = eig_0
                else
                    #turn the alpha
                    delta_E0 = energy-energy_0
                    delta_dE = eig_0 - energy
                    # @show delta_E0,delta_dE,alpha
                    energy_0 = eig_0
                    if abs(delta_dE)<1e-8 || delta_dE<0
                        # alpha *= 1.02
                    end
                    if abs(abs(delta_dE)-abs(delta_E0))<1e-8
                        # alpha /=1.02
                    end
                    # alpha = max(alpha, 0.001)
                end
                energy = vals[1]
                W = PH.H[b]
                LR = nothing
                if left_to_right
                    LR = lproj(PH)
                else
                    LR = rproj(PH)
                end
                #P=alpha*L*M*W
                P = LR * W
                #energy before optimization
                P = alpha* P * phi
                #the common inds between expander P and phi
                P_phi_com_idx = commonind(P, phi)
                #remove prime
                noprime!(P)
                #the mpo index
                mpo_idx = uniqueind(P, phi) 
                combiner_tensor = combiner(P_phi_com_idx, mpo_idx)
                P = P * combiner_tensor #combine vitural and mpo_idx
                exp_indx = uniqueind(P, phi)
                #then expand the phi
                A , sA= directsum(P=>exp_indx, phi=>P_phi_com_idx)

                #now, expand the next tensor with zero tensors
                #reverse the direction of exp_indx
                exp_indx = dag(exp_indx)
                B = nothing
                if left_to_right
                    #need to check the bond indices
                    out_idx = uniqueinds(psi[b+1], phi)
                    com_idx = commonind(psi[b+1], phi)
                    zero_ten = ITensor(exp_indx, out_idx)
                    B ,sB= directsum(zero_ten=>exp_indx, psi[b+1]=>com_idx)
                else
                    out_idx = uniqueinds(psi[b-1], phi)
                    com_idx = commonind(psi[b-1], phi)
                    zero_ten = ITensor(exp_indx, out_idx)
                    B ,sB= directsum(zero_ten=>exp_indx, psi[b-1]=>com_idx)
                end
                replaceind!(B, sB, sA)
                ortho = ha == 1 ? "left" : "right"

                drho = nothing
                rinds = uniqueinds(A, B)
                ltags = tags(commonind(A,B))
                if left_to_right
                    U, S, V, spec = svd(A, rinds; righttags = ltags, 
                        maxdim = maxdim(sweeps, sw),
                        mindim = mindim(sweeps, sw),
                        cutoff = cutoff(sweeps, sw)
                    )                    
                    psi[b] = U 
                    psi[b+1]= S * V * B
                else
                    U, S, V, spec = svd(A, rinds; righttags = ltags, 
                        maxdim = maxdim(sweeps, sw),
                        mindim = mindim(sweeps, sw),
                        cutoff = cutoff(sweeps, sw)
                    )                    
                    psi[b] = U 
                    psi[b-1]= S*V*B
                end
                 
                maxtruncerr = max(maxtruncerr, spec.truncerr)


                if outputlevel >= 2
                    @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
                    @printf(
                        "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
                        cutoff(sweeps, sw),
                        maxdim(sweeps, sw),
                        mindim(sweeps, sw)
                    )
                    @printf(
                        "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
                    )
                    flush(stdout)
                end

                sweep_is_done = (b == 1 && ha == 2)
                measure!(
                    observer;
                    energy,
                    psi,
                    projected_operator = PH,
                    bond = b,
                    sweep = sw,
                    half_sweep = ha,
                    spec,
                    outputlevel,
                    sweep_is_done,
                )
                @label Boundary
                if b== N && left_to_right
                    left_to_right = false
                end
                if b== 1 && !left_to_right
                    left_to_right = true
                end
            end
        end
        if outputlevel >= 1
            @printf(
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                sw_time
            )
            flush(stdout)
        end
        isdone = checkdone!(observer; energy, psi, sweep = sw, outputlevel)
        isdone && break
    end
    return (energy, psi)
end
