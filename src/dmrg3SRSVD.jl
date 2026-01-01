function dmrg3SRSVD(
    H,
    psi0::MPS;
    nsweeps,
    maxdim=ITensorMPS.default_maxdim(),
    mindim=ITensorMPS.default_mindim(),
    cutoff=ITensorMPS.default_cutoff(Float64),
    noise=ITensorMPS.default_noise(),
    kwargs...,
)
    H = permute(H, (linkind, siteinds, linkind))
    PH = ProjMPO(0, length(H) + 1, 1, H, Vector{ITensor}(undef, length(H)))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim...)
    setmindim!(sweeps, mindim...)
    setcutoff!(sweeps, cutoff...)
    setnoise!(sweeps, noise...)

    return dmrg3SRSVD(PH, psi0, sweeps; kwargs...)
end
function dmrg3SRSVD(
    PH,
    psi0::MPS,
    sweeps::Sweeps;
    which_decomp=nothing,
    svd_alg=nothing,
    observer=NoObserver(),
    outputlevel=1,
    write_when_maxdim_exceeds=nothing,
    write_path=tempdir(),
    # eigsolve kwargs
    eigsolve_tol=1.0e-14,
    eigsolve_krylovdim=3,
    eigsolve_maxiter=1,
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:SR,
    ishermitian=true,
    # rsvd
    rsvd_qn_min_dim=2,
    rsvd_power_iteration=0,
    # expansion
    alpha=2e-2,
    alpha_min=1e-8,
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
            PH = disk(PH; path=write_path)
        end
    end
    PH = position!(PH, psi, 1)
    energy_0 = 0.0
    energy = 0.0
    energy_i = 0.0
    spec = nothing
    timer = TimerOutput()
    Nexp = 0
    alg = "global_krylov"
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
                PH = disk(PH; path=write_path)
            end
            #from left to right
            left_to_right = true
            for (b, ha) in sweepnext(N, ncenter=1) #single site 
                PH = position!(PH, psi, b)
                phi = psi[b]
                if b == N && left_to_right
                    @goto Boundary
                end
                if b == 1 && !left_to_right
                    @goto Boundary
                end
                @timeit timer "dmrg: eigsolve" begin
                    vals, vecs = eigsolve(
                        PH,
                        phi,
                        1,
                        eigsolve_which_eigenvalue;
                        ishermitian,
                        tol=eigsolve_tol,
                        krylovdim=eigsolve_krylovdim,
                        maxiter=eigsolve_maxiter,
                        verbosity=eigsolve_verbosity,
                    )
                end

                energy = vals[1]
                phi = vecs[1]

                poi = b - 1
                if left_to_right
                    poi = b + 1
                end
                A = phi
                B = psi[poi]
                @timeit timer "dmrg: 1-site-SVD" begin
                    rinds = uniqueinds(A, B)
                    ltags = tags(commonind(A, B))
                    Ua, S, V, spec = svd(A, rinds; lefttags=ltags,
                        maxdim=maxdim(sweeps, sw),
                        mindim=mindim(sweeps, sw),
                    )
                    # @show commoninds(Ua,phi)
                    psi[b] = Ua*S
                    psi[poi] =  V * B
                end
                maxtruncerr = max(maxtruncerr, spec.truncerr)
                phi = psi[b]
                if maxtruncerr > alpha_min
                    alpha = maxtruncerr #max(1e-4, maxtruncerr)
                end
                @timeit timer "dmrg: Expansion" begin
                    #a different way for expansion
                    #using random matrix
                    LR = nothing
                    W = PH.H[b]
                    com_ind = commonind(psi[poi], phi)
                    w_ind = commonind(PH.H[poi], W)

                    ind_c = combinedind(combiner(w_ind, com_ind))
                    dim_all = dim(ind_c)
                    dim_phi = dim(com_ind)
                    target_dim = ceil(Int, min(dim_phi, 0.1maxdim(sweeps, sw)))
                    ratio = target_dim / dim_all
                    new_idx = Pair{QN,Int64}[]
                    for sp in ind_c.space
                        push!(new_idx, sp[1] => max(ceil(Int, sp[2] * ratio),rsvd_qn_min_dim))
                    end
                    ind_cnew = Index(new_idx)
                    if dir(ind_cnew) == dir(ind_c)
                        ind_cnew = dag(ind_cnew)
                    end
                    @timeit timer "Exp: RandomTensor" rand_ten = random_itensor(ind_cnew, w_ind, com_ind)
                    LR = nothing
                    if left_to_right
                        LR = lproj(PH)
                    else
                        LR = rproj(PH)
                    end
                    @timeit timer "Exp: Tensor" M = (LR * phi) * W
                    noprime!(M)
                    @timeit timer "Exp: Tensor" M = M - (M * dag(Ua)) * Ua
                    @timeit timer "Exp: Tensor" P = M * rand_ten
                    if b == 1 || b == N
                        cR = commonind(P, W)
                    else
                        cR = [commonind(P, LR), commonind(P, W)]
                    end
                    @timeit timer "Exp: QR" Q, _ = qr(P, cR)
                    for it_qr in 1:rsvd_power_iteration
                        @timeit timer "Exp: Tensor" Q = (dag(M)*Q)*M
                        @timeit timer "Exp: QR" Q, _ = qr(Q, cR)
                    end
                    qr_ind = uniqueind(Q, P)
                    @timeit timer "Exp: Tensor" MQ = M * dag(Q)
                    @timeit timer "Exp: svd" U, _= factorize(MQ, dag(qr_ind),maxdim=min(dim(ind_cnew)-10,target_dim),
                    ortho="right",which_decomp="svd")
                    @timeit timer "Exp: Tensor" P = Q * U * alpha
                    noprime!(P)
                    P_phi_com_idx = commonind(phi, psi[poi])
                    #then expand the phi and psi[poi] tensor
                    exp_indx = uniqueind(P, phi)
                    #then expand the phi
                    @timeit timer "Exp: DirectSum" A, sA = directsum(P => exp_indx, phi => P_phi_com_idx)

                    #now, expand the next tensor with zero tensors
                    out_idx = uniqueinds(psi[poi], phi)
                    com_idx = commonind(psi[poi], phi)
                    zero_ten = ITensor(dag(exp_indx), out_idx)
                    @timeit timer "Exp: DirectSum" B, sB = directsum(zero_ten => dag(exp_indx), psi[poi] => com_idx)
                    replaceind!(B, sB, sA)
                end
                @timeit timer "dmrg: SVD" begin
                    rinds = uniqueinds(A, B)
                    ltags = tags(commonind(A, B))
                    U, V = factorize(A, rinds; tags=ltags,ortho="left",which_decomp="svd")
                    psi[b] = U
                    psi[poi] = V*B
                end
                @label next
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
                @label Boundary
                if b == N && left_to_right
                    left_to_right = false
                end
                if b == 1 && !left_to_right
                    left_to_right = true
                end
                sweep_is_done = (b == 1 && ha == 2)
                measure!(
                    observer;
                    energy,
                    psi,
                    projected_operator=PH,
                    bond=b,
                    sweep=sw,
                    half_sweep=ha,
                    spec,
                    outputlevel,
                    sweep_is_done,
                )
            end
        end
        if outputlevel >= 1
            @printf(
                "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E mixer=%.2E time=%.3f\n",
                sw,
                energy,
                maxlinkdim(psi),
                maxtruncerr,
                alpha,
                sw_time
            )
            flush(stdout)
        end
        isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
        isdone && break
    end
    show(timer)
    return (energy, psi)
end
