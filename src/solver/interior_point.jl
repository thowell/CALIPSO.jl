# residual
function r!(r, z, θ, κ)
    @warn "residual not defined"
    nothing
end

# residual Jacobian wrt z
function rz!(rz, z, θ)
    @warn "residual Jacobian wrt z not defined"
    nothing
end

# residual Jacobian wrt θ
function rθ!(rθ, z, θ)
    @warn "residual Jacobian wrt θ not defined"
    nothing
end

mutable struct ResidualMethods
    r!
    rz!
    rθ!
end

# optimization spaces
abstract type Space end

# Euclidean
struct Euclidean <: Space
    n::Int
end

function candidate_point!(z̄::Vector{T}, ::Euclidean, z::Vector{T}, Δ::Vector{T}, α::T) where T
    z̄ .= z 
    z̄ .-= α .* Δ
    return z̄
end

function update_point!(z::Vector{T}, ::Space, z̄::Vector{T}) where T
    z .= z̄
end

function mapping!(δz, s::Euclidean, δzs, z) # TODO: make allocation free
    δz .= δzs
end

# interior-point solver options
Base.@kwdef mutable struct InteriorPointOptions{T}
    r_tol::T = 1.0e-5
    κ_tol::T = 1.0e-5
    ls_scale::T = 0.5
    max_iter::Int = 100
    max_ls::Int = 3
    max_time::T = 1e5
    diff_sol::Bool = false
    reg::Bool = false
    ϵ_min = 0.05 # ∈ [0.005, 0.25]
        # smaller -> faster
        # larger  -> slower, more robust
    κ_reg = 1e-3 # bilinear constraint violation level at which regularization is triggered [1e-3, 1e-4]
    γ_reg = 1e-1 # regularization scaling parameters ∈ [0, 0.1]:
        # 0   -> faster & ill-conditioned
        # 0.1 -> slower & better-conditioned
        # simulation choose γ_reg = 0.1
        # MPC choose γ_reg = 0.0
    solver::Symbol = :lu_solver
    undercut::T = 2.0 # the solver will aim at reaching κ_vio = κ_tol / undercut
        # simulation: Inf
        # MPC: 5.0
    verbose::Bool = false
    warn::Bool = false
end

mutable struct InteriorPoint{T,R,RZ,Rθ}
    s::Space
    idx::IndicesOptimization
    methods::ResidualMethods
    z::Vector{T}               # current point
    z̄::Vector{T}               # candidate point
    r::R                       # residual
    rz::RZ                     # residual Jacobian wrt z
    rθ::Rθ                     # residual Jacobian wrt θ
    Δ::Vector{T}               # search direction
    δz::Matrix{T}              # solution gradients (this is always dense)
    δzs::Matrix{T}             # solution gradients (in optimization space; δz = δzs for Euclidean)
    θ::Vector{T}               # problem data
    solver::LinearSolver
    z_reg::Vector{T}           # current point w/ regularization on cone variables
    reg_val::Vector{T}
    iterations::Int
    opts::InteriorPointOptions{T}
    κ::Vector{T}
    zort::Vector{Vector{T}}
    Δort::Vector{Vector{T}}
    zsoc::Vector{Vector{Vector{T}}}
    Δsoc::Vector{Vector{Vector{T}}}
    ρv::Vector{Vector{Vector{T}}}
    idx_soce::Vector{Vector{Vector{Int}}}
end

function interior_point(z, θ;
        s = Euclidean(length(z)),
        idx = IndicesOptimization(),
        r! = r!, rz! = rz!, rθ! = rθ!,
        r  = zeros(idx.nΔ),
        rz = spzeros(idx.nΔ, idx.nΔ),
        rθ = spzeros(idx.nΔ, length(θ)),
        opts::InteriorPointOptions = InteriorPointOptions()) where T

    rz!(rz, z, θ) # compute Jacobian for pre-factorization
    num_data = length(θ)

    zort = [zeros(length(idx.ortz[1])), zeros(length(idx.ortz[2]))]
    Δort = [zeros(length(idx.ortΔ[1])), zeros(length(idx.ortΔ[2]))]
    zsoc = [[zeros(length(i)), zeros(length(i))] for i in idx.socri]
    Δsoc = [[zeros(length(i)), zeros(length(i))] for i in idx.socri]
    ρv = [[zeros(max(0, length(i) - 1)), zeros(max(0, length(i) - 1))] for i in idx.socri]
    idx_soce = [[collect(2:length(i)), collect(2:length(i))] for i in idx.socri]

    InteriorPoint{typeof.([z[1], r, rz, rθ])...}(
        s,
        idx,
        ResidualMethods(r!, rz!, rθ!),
        z,
        zero(z),
        r,
        rz,
        rθ,
        zeros(idx.nΔ),
        zeros(idx.nz, num_data),
        zeros(idx.nΔ, num_data),
        θ,
        eval(opts.solver)(rz),
        zero(z),
        zeros(1),
        0,
        opts,
        zeros(1),
        zort, 
        Δort,
        zsoc, 
        Δsoc, 
        ρv, 
        idx_soce)
end

# interior point solver
function interior_point_solve!(ip::InteriorPoint{T,R,RZ,Rθ}) where {T,R,RZ,Rθ} #TODO: change to solve!

    # space
    s = ip.s

    # options
    opts = ip.opts
    r_tol = opts.r_tol
    κ_tol = opts.κ_tol
    ls_scale = opts.ls_scale
    max_iter = opts.max_iter
    max_time = opts.max_time
    max_ls = opts.max_ls
    diff_sol = opts.diff_sol
    ϵ_min = opts.ϵ_min
    κ_reg = opts.κ_reg
    γ_reg = opts.γ_reg
    reg = opts.reg
    verbose = opts.verbose
    warn = opts.warn

    # unpack pre-allocated data
    z = ip.z
    z̄ = ip.z̄
    r = ip.r
    rz = ip.rz
    Δ = ip.Δ
    θ = ip.θ

    # regularization
    ip.reg_val[1] = 0.0 # reset

    zort = ip.zort 
    Δort = ip.Δort
    zsoc = ip.zsoc 
    Δsoc = ip.Δsoc 
    ρv = ip.ρv 

    # indices
    idx = ip.idx
    ortz = idx.ortz
    ortΔ = idx.ortΔ
    socz = idx.socz
    socΔ = idx.socΔ
    bil = idx.bil
    ortr = idx.ortr
    socr = idx.socr
    sorci = idx.socri
    soce = ip.idx_soce

    # initialization
    solver = ip.solver
    ip.iterations = 0

    # evaluate residual
    ip.methods.r!(r, z, θ, 0.0)

    # evaluate bilinear constraint violation
    κ_vio = bilinear_violation(ip, r)

    # evaluate equality constraint violation 
    r_vio = residual_violation(ip, r)

    # begin timer
    elapsed_time = 0.0

    for j = 1:max_iter
        elapsed_time >= max_time && break
        elapsed_time += @elapsed begin
            # check for converged residual
            if (r_vio < r_tol) && (κ_vio < κ_tol)
                break
            end
            ip.iterations += 1

            # Compute regularization level
            regularization!(ip, κ_vio, κ_reg, γ_reg)

            # compute residual Jacobian
            rz!(ip, rz, z, θ, reg=ip.reg_val[1]) # this is not adapted to the second-order cone

            # compute step
            linear_solve!(solver, Δ, rz, r, reg=ip.reg_val[1])

            α_ort = ort_step_length(z, Δ, ortz, ortΔ, τ=1.0)
            α_soc = soc_step_length(z, Δ, socz, socΔ, zsoc, Δsoc, ρv, soce, τ=1.0, verbose=false)

            α = min(α_ort, α_soc)
            μ, σ = centering(z, Δ, ortz, ortΔ, socz, socΔ, zort, Δort, zsoc, Δsoc, α)

            # Compute corrector residual
            ip.κ[1] = max(σ * μ , κ_tol / opts.undercut)
            ip.methods.r!(r, z, θ, ip.κ) # here we set κ = σ*μ, Δ = Δaff
            general_correction_term!(r, Δ, ortr, socr, sorci, ortΔ, socΔ)

            # Compute corrector search direction
            linear_solve!(solver, Δ, rz, r, reg=ip.reg_val[1], fact=false)

            τ = max(0.95, 1 - max(r_vio, κ_vio)^2)

            α_ort = ort_step_length(z, Δ, ortz, ortΔ, τ=τ)
            α_soc = soc_step_length(z, Δ, socz, socΔ, zsoc, Δsoc, ρv, soce, τ=min(τ, 0.99), verbose=false)
            α = min(α_ort, α_soc)

            # reduce norm of residual
            candidate_point!(z, s, z, Δ, α)

            κ_vio_cand = 0.0
            r_vio_cand = 0.0
            for i = 1:max_ls
                ip.methods.r!(r, z, θ, 0.0)
                κ_vio_cand = bilinear_violation(ip, r)
                r_vio_cand = residual_violation(ip, r)
                if (r_vio_cand <= r_vio) || (κ_vio_cand <= κ_vio)
                    break
                end
                verbose && println("linesearch $i")

                # backtracking
                candidate_point!(z, s, z, Δ, -α * ls_scale^i)
            end

            # update
            κ_vio = κ_vio_cand
            r_vio = r_vio_cand
           
            verbose && iterate_info(j, r, r_vio, κ_vio, Δ, α) 
        end
    end

    if (r_vio < r_tol) && (κ_vio < κ_tol)
        # differentiate solution
        regularization!(ip, κ_vio, ip.opts.γ_reg)
        diff_sol && differentiate_solution!(ip, reg=ip.reg_val[1])
        return true
    else
        return false
    end
end

function rz!(ip::InteriorPoint, rz::AbstractMatrix{T}, z::AbstractVector{T},
    θ::AbstractVector{T}; reg::T=0.0) where {T}
    z_reg = ip.z_reg
    ortz = ip.idx.ortz
    socz = ip.idx.socz
    z_reg .= z
    if reg >= 0.0
        for i in eachindex(ortz) # primal-dual
            zi_reg = @view z_reg[ortz[i]]
            zi_reg .= max.(zi_reg, reg)
        end
    end
    ip.methods.rz!(rz, z_reg, θ)
    return nothing
end

function rθ!(ip::InteriorPoint, rθ::AbstractMatrix{T}, z::AbstractVector{T},
        θ::AbstractVector{T}) where {T}
    ip.methods.rθ!(rθ, z, θ)
    return nothing
end

function general_correction_term!(r::AbstractVector{T}, Δ::AbstractVector{T},
    ortr::Vector{Int}, socr::Vector{Int}, socri::Vector{Vector{Int}},
    ortΔ::Vector{Vector{Int}}, socΔ::Vector{Vector{Vector{Int}}}) where {T}
    Δo1 = @views Δ[ortΔ[1]]
    Δo2 = @views Δ[ortΔ[2]]
    rortr = @views r[ortr]
    rortr .+= Δo1 .* Δo2

    num_cone = length(socri)
    for i = 1:num_cone
        Δso1 = @views Δ[socΔ[i][1]]
        Δso2 = @views Δ[socΔ[i][2]]
        rsocri = @views r[socri[i]]
        cone_product!(rsocri, Δso2, Δso1, reset=false)
    end
    return nothing
end

function regularization!(ip::InteriorPoint, κ_vio::T, κ_reg::T, γ_reg::T) where T 
    ip.reg_val[1] = κ_vio < κ_reg ? κ_vio * γ_reg : 0.0
    return nothing
end

function regularization!(ip::InteriorPoint, κ_vio::T, γ_reg::T) where T 
    reg = κ_vio * γ_reg
    reg > ip.reg_val[1] && (ip.reg_val[1] = reg)
    return nothing
end

function residual_violation(ip::InteriorPoint, r::AbstractVector{T}) where {T}
    req = @views r[ip.idx.equr]
    return norm(req, Inf)
end

function centering(z::AbstractVector{T}, Δaff::AbstractVector{T},
    ortz::Vector{Vector{Int}}, ortΔ::Vector{Vector{Int}},
    socz::Vector{Vector{Vector{Int}}}, socΔ::Vector{Vector{Vector{Int}}}, 
    zort::Vector{Vector{T}}, Δort::Vector{Vector{T}},
    zsoc::Vector{Vector{Vector{T}}}, Δsoc::Vector{Vector{Vector{T}}},
    αaff::T) where {T}
    # See Section 5.1.3 in CVXOPT
    # μ only depends on the dot products (no cone product)
    # The CVXOPT linear and quadratic cone program solvers
    n = isempty(ortz[1]) ? 0 : length(ortz[1])
    for sc in socz 
        n += isempty(sc[1]) ? 0 : 1 # dimension of each soc "variable" is 1
    end

    # ineq
    zo1 = @views z[ortz[1]]
    zo2 = @views z[ortz[2]]
    Δo1 = @views Δaff[ortΔ[1]]
    Δo2 = @views Δaff[ortΔ[2]]
    μ = zo1' * zo2

    Δort[1] .= Δo1 
    Δort[1] .*= -αaff 
    Δort[1] .+= zo1 

    Δort[2] .= Δo2 
    Δort[2] .*= -αaff 
    Δort[2] .+= zo2 

    μaff = Δort[1]' * Δort[2]

    # soc
    num_cone = length(socz)
    for i = 1:num_cone
        zs1 = @views z[socz[i][1]]
        zs2 = @views z[socz[i][2]]
        Δs1 = @views Δaff[socΔ[i][1]] 
        Δs2 = @views Δaff[socΔ[i][2]]

        Δsoc[i][1] .= Δs1 
        Δsoc[i][1] .*= -αaff 
        Δsoc[i][1] .+= zs1 

        Δsoc[i][2] .= Δs2 
        Δsoc[i][2] .*= -αaff 
        Δsoc[i][2] .+= zs2 

        μ += zs1' * zs2
        μaff += Δsoc[i][1]' * Δsoc[i][2]
    end
    μ /= n
    μaff /= n
    σ = clamp(μaff / μ, 0.0, 1.0)^3
    return μ, σ
end

function bilinear_violation(ip::InteriorPoint, r::AbstractVector{T}) where {T}
    rbil = @views r[ip.idx.bil]
    return norm(rbil, Inf)
end

function soc_step_length(λ::AbstractVector{T}, Δ::AbstractVector{T}, ρv::AbstractVector{T}, idx::Vector{Int};
    τ::T=0.99, ϵ::T=1e-14, verbose::Bool=false) where {T}
    # check Section 8.2 CVXOPT

    # Adding to slack ϵ to make sure that we never get out of the cone
    λ0 = λ[1] #- ϵ
    λe = @views λ[idx]
    Δe = @views Δ[idx]
    λ_λ = max(λ0^2 - λe' * λe, 1.0e-25)
    λ_λ += ϵ
    λ_Δ = λ0 * Δ[1] - λe' * Δe + ϵ

    ρs = λ_Δ / λ_λ
    ρv .= Δe 
    ρv ./= sqrt(λ_λ)
    β = (λ_Δ / sqrt(λ_λ) .+ Δ[1]) / (λ0 / sqrt(λ_λ) + 1)  / λ_λ
    ρv .-= β .* λe
    # # we make sre that the inverse always exists with ϵ,
    # # if norm(ρv) - ρs) is negative (Δ is pushing towards a more positive cone)
    #     # the computation is ignored and we get the maximum value for α = 1.0
    # # else we have α = τ / norm(ρv) - ρs)
    # # we add ϵ to the denumerator to ensure strict positivity and avoid 1e-16 errors.
    α = 1.0
    if norm(ρv) - ρs > 0.0
        α = min(α, τ / (norm(ρv) - ρs))
    end
    return α
end

function soc_step_length(z::AbstractVector{T}, Δ::AbstractVector{T},
    socz::Vector{Vector{Vector{Int}}}, socΔ::Vector{Vector{Vector{Int}}}, 
    zsoc::Vector{Vector{Vector{T}}}, Δsoc::Vector{Vector{Vector{T}}},
    ρv::Vector{Vector{Vector{T}}}, idx::Vector{Vector{Vector{Int}}};
    τ::T=0.99, verbose::Bool=false) where {T}
    α = 1.0
    num_cone = length(socz)
    for i = 1:num_cone # primal-dual
        for j in eachindex(socz[i]) # number of cones
            # we need -Δ here because we will taking the step x - α Δ
            zsoc[i][j] .= @views z[socz[i][j]]
            Δsoc[i][j] .= @views Δ[socΔ[i][j]]
            Δsoc[i][j] .*= -1.0
            α = min(α, soc_step_length(zsoc[i][j], Δsoc[i][j], ρv[i][j], idx[i][j], τ=τ, verbose=verbose))
        end
    end
    return α
end

function ort_step_length(z::AbstractVector{T}, Δ::AbstractVector{T},
		ortz::Vector{Vector{Int}}, ortΔ::Vector{Vector{Int}};
        τ::T=0.9995) where {T}
    α = 1.0
    num_cone = length(ortz)
    for i = 1:num_cone # primal-dual
        for j in eachindex(ortz[i])
            k = ortz[i][j] # z
            ks = ortΔ[i][j] # Δz
            if Δ[ks] > 0.0
                α = min(α, τ * z[k] / Δ[ks])
            end
        end
    end
    return α
end

function differentiate_solution!(ip::InteriorPoint; reg=0.0)
    s = ip.s
    z = ip.z
    θ = ip.θ
    rz = ip.rz
    rθ = ip.rθ
    δz = ip.δz
    δzs = ip.δzs

    rz!(ip, rz, z, θ, reg = reg)
    rθ!(ip, rθ, z, θ) 

    linear_solve!(ip.solver, δzs, rz, rθ, reg=reg)
    δzs .*= -1.0

    mapping!(δz, s, δzs, z)
    return nothing
end

function iterate_info(j, r, r_vio, κ_vio, Δ, α) 
    println("iter:", j,
            "  r: ", scn(norm(r, Inf)),
            "  r_vio: ", scn(r_vio),
            "  κ_vio: ", scn(κ_vio),
            "  Δ: ", scn(norm(Δ)),
            "  α: ", scn(norm(α)))
end