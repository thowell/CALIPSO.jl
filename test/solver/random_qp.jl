@testset "Solver: Random QP w/ inequalities (LU)" begin
    """
        minimize   x' P x + q' x
        subject to    x >= 0
    """

    n = 10

    _P = rand(n)
    P = Diagonal(_P)
    q = rand(n)
    θ = [_P; q]
    z = ones(2 * n)
    r = zeros(2 * n)
    rz = zeros(2 * n, 2 * n)
    rθ = zeros(2 * n, 2 * n)
    κ = [1.0]

    # residual
    function _r!(r, z, θ, κ)
        x = z[1:10]
        y = z[11:20]
        P = θ[1:10]
        q = θ[11:20]

        r[1:10] = 2.0 * Diagonal(P) * x + q - y
        r[11:20] = Diagonal(x) * y .- κ[1]
        nothing
    end

    @variables r_sym[1:20]
    @variables z_sym[1:20]
    @variables θ_sym[1:20]
    @variables κ_sym[1:1]

    parallel = Symbolics.SerialForm()
    _r!(r_sym, z_sym, θ_sym, κ_sym)
    rf! = eval(Symbolics.build_function(r_sym, z_sym, θ_sym, κ_sym,
        parallel = parallel)[2])
    rz_exp = Symbolics.jacobian(r_sym, z_sym)
    rθ_exp = Symbolics.jacobian(r_sym, θ_sym)
    rz_sp = similar(rz_exp, Float64)
    rθ_sp = similar(rθ_exp, Float64)
    rzf! = eval(Symbolics.build_function(rz_exp, z_sym, θ_sym,
        parallel = parallel)[2])
    rθf! = eval(Symbolics.build_function(rθ_exp, z_sym, θ_sym,
        parallel = parallel)[2])

    # options
    opts = CALIPSO.InteriorPointOptions(diff_sol=true, verbose=false)
    idx = CALIPSO.IndicesOptimization(
            2n, 2n, 
            [collect(1:n), collect(n .+ (1:n))],
            [collect(1:n), collect(n .+ (1:n))], 
            Vector{Vector{Int}}[], 
            Vector{Vector{Int}}[], 
            collect(1:n), 
            collect(n .+ (1:n)), 
            collect(1:0),
            Vector{Int}[],
            collect(n .+ (1:n)))

    # solver
    ip = CALIPSO.interior_point(z, θ,
        idx=idx,
        r! = rf!, rz! = rzf!, rθ! = rθf!,
        rz = rz_sp,
        rθ = rθ_sp,
        opts = opts)

    # solve
    status = CALIPSO.interior_point_solve!(ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test all(ip.z[idx.ortz[1]] .> 0.0)
    @test all(ip.z[idx.ortz[2]] .> 0.0)
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0
end


@testset "Solver: Random QP w/ equalities (QDLDL)" begin
    """
        minimize   x' P x + q' x
        subject to    Ax = b
    """

    ##### 
    n = 10
    m = 3

    nw = n + m 
    nθ = 2n + m * n + m

    function obj(z, θ)
        x = z[1:n]

        P = Diagonal(θ[1:n])
        p = θ[n .+ (1:n)]
        return transpose(x) * P * x + transpose(p) * x 
    end

    function constraints(z, θ) 
        x = z[1:n] 
        A = reshape(θ[2n .+ (1:(m * n))], m, n)
        b = θ[2n + m * n .+ (1:m)]

        A * x - b
    end

    function lagrangian(w, θ)
        z = w[1:n]
        y = w[n .+ (1:m)]

        L = 0.0 
        L += obj(z, θ)
        L += dot(y, constraints(z, θ))

        return L
    end

    # residual
    function _r(w, θ, κ)
        L = lagrangian(w, θ)
        Lz = Symbolics.gradient(L, w[1:n])
        c = constraints(w[1:n], θ)

        [
            Lz; 
            c; 
        ]
    end

    @variables z_sym[1:nw]
    @variables θ_sym[1:nθ]
    @variables κ_sym[1:1]

    r_sym = _r(z_sym, θ_sym, κ_sym)
    rf! = eval(Symbolics.build_function(r_sym, z_sym, θ_sym, κ_sym)[2])
    rz_exp = Symbolics.jacobian(r_sym, z_sym)
    rθ_exp = Symbolics.jacobian(r_sym, θ_sym)
    rz_sp = similar(rz_exp, Float64)
    rθ_sp = similar(rθ_exp, Float64)
    rzf! = eval(Symbolics.build_function(rz_exp, z_sym, θ_sym)[2])
    rθf! = eval(Symbolics.build_function(rθ_exp, z_sym, θ_sym)[2])

    x0 = randn(n)
    A = rand(m, n)
    b = A * x0
    z = [randn(n); zeros(m)] 
    θ = [ones(n); zeros(n); vec(A); b]
    κ = [1.0]

    r = zeros(nw) 
    rz = zeros(nw, nw) 
    rθ = zeros(nw, nθ) 

    opts = CALIPSO.InteriorPointOptions(diff_sol=true, 
        undercut=10.0,
        max_ls=25, 
        max_iter=100, 
        verbose=false, 
        solver=:lu_solver)

    idx = CALIPSO.IndicesOptimization(
        nw, nw, 
        [collect(1:0), collect(1:0)],
        [collect(1:0), collect(1:0)],
        Vector{Vector{Int}}[], 
        Vector{Vector{Int}}[], 
        collect(1:(n + m)), 
        Vector{Int}(), 
        Vector{Int}(),
        Vector{Int}[],
        Vector{Int}())

    # solver
    ip = CALIPSO.interior_point(z, θ,
        idx=idx,
        r! = rf!, rz! = rzf!, rθ! = rθf!,
        rz = rz,
        rθ = rθ,
        opts = opts)

    status = CALIPSO.interior_point_solve!(ip)
    # @benchmark CALIPSO.interior_point_solve!($ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test norm(A * ip.z[1:n] - b, Inf) < 1.0e-6
    @test norm(ip.δz, 1) != 0.0
end
