@testset "Solver: second-order cone program (friction)" begin
	"""
		minimize   v' b
		subject to ||b|| <= μn
	"""

    # second-order cone
    a = [2.0; 1.0] 
    b = [1.0; 0.1] 

    c = CALIPSO.cone_product(a, b)
    @test norm(c - [2.1; 1.2]) < 1.0e-8

    d = zeros(2) 
    CALIPSO.cone_product!(d, a, b)
    @test norm(d - c) < 1.0e-8

    # number of decision variables
    n = 7
    m = 3
    idx = CALIPSO.IndicesOptimization(
        n, n, 
        [Int[], Int[]], 
        [Int[], Int[]],
        [[collect(1:3), collect(5:7)]], 
        [[collect(1:3), collect(5:7)]],
        collect(1:4),
        collect(1:0),
        collect(5:7),
        [collect(5:7)],
        collect(5:7),
    )

    # residual
    function _r!(r, z, θ, κ)
        x = z[1:3]
        y = z[4:4]
        z = z[5:7]

        v = θ[1:2]
        μγ = θ[3]

        r[1:3] = [0.0; v] + [y; 0.0; 0.0] - z
        r[4] = x[1] - μγ
        r[5:7] = CALIPSO.cone_product(x, z) - [κ[1]; 0.0; 0.0]
        nothing
    end

    @variables r_sym[1:n]
    @variables z_sym[1:n]
    @variables θ_sym[1:m]
    @variables κ_sym[1:1]

    parallel = Symbolics.SerialForm()
    _r!(r_sym, z_sym, θ_sym, κ_sym)
    r_sym = simplify.(r_sym)
    rf! = eval(Symbolics.build_function(r_sym, z_sym, θ_sym, κ_sym,
        parallel = parallel)[2])
    rz_exp = Symbolics.jacobian(r_sym, z_sym, simplify = true)
    rθ_exp = Symbolics.jacobian(r_sym, θ_sym, simplify = true)
    rz_sp = similar(rz_exp, Float64)
    rθ_sp = similar(rθ_exp, Float64)
    rzf! = eval(Symbolics.build_function(rz_exp, z_sym, θ_sym,
        parallel = parallel)[2])
    rθf! = eval(Symbolics.build_function(rθ_exp, z_sym, θ_sym,
        parallel = parallel)[2])

    # problem setup
    v = [0.0; 0.0]
    μγ = 1.0
    θ = [v; μγ]

    z = ones(n)
    z[1] += 1.0
    z[5] += 1.0

    # solver
    opts = CALIPSO.InteriorPointOptions(diff_sol = true)
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
    @test all([norm(ip.z[j][2:3]) <= ip.z[j][1] for i in ip.idx.socz for j in i])
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0

    # problem setup
    v = [1.0; 0.0]
    μγ = 1.0
    θ = [v; μγ]

    z = ones(n)
    z[1] += 1.0
    z[5] += 1.0

    ip.z .= z
    ip.θ .= θ

    # solve
    status = CALIPSO.interior_point_solve!(ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test all([norm(ip.z[j][2:3]) <= ip.z[j][1] for i in ip.idx.socz for j in i])
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0

    # problem setup
    v = [1.0; 10.0]
    μγ = 1.0
    θ = [v; μγ]

    z = ones(n)
    z[1] += 1.0
    z[5] += 1.0

    ip.z .= z
    ip.θ .= θ

    # solve
    status = CALIPSO.interior_point_solve!(ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test all([norm(ip.z[j][2:3]) <= ip.z[j][1] for i in ip.idx.socz for j in i])
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0

    # problem setup
    v = [1.0; 10.0]
    μγ = 0.0
    θ = [v; μγ]

    z = ones(n)
    z[1] += 1.0
    z[5] += 1.0

    ip.z .= z
    ip.θ .= θ

    # solve
    status = CALIPSO.interior_point_solve!(ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test all([norm(ip.z[j][2:3]) <= ip.z[j][1] for i in ip.idx.socz for j in i])
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0

    # problem setup
    v = [0.0; 0.0]
    μγ = 0.0
    θ = [v; μγ]

    z = ones(n)
    z[1] += 1.0
    z[5] += 1.0

    ip.z .= z
    ip.θ .= θ

    # solve
    status = CALIPSO.interior_point_solve!(ip)

    # test
    @test status
    @test norm(ip.r, Inf) < opts.r_tol
    @test all([norm(ip.z[j][2:3]) <= ip.z[j][1] for i in ip.idx.socz for j in i])
    @test ip.κ[1] < opts.κ_tol
    @test norm(ip.δz, 1) != 0.0
end
