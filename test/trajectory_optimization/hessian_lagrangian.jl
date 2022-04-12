@testset "Hessian of Lagrangian" begin
    # horizon 
    T = 3

    # acrobot 
    num_state = 4 
    num_action = 1 
    num_parameter = 0 
    parameter_dimensions = [num_parameter for t = 1:T]

    function acrobot(x, u, w)
        # dimensions
        n = 4
        m = 1
        d = 0

        # link 1
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        # link 2
        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        # mass matrix
        function M(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return [a b; b c]
        end

        function Minv(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return 1.0 / (a * c - b * b) * [c -b; -b a]
        end

        # dynamics bias
        function τ(x, w)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x, w)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        # input Jacobian
        function B(x, w)
            [0.0; 1.0]
        end

        # dynamics
        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = Minv(q, w) * (-1.0 * C(x, w) * v
                + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function midpoint_implicit(y, x, u, w)
        h = 0.05 # timestep 
        y - (x + h * acrobot(0.5 * (x + y), u, w))
    end

    dt = Dynamics(midpoint_implicit, num_state, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    dyn = [dt for t = 1:T-1] 

    # initial state 
    x1 = [0.0; 0.0; 0.0; 0.0] 

    # goal state
    xT = [0.0; π; 0.0; 0.0] 

    # objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    objt = Cost(ot, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    objT = Cost(oT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    obj = [[objt for t = 1:T-1]..., objT]

    # equality constraints
    et = (x, u, w) -> [-5.0 * ones(num_action) - cos.(u) .* sum(x.^2); cos.(x) .* tan.(u) - 5.0 * ones(num_state)]
    eT = (x, u, w) -> sin.(x.^3.0)
    eqt = Constraint(et, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    eqT = Constraint(eT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    eq = [[eqt for t = 1:T-1]..., eqT]

    # non-negative constraints
    it = (x, u, w) -> sin.(x) .- sum(u)
    iT = (x, u, w) -> cos.(x) .+ sum(u)
    ineqt = Constraint(it, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    ineqT = Constraint(iT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    ineq = [[ineqt for t = 1:T-1]..., ineqT]

    # non-negative constraints
    st = (x, u, w) -> tan.(x) .- sqrt(sum(u.^2))
    sT = (x, u, w) -> tan.(x) .+ cos(sum(u))
    sot = Constraint(st, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    soT = Constraint(sT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    so = [[sot for t = 1:T-1]..., soT]

    # data 
    trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so, evaluate_hessian=true)

    # dimensions 
    np = num_state + num_action + num_state + num_action + num_state
    nde = num_state + num_state + num_action + num_state + num_action + num_state + num_state
    ndi = T * num_state
    nds = T * num_state
    ndc = ndi + nds
    nd = nde + ndc
    nz = np + nd

    # Lagrangian
    function lagrangian(z) 
        x1 = z[1:num_state] 
        u1 = z[num_state .+ (1:num_action)] 
        x2 = z[num_state + num_action .+ (1:num_state)] 
        u2 = z[num_state + num_action + num_state .+ (1:num_action)] 
        x3 = z[num_state + num_action + num_state + num_action .+ (1:num_state)]

        λ1_dyn = z[np .+ (1:num_state)] 
        λ2_dyn = z[np + num_state .+ (1:num_state)] 

        λ1_eq = z[np + num_state + num_state .+ (1:(num_action + num_state))] 
        λ2_eq = z[np + num_state + num_state + num_action + num_state .+ (1:(num_action + num_state))] 
        λ3_eq = z[np + num_state + num_state + 2 * (num_action + num_state) .+ (1:num_state)]

        λ1_ineq = z[np + nde .+ (1:num_state)]
        λ2_ineq = z[np + nde + num_state .+ (1:num_state)]
        λ3_ineq = z[np + nde + num_state + num_state .+ (1:num_state)]

        λ1_soc = z[np + nde + ndi .+ (1:num_state)]
        λ2_soc = z[np + nde + ndi + num_state .+ (1:num_state)]
        λ3_soc = z[np + nde + ndi + num_state + num_state .+ (1:num_state)]

        L = 0.0 

        L += ot(x1, u1, zeros(num_parameter)) 
        L += ot(x2, u2, zeros(num_parameter)) 
        L += oT(x3, zeros(0), zeros(num_parameter))

        L += dot(λ1_dyn, midpoint_implicit(x2, x1, u1, zeros(num_parameter))) 
        L += dot(λ2_dyn, midpoint_implicit(x3, x2, u2, zeros(num_parameter))) 

        L += dot(λ1_eq, et(x1, u1, zeros(num_parameter)))
        L += dot(λ2_eq, et(x2, u2, zeros(num_parameter)))
        L += dot(λ3_eq, eT(x3, zeros(0), zeros(num_parameter)))

        L += dot(λ1_ineq, it(x1, u1, zeros(num_parameter)))
        L += dot(λ2_ineq, it(x2, u2, zeros(num_parameter)))
        L += dot(λ3_ineq, iT(x3, zeros(0), zeros(num_parameter)))

        L += dot(λ1_soc, st(x1, u1, zeros(num_parameter)))
        L += dot(λ2_soc, st(x2, u2, zeros(num_parameter)))
        L += dot(λ3_soc, sT(x3, zeros(0), zeros(num_parameter)))

        return L
    end

    @variables z[1:nz]
    L = lagrangian(z)
    Lxx = Symbolics.hessian(L, z[1:np])
    Lxx_sp = Symbolics.sparsehessian(L, z[1:np])
    spar = [findnz(Lxx_sp)[1:2]...]
    Lxx_func = eval(Symbolics.build_function(Lxx, z)[1])
    Lxx_sp_func = eval(Symbolics.build_function(Lxx_sp.nzval, z)[1])

    sparsity_objective_hessians = CALIPSO.sparsity_hessian(obj, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    sparsity_dynamics_hessians = CALIPSO.sparsity_hessian(dyn, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    sparsity_equality_hessian = CALIPSO.sparsity_hessian(eq, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    sparsity_nonnegative_hessian = CALIPSO.sparsity_hessian(ineq, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    sparsity_second_order_hessian = CALIPSO.sparsity_hessian(so, trajopt.data.state_dimensions, trajopt.data.action_dimensions)

    hessian_sparsity = collect([(sparsity_objective_hessians...)..., 
        (sparsity_dynamics_hessians...)..., 
        (sparsity_equality_hessian...)...,
        (sparsity_nonnegative_hessian...)...,
        (sparsity_second_order_hessian...)...,
        ]) 
    sp_key = sort(unique(hessian_sparsity))

    idx_objective_hessians = CALIPSO.hessian_indices(obj, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_dynamics_hessians = CALIPSO.hessian_indices(dyn, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_eq_hess = CALIPSO.hessian_indices(eq, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_nn_hess = CALIPSO.hessian_indices(ineq, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_so_hess = CALIPSO.hessian_indices(so, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)

    # indices
    @test sp_key[vcat(idx_objective_hessians...)] == [(sparsity_objective_hessians...)...]
    @test sp_key[vcat(idx_dynamics_hessians...)] == [(sparsity_dynamics_hessians...)...]
    @test sp_key[vcat(idx_eq_hess...)] == [(sparsity_equality_hessian...)...]
    @test sp_key[vcat(idx_nn_hess...)] == [(sparsity_nonnegative_hessian...)...]
    @test sp_key[vcat(idx_so_hess...)] == [(sparsity_second_order_hessian...)...]

    z0 = rand(nz)
    σ = 1.0
    ho = zeros(trajopt.num_variables, trajopt.num_variables)
    he = zeros(trajopt.num_variables, trajopt.num_variables)
    hc = zeros(trajopt.num_variables, trajopt.num_variables)

    CALIPSO.objective_hessian!(ho, trajopt, z0[1:np])
    CALIPSO.equality_hessian!(he, trajopt, z0[1:np], z0[np .+ (1:nde)])
    CALIPSO.cone_hessian!(hc, trajopt, z0[1:np], z0[np + nde .+ (1:ndc)])

    @test norm(norm((ho + he + hc) - Lxx_func(z0))) < 1.0e-8
end
