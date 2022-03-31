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

    # inequality constraints
    it = (x, u, w) -> sin.(x) .- sum(u)
    iT = (x, u, w) -> cos.(x) .+ sum(u)
    ineqt = Constraint(it, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    ineqT = Constraint(iT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    ineq = [[ineqt for t = 1:T-1]..., ineqT]

    # data 
    trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, evaluate_hessian=true)

    # Lagrangian
    function lagrangian(z) 
        x1 = z[1:num_state] 
        u1 = z[num_state .+ (1:num_action)] 
        x2 = z[num_state + num_action .+ (1:num_state)] 
        u2 = z[num_state + num_action + num_state .+ (1:num_action)] 
        x3 = z[num_state + num_action + num_state + num_action .+ (1:num_state)]
        λ1_dyn = z[num_state + num_action + num_state + num_action + num_state .+ (1:num_state)] 
        λ2_dyn = z[num_state + num_action + num_state + num_action + num_state + num_state .+ (1:num_state)] 

        λ1_eq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state .+ (1:(num_action + num_state))] 
        λ2_eq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state .+ (1:(num_action + num_state))] 
        λ3_eq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state .+ (1:num_state)]

        λ1_ineq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state + num_state .+ (1:num_state)]
        λ2_ineq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state + num_state + num_state .+ (1:num_state)]
        λ3_ineq = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_state .+ (1:num_state)]

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
        return L
    end

    nz = num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state + num_state + T * num_state
    np = num_state + num_action + num_state + num_action + num_state
    nd = num_state + num_state + num_action + num_state + num_action + num_state + num_state + T * num_state
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
    sparsity_inequality_hessian = CALIPSO.sparsity_hessian(ineq, trajopt.data.state_dimensions, trajopt.data.action_dimensions)

    hessian_sparsity = collect([(sparsity_objective_hessians...)..., 
        (sparsity_dynamics_hessians...)..., 
        (sparsity_equality_hessian...)...,
        (sparsity_inequality_hessian...)...]) 
    sp_key = sort(unique(hessian_sparsity))

    idx_objective_hessians = CALIPSO.hessian_indices(obj, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_dynamics_hessians = CALIPSO.hessian_indices(obj, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_eq_hess = CALIPSO.hessian_indices(eq, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)
    idx_ineq_hess = CALIPSO.hessian_indices(ineq, sp_key, trajopt.data.state_dimensions, trajopt.data.action_dimensions)

    # indices
    @test sp_key[vcat(idx_objective_hessians...)] == [(sparsity_objective_hessians...)...]
    @test sp_key[vcat(idx_dynamics_hessians...)] == sp_key[vcat(idx_dynamics_hessians...)]
    @test sp_key[vcat(idx_eq_hess...)] == [(sparsity_equality_hessian...)...]
    @test sp_key[vcat(idx_ineq_hess...)] == [(sparsity_inequality_hessian...)...]

    z0 = rand(nz)
    σ = 1.0
    h0 = zeros(trajopt.num_variables, trajopt.num_variables)
    CALIPSO.hessian_lagrangian(h0, trajopt, z0[1:np], z0[np .+ (1:nd)], scaling=σ)
    @test norm(norm(h0 - Lxx_func(z0))) < 1.0e-8
end
