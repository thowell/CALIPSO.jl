@testset "Hessian of Lagrangian" begin
    MOI = CALIPSO.MOI
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
    dynamics = [dt for t = 1:T-1] 

    # initial state 
    x1 = [0.0; 0.0; 0.0; 0.0] 

    # goal state
    xT = [0.0; π; 0.0; 0.0] 

    # objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    objt = Cost(ot, num_state, num_action, num_parameter=num_parameter, evaluate_hessian=true)
    objT = Cost(oT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    objective = [[objt for t = 1:T-1]..., objT]

    # constraints
    bnd1 = Bound(num_state, num_action, state_lower=x1, state_upper=x1)
    bndt = Bound(num_state, num_action)
    bndT = Bound(num_state, 0, state_lower=xT, state_upper=xT)
    bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

    ct = (x, u, w) -> [-5.0 * ones(num_action) - cos.(u) .* sum(x.^2); cos.(x) .* tan.(u) - 5.0 * ones(num_state)]
    cT = (x, u, w) -> sin.(x.^3.0)
    cont = Constraint(ct, num_state, num_action, num_parameter=num_parameter, indices_inequality=collect(1:(num_action + num_state)), evaluate_hessian=true)
    conT = Constraint(cT, num_state, 0, num_parameter=num_parameter, evaluate_hessian=true)
    constraints = [[cont for t = 1:T-1]..., conT]

    # data 
    solver = Solver(dynamics, objective, constraints, bounds, evaluate_hessian=true)

    # Lagrangian
    function lagrangian(z) 
        x1 = z[1:num_state] 
        u1 = z[num_state .+ (1:num_action)] 
        x2 = z[num_state + num_action .+ (1:num_state)] 
        u2 = z[num_state + num_action + num_state .+ (1:num_action)] 
        x3 = z[num_state + num_action + num_state + num_action .+ (1:num_state)]
        λ1_dyn = z[num_state + num_action + num_state + num_action + num_state .+ (1:num_state)] 
        λ2_dyn = z[num_state + num_action + num_state + num_action + num_state + num_state .+ (1:num_state)] 

        λ1_stage = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state .+ (1:(num_action + num_state))] 
        λ2_stage = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state .+ (1:(num_action + num_state))] 
        λ3_stage = z[num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state .+ (1:num_state)]

        L = 0.0 
        L += ot(x1, u1, zeros(num_parameter)) 
        L += ot(x2, u2, zeros(num_parameter)) 
        L += oT(x3, zeros(0), zeros(num_parameter))
        L += dot(λ1_dyn, midpoint_implicit(x2, x1, u1, zeros(num_parameter))) 
        L += dot(λ2_dyn, midpoint_implicit(x3, x2, u2, zeros(num_parameter))) 
        L += dot(λ1_stage, ct(x1, u1, zeros(num_parameter)))
        L += dot(λ2_stage, ct(x2, u2, zeros(num_parameter)))
        L += dot(λ3_stage, cT(x3, zeros(0), zeros(num_parameter)))
        return L
    end

    nz = num_state + num_action + num_state + num_action + num_state + num_state + num_state + num_action + num_state + num_action + num_state + num_state
    np = num_state + num_action + num_state + num_action + num_state
    nd = num_state + num_state + num_action + num_state + num_action + num_state + num_state
    @variables z[1:nz]
    L = lagrangian(z)
    Lxx = Symbolics.hessian(L, z[1:np])
    Lxx_sp = Symbolics.sparsehessian(L, z[1:np])
    spar = [findnz(Lxx_sp)[1:2]...]
    Lxx_func = eval(Symbolics.build_function(Lxx, z)[1])
    Lxx_sp_func = eval(Symbolics.build_function(Lxx_sp.nzval, z)[1])

    z0 = rand(nz)
    nh = length(solver.nlp.hessian_lagrangian_sparsity)
    h0 = zeros(nh)

    σ = 1.0
    fill!(h0, 0.0)
    CALIPSO.trajectory!(solver.nlp.trajopt.states, solver.nlp.trajopt.actions, z0[1:np], 
        solver.nlp.indices.states, solver.nlp.indices.actions)
    CALIPSO.duals!(solver.nlp.trajopt.duals_dynamics, solver.nlp.trajopt.duals_constraints, solver.nlp.duals_general, z0[np .+ (1:nd)], solver.nlp.indices.dynamics_constraints, solver.nlp.indices.stage_constraints, solver.nlp.indices.general_constraint)
    CALIPSO.hessian!(h0, solver.nlp.indices.objective_hessians, solver.nlp.trajopt.objective, solver.nlp.trajopt.states, solver.nlp.trajopt.actions, solver.nlp.trajopt.parameters, σ)
    CALIPSO.hessian_lagrangian!(h0, solver.nlp.indices.dynamics_hessians, solver.nlp.trajopt.dynamics, solver.nlp.trajopt.states, solver.nlp.trajopt.actions, solver.nlp.trajopt.parameters, solver.nlp.trajopt.duals_dynamics)
    CALIPSO.hessian_lagrangian!(h0, solver.nlp.indices.stage_hessians, solver.nlp.trajopt.constraints, solver.nlp.trajopt.states, solver.nlp.trajopt.actions, solver.nlp.trajopt.parameters, solver.nlp.trajopt.duals_constraints)

    sparsity_objective_hessians = CALIPSO.sparsity_hessian(objective, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)
    sparsity_dynamics_hessians = CALIPSO.sparsity_hessian(dynamics, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)
    sparsity_constraint_hessian = CALIPSO.sparsity_hessian(constraints, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)
    hessian_sparsity = collect([sparsity_objective_hessians..., sparsity_dynamics_hessians..., sparsity_constraint_hessian...]) 
    sp_key = sort(unique(hessian_sparsity))

    idx_objective_hessians = CALIPSO.hessian_indices(objective, sp_key, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)
    idx_dynamics_hessians = CALIPSO.hessian_indices(dynamics, sp_key, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)
    idx_con_hess = CALIPSO.hessian_indices(constraints, sp_key, solver.nlp.trajopt.state_dimensions, solver.nlp.trajopt.action_dimensions)

    # indices
    @test sp_key[vcat(idx_objective_hessians...)] == sparsity_objective_hessians
    @test sp_key[vcat(idx_dynamics_hessians...)] == sparsity_dynamics_hessians
    @test sp_key[vcat(idx_con_hess...)] == sparsity_constraint_hessian

    # Hessian
    h0_full = zeros(np, np)
    for (i, h) in enumerate(h0)
        h0_full[sp_key[i][1], sp_key[i][2]] = h
    end
    @test norm(h0_full - Lxx_func(z0)) < 1.0e-8
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    h0 = zeros(nh)
    MOI.eval_hessian_lagrangian(solver.nlp, h0, z0[1:np], σ, z0[np .+ (1:nd)])
    @test norm(norm(h0 - Lxx_sp_func(z0))) < 1.0e-8

    # a = z0[1:np]
    # b = z0[np .+ (1:nd)]
    # info = @benchmark MOI.evaluate_hessianian_lagrangian($solver, $h0, $a, $σ, $b)
end
