@testset "Hessian of Lagrangian" begin
    # horizon
    horizon = 3

    # acrobot
    num_state = 4
    num_action = 1
    num_parameter = 0
    num_states = [num_state for t = 1:horizon]
    num_actions = [num_action for t = 1:horizon-1]
    num_parameters = [num_parameter for t = 1:horizon]

    function acrobot_continuous(x, u)
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
        function M(x)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return [a b; b c]
        end

        function Minv(x)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return 1.0 / (a * c - b * b) * [c -b; -b a]
        end

        # dynamics bias
        function τ(x)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        # input Jacobian
        function B(x)
            [0.0; 1.0]
        end

        # dynamics
        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = Minv(q) * (-1.0 * C(x) * v
                + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function acrobot_discrete(y, x, u)
        timestep = 0.05 # timestep
        y - (x + timestep * acrobot_continuous(0.5 * (x + y), u))
    end

    dynamics = [acrobot_discrete for t = 1:horizon-1]
    dyn = CALIPSO.generate_methods(dynamics, num_states, num_actions, num_parameters, :Dynamics)

    # initial state
    x1 = [0.0; 0.0; 0.0; 0.0]

    # goal state
    xT = [0.0; π; 0.0; 0.0]

    # objective
    objective = [
            [(x, u) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u) for t = 1:horizon-1]...,
            (x, u) -> 0.1 * dot(x[3:4], x[3:4]),
    ]
    obj = CALIPSO.generate_methods(objective, num_states, num_actions, num_parameters, :Cost)

    # equality constraints
    equality = [
            [(x, u) -> [-5.0 * ones(num_actions[t]) - cos.(u) .* sum(x.^2); cos.(x) .* tan.(u) - 5.0 * ones(num_states[t])] for t = 1:horizon-1]...,
            (x, u) -> sin.(x.^3.0),
    ]
    eq = CALIPSO.generate_methods(equality, num_states, num_actions, num_parameters, :Constraint)

    eg = CALIPSO.EqualityGeneral()

    # nonnegative constraints
    inequality = [
            [(x, u) -> sin.(x) .- sum(u) for t = 1:horizon-1]...,
            (x, u) -> cos.(x) .+ sum(u),
    ]
    ineq = CALIPSO.generate_methods(inequality, num_states, num_actions, num_parameters, :Constraint)

    # second-order constraints
    st = (x, u, e) -> e .* tan.(x) .- sqrt(sum(u.^2) + e^2)
    sT = (x, u, e) -> e .* tan.(x) .+ cos(sum(u) + e^2)
    nsi = 2
    rr = [randn(1)[1] for i = 1:nsi]
    sot = [(x, u) -> st(x, u, rr[i]) for i = 1:nsi]
    soT = [(x, u) -> sT(x, u, rr[i]) for i = 1:nsi]
    second_order = [[sot for t = 1:horizon-1]..., soT]
    so = [[Constraint(s, num_states[t], t == horizon ? 0 : num_actions[t];
            num_parameter=num_parameters[t]) for s in second_order[t]] for t = 1:horizon]

    # data
    trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, eg, ineq, so, constraint_tensor=true);

    # dimensions
    np = num_state + num_action + num_state + num_action + num_state
    nde = num_state + num_state + num_action + num_state + num_action + num_state + num_state
    ndi = horizon * num_state
    nds = horizon * sum([num_state for i = 1:nsi])
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

        λ1_soc = [z[np + nde + ndi + (i - 1) * num_state .+ (1:num_state)] for i = 1:nsi]
        λ2_soc = [z[np + nde + ndi + nsi * num_state + (i - 1) * num_state .+ (1:num_state)] for i = 1:nsi]
        λ3_soc = [z[np + nde + ndi + nsi * num_state + nsi * num_state + (i - 1) * num_state .+ (1:num_state)] for i = 1:nsi]

        L = 0.0

        L += objective[1](x1, u1)
        L += objective[2](x2, u2)
        L += objective[3](x3, zeros(0))

        L += dot(λ1_dyn, acrobot_discrete(x2, x1, u1))
        L += dot(λ2_dyn, acrobot_discrete(x3, x2, u2))

        L += dot(λ1_eq, equality[1](x1, u1))
        L += dot(λ2_eq, equality[2](x2, u2))
        L += dot(λ3_eq, equality[3](x3, zeros(0)))

        L += dot(λ1_ineq, inequality[1](x1, u1))
        L += dot(λ2_ineq, inequality[2](x2, u2))
        L += dot(λ3_ineq, inequality[3](x3, zeros(0)))

        L += sum([dot(λ1_soc[i], st(x1, u1, rr[i])) for i = 1:nsi])
        L += sum([dot(λ2_soc[i], st(x2, u2, rr[i])) for i = 1:nsi])
        L += sum([dot(λ3_soc[i], sT(x3, zeros(0), rr[i])) for i = 1:nsi])

        return L
    end

    z = Symbolics.variables(:z, 1:nz)

    L = lagrangian(z)
    Lxx = Symbolics.hessian(L, z[1:np])
    Lxx_sp = Symbolics.sparsehessian(L, z[1:np])
    spar = [findnz(Lxx_sp)[1:2]...]

    Lxx_func = eval(Symbolics.build_function(Lxx, z)[1])
    Lxx_sp_func = eval(Symbolics.build_function(Lxx_sp.nzval, z)[1])

    sparsity_objective_jacobians_variables_variables = CALIPSO.sparsity_jacobian_variables_variables(obj, trajopt.dimensions.states, trajopt.dimensions.actions)
    sparsity_objective_jacobians_variables_parameters = CALIPSO.sparsity_jacobian_variables_parameters(obj, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    sparsity_dynamics_jacobian_variables_variables = CALIPSO.sparsity_jacobian_variables_variables(dyn, trajopt.dimensions.states, trajopt.dimensions.actions)
    sparsity_dynamics_jacobian_variables_parameters = CALIPSO.sparsity_jacobian_variables_parameters(dyn, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    sparsity_equality_jacobian_variables_variables = CALIPSO.sparsity_jacobian_variables_variables(eq, trajopt.dimensions.states, trajopt.dimensions.actions)
    sparsity_equality_jacobian_variables_parameters = CALIPSO.sparsity_jacobian_variables_parameters(eq, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    sparsity_nonnegative_jacobian_variables_variables = CALIPSO.sparsity_jacobian_variables_variables(ineq, trajopt.dimensions.states, trajopt.dimensions.actions)
    sparsity_nonnegative_jacobian_variables_parameters = CALIPSO.sparsity_jacobian_variables_parameters(ineq, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    sparsity_second_order_jacobian_variables_variables = CALIPSO.sparsity_jacobian_variables_variables(so, trajopt.dimensions.states, trajopt.dimensions.actions)
    sparsity_second_order_jacobian_variables_parameters = CALIPSO.sparsity_jacobian_variables_parameters(so, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    jacobian_variables_variables_sparsity = collect([
        (sparsity_objective_jacobians_variables_variables...)...,
        (sparsity_objective_jacobians_variables_variables...)...,
        (sparsity_dynamics_jacobian_variables_variables...)...,
        (sparsity_equality_jacobian_variables_variables...)...,
        (sparsity_nonnegative_jacobian_variables_variables...)...,
        ((sparsity_second_order_jacobian_variables_variables...)...)...,
        ])
    sp_vv_key = sort(unique(jacobian_variables_variables_sparsity))

    # jacobian_variables_parameters_sparsity = collect([
    #     (sparsity_objective_jacobians_variables_parameters...)...,
    #     (sparsity_objective_jacobians_variables_parameters...)...,
    #     (sparsity_dynamics_jacobian_variables_parameters...)...,
    #     (sparsity_equality_jacobian_variables_parameters...)...,
    #     (sparsity_nonnegative_jacobian_variables_parameters...)...,
    #     ((sparsity_second_order_jacobian_variables_parameters...)...)...,
    #     ])
    # sp_vp_key = sort(unique(jacobian_variables_parameters_sparsity))

    # jacobian_variables_parameters_sparsity = Tuple{Int64,Int64}[]
    # sp_vp_key = Int[]

    idx_objective_jacobians_variables_variables = CALIPSO.jacobian_variables_variables_indices(obj, sp_vv_key, trajopt.dimensions.states, trajopt.dimensions.actions)
    idx_dynamics_jacobian_variables_variables = CALIPSO.jacobian_variables_variables_indices(dyn, sp_vv_key, trajopt.dimensions.states, trajopt.dimensions.actions)
    idx_eq_jacobian_variables_variables = CALIPSO.jacobian_variables_variables_indices(eq, sp_vv_key, trajopt.dimensions.states, trajopt.dimensions.actions)
    idx_nn_jacobian_variables_variables = CALIPSO.jacobian_variables_variables_indices(ineq, sp_vv_key, trajopt.dimensions.states, trajopt.dimensions.actions)
    idx_so_jacobian_variables_variables = CALIPSO.jacobian_variables_variables_indices(so, sp_vv_key, trajopt.dimensions.states, trajopt.dimensions.actions)

    # idx_objective_jacobians_variables_parameters = CALIPSO.jacobian_variables_parameters_indices(obj, sp_vp_key, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)
    # idx_dynamics_jacobian_variables_parameters = CALIPSO.jacobian_variables_parameters_indices(dyn, sp_vp_key, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)
    # idx_eq_jacobian_variables_parameters = CALIPSO.jacobian_variables_parameters_indices(eq, sp_vp_key, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)
    # idx_nn_jacobian_variables_parameters = CALIPSO.jacobian_variables_parameters_indices(ineq, sp_vp_key, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)
    # idx_so_jacobian_variables_parameters = CALIPSO.jacobian_variables_parameters_indices(so, sp_vp_key, trajopt.dimensions.states, trajopt.dimensions.actions, trajopt.dimensions.parameters)

    # indices
    @test sp_vv_key[vcat(idx_objective_jacobians_variables_variables...)] == [(sparsity_objective_jacobians_variables_variables...)...]
    @test sp_vv_key[vcat(idx_dynamics_jacobian_variables_variables...)] == [(sparsity_dynamics_jacobian_variables_variables...)...]
    @test sp_vv_key[vcat(idx_eq_jacobian_variables_variables...)] == [(sparsity_equality_jacobian_variables_variables...)...]
    @test sp_vv_key[vcat(idx_nn_jacobian_variables_variables...)] == [(sparsity_nonnegative_jacobian_variables_variables...)...]
    @test sp_vv_key[vcat(idx_so_jacobian_variables_variables[1]...)] == [(sparsity_second_order_jacobian_variables_variables[1]...)...]
    @test sp_vv_key[vcat(idx_so_jacobian_variables_variables[2]...)] == [(sparsity_second_order_jacobian_variables_variables[2]...)...]
    @test sp_vv_key[vcat(idx_so_jacobian_variables_variables[3]...)] == [(sparsity_second_order_jacobian_variables_variables[3]...)...]

    # @test sp_vp_key[vcat(idx_objective_jacobians_variables_parameters...)] == [(sparsity_objective_jacobians_variables_parameters...)...]
    # @test sp_vp_key[vcat(idx_dynamics_jacobian_variables_parameters...)] == [(sparsity_dynamics_jacobian_variables_parameters...)...]
    # @test sp_vp_key[vcat(idx_eq_jacobian_variables_parameters...)] == [(sparsity_equality_jacobian_variables_parameters...)...]
    # @test sp_vp_key[vcat(idx_nn_jacobian_variables_parameters...)] == [(sparsity_nonnegative_jacobian_variables_parameters...)...]
    # @test sp_vp_key[vcat(idx_so_jacobian_variables_parameters[1]...)] == [(sparsity_second_order_jacobian_variables_parameters[1]...)...]
    # @test sp_vp_key[vcat(idx_so_jacobian_variables_parameters[2]...)] == [(sparsity_second_order_jacobian_variables_parameters[2]...)...]
    # @test sp_vp_key[vcat(idx_so_jacobian_variables_parameters[3]...)] == [(sparsity_second_order_jacobian_variables_parameters[3]...)...]

    z0 = rand(nz)

    ho = zeros(length(vcat(sparsity_objective_jacobians_variables_variables...)))
    he = zeros(length(vcat(sparsity_dynamics_jacobian_variables_variables..., sparsity_equality_jacobian_variables_variables...)))
    hc = zeros(length(vcat(sparsity_nonnegative_jacobian_variables_variables..., (sparsity_second_order_jacobian_variables_variables...)...)))

    Ho = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)
    He = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)
    Hc = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)

    length(vcat(sparsity_dynamics_jacobian_variables_variables...))
    length(vcat(trajopt.sparsity.dynamics_jacobian_variables_variables...))
    length(vcat(sparsity_equality_jacobian_variables_variables...))
    CALIPSO.objective_jacobian_variables_variables!(ho, trajopt, z0[1:np], zeros(trajopt.dimensions.total_parameters))
    CALIPSO.equality_jacobian_variables_variables!(he, trajopt, z0[1:np], z0[np .+ (1:nde)], zeros(trajopt.dimensions.total_parameters))
    CALIPSO.cone_jacobian_variables_variables!(hc, trajopt, z0[1:np], z0[np + nde .+ (1:ndc)], zeros(trajopt.dimensions.total_parameters))

    for (i, idx) in enumerate(vcat(sparsity_objective_jacobians_variables_variables...))
        Ho[idx...] = ho[i]
    end

    for (i, idx) in enumerate(vcat(sparsity_dynamics_jacobian_variables_variables..., sparsity_equality_jacobian_variables_variables...))
        He[idx...] += he[i]
    end

    for (i, idx) in enumerate(vcat(sparsity_nonnegative_jacobian_variables_variables..., (sparsity_second_order_jacobian_variables_variables...)...))
        Hc[idx...] += hc[i]
    end

    @test norm(norm((Ho + He + Hc) - Lxx_func(z0))) < 1.0e-8
end
