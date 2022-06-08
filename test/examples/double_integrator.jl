@testset "Examples: Double integrator (differentiate wrt parameters)" begin
    # ## horizon 
    horizon = 5

    # ## dimensions 
    num_states = [2 for t = 1:horizon]
    num_actions = [1 for t = 1:horizon-1] 

    # ## dynamics
    function double_integrator(y, x, u, w)
        A = reshape(w[1:4], 2, 2) 
        B = w[4 .+ (1:2)] 

        return y - (A * x + B * u[1])
    end

    # ## model
    dynamics = [double_integrator for t = 1:horizon-1]

    # ## parameters
    state_initial = [0.0; 0.0] 
    state_goal = [1.0; 0.0] 

    A = [1.0 1.0; 0.0 1.0]
    B = [0.0; 1.0]
    Qt = [1.0 0.0; 0.0 1.0] 
    Rt = [0.1]
    QT = [10.0 0.0; 0.0 10.0] 
    θ1 = [vec(A); B; diag(Qt); Rt; state_initial]
    θt = [vec(A); B; diag(Qt); Rt]  
    θT = [diag(QT); state_goal] 
    parameters = [θ1, [θt for t = 2:horizon-1]..., θT]

    # ## objective 
    function obj1(x, u, w) 
        Q1 = Diagonal(w[6 .+ (1:2)])
        R1 = w[8 + 1]
        return 0.5 * transpose(x) * Q1 * x + 0.5 * R1 * transpose(u) * u
    end

    function objt(x, u, w) 
        Qt = Diagonal(w[6 .+ (1:2)])
        Rt = w[8 + 1]
        return 0.5 * transpose(x) * Qt * x + 0.5 * Rt * transpose(u) * u
    end

    function objT(x, u, w) 
        QT = Diagonal(w[0 .+ (1:2)])
        return 0.5 * transpose(x) * QT * x
    end

    objective = [
                    obj1,
                    [objt for t = 2:horizon-1]...,
                    objT,
    ]

    # ## constraints 
    equality = [
            (x, u, w) -> 1 * (x - w[9 .+ (1:2)]),
            [empty_constraint for t = 2:horizon-1]...,
            (x, u, w) -> 1 * (x - w[2 .+ (1:2)]),
    ]

    # ## options 
    options = Options(
            residual_tolerance=1.0e-12, 
            equality_tolerance=1.0e-8,
            complementarity_tolerance=1.0e-8,
            differentiate=true)

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions;
        parameters=parameters,
        equality=equality,
        options=options);

    # ## initialize
    state_guess = linear_interpolation(state_initial, state_goal, horizon)
    action_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
    initialize_states!(solver, state_guess) 
    initialize_actions!(solver, action_guess)

    # ## solve 
    solve!(solver)

    # ## test 
    num_parameters = [length(p) for p in parameters]
    @test norm(solver.parameters - vcat([θ1, [θt for t = 2:horizon-1]..., θT]...), Inf) < 1.0e-5

    opt_norm = max(
        norm(solver.data.residual.variables, Inf),
        norm(solver.data.residual.cone_slack, Inf),
        # norm(λ - y, Inf),
    )
    @test opt_norm < solver.options.optimality_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

    nz = sum(num_states) + sum(num_actions)
    nz += num_states[1] + num_states[end] # t=1, t=T
    for t = 2:horizon-1 
        nz += num_states[t]
    end
    nz += num_states[end]

    nθ = sum(num_parameters)

    function lagrangian(z, θ)
        x = [z[sum(num_states[1:(t-1)]) + sum(num_actions[1:(t-1)]) .+ (1:num_states[t])] for t = 1:horizon]
        u = [z[sum(num_states[1:t]) + sum(num_actions[1:(t-1)]) .+ (1:num_actions[t])] for t = 1:horizon-1]
        λdyn = [z[sum(num_states) + sum(num_actions) + sum(num_states[1 .+ (1:(t-1))]) .+ (1:num_states[t+1])] for t = 1:horizon-1] 
        λx1 = z[sum(num_states) + sum(num_actions) + sum(num_states[2:end]) .+ (1:num_states[1])] 
        λxT = z[sum(num_states) + sum(num_actions) + sum(num_states[2:end]) + num_states[1] .+ (1:num_states[end])]

        w = [θ[sum(num_parameters[1:(t-1)]) .+ (1:num_parameters[t])] for t = 1:horizon]

        L = 0.0 

        for t = 1:horizon 
            if t == 1
                L += obj1(x[1], u[1], w[1])
                L += transpose(λdyn[1]) * double_integrator(x[2], x[1], u[1], w[1]);
                L += transpose(λx1) * (x[1] - w[1][9 .+ (1:2)])
            elseif t == horizon 
                L += objT(x[horizon], zeros(0), w[horizon]) 
                L += transpose(λxT) * (x[horizon] - w[horizon][2 .+ (1:2)])
            else 
                L += objt(x[t], u[t], w[t]) 
                L += transpose(λdyn[t]) * double_integrator(x[t+1], x[t], u[t], w[t])
            end 
        end

        return L
    end

    @variables zv[1:nz] θv[1:nθ]

    L = lagrangian(zv, θv)
    Lz = Symbolics.gradient(L, zv)
    Lzz = Symbolics.jacobian(Lz, zv)
    Lzθ = Symbolics.jacobian(Lz, θv)

    Lzz_func = eval(Symbolics.build_function(Lzz, zv, θv)[2])
    Lzθ_func = eval(Symbolics.build_function(Lzθ, zv, θv)[2])

    Lzz0 = zeros(nz, nz)
    Lzθ0 = zeros(nz, nθ)

    Lzz_func(Lzz0, [solver.solution.variables; solver.solution.equality_dual], solver.parameters)
    Lzθ_func(Lzθ0, [solver.solution.variables; solver.solution.equality_dual], solver.parameters)

    sensitivity = -1.0 * Lzz0 \ Lzθ0

    @test norm(Lzθ0[solver.indices.variables, :] - solver.problem.objective_jacobian_variables_parameters - solver.problem.equality_dual_jacobian_variables_parameters, Inf) < 1.0e-5
    @test norm(Lzθ0[solver.dimensions.variables .+ (1:solver.dimensions.equality_dual), :] - solver.problem.equality_jacobian_parameters, Inf) < 1.0e-5
    @test norm(sensitivity[solver.indices.variables, :] - solver.data.solution_sensitivity[solver.indices.variables, :], Inf) < 1.0e-3
end