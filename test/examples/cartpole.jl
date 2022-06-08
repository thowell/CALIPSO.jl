@testset "Examples: Cart-pole" begin 
    # ## horizon 
    horizon = 51

    # ## dimensions 
    num_states = [4 for t = 1:horizon]
    num_actions = [1 for t = 1:horizon-1]

    # ## dynamics
    function cartpole_continuous(x, u)
        mc = 1.0 
        mp = 0.2 
        l = 0.5 
        g = 9.81 

        q = x[1:2]
        qd = x[3:4]

        s = sin(q[2])
        c = cos(q[2])

        H = [mc + mp mp * l * c; mp * l * c mp * l^2]
        Hinv = 1.0 / (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) * [H[2, 2] -H[1, 2]; -H[2, 1] H[1, 1]]
        
        C = [0 -mp * qd[2] * l * s; 0 0]
        G = [0, mp * g * l * s]
        B = [1, 0]

        qdd = -Hinv * (C * qd + G - B * u[1])

        return [qd; qdd]
    end

    function cartpole_discrete(x, u)
        h = 0.05 # timestep 
        x + h * cartpole_continuous(x + 0.5 * h * cartpole_continuous(x, u), u)
    end

    function cartpole_discrete(y, x, u)
        y - cartpole_discrete(x, u)
    end

    dynamics = [cartpole_discrete for t = 1:horizon-1] 

    # ## states
    state_initial = [0.0; 0.0; 0.0; 0.0] 
    state_goal = [0.0; π; 0.0; 0.0] 

    # ## objective 
    objective = [
        [(x, u) -> 0.5 * 1.0e-2 * dot(x - state_goal, x - state_goal) + 0.5 * 1.0e-1  * dot(u, u) for t = 1:horizon-1]..., 
        (x, u) -> 0.5 * 1.0e2 * dot(x - state_goal, x - state_goal),
    ];

    # ## constraints
    equality = [
        (x, u) -> x - state_initial, 
        [empty_constraint for t = 2:horizon-1]..., 
        (x, u) -> x - state_goal,
    ];

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality);

    # ## initialize
    state_guess = linear_interpolation(state_initial, state_goal, horizon)
    action_guess = [0.01 * ones(num_actions[t]) for t = 1:horizon-1]
    initialize_states!(solver, state_guess) 
    initialize_actions!(solver, action_guess)

    # ## solve 
    solve!(solver)

    # ## solution
    state_solution, action_solution = get_trajectory(solver);

    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end

