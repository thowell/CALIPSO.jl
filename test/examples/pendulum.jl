@testset "Examples: Pendulum" begin 
    # ## horizon 
    horizon = 11 

    # ## dimensions 
    num_states = [2 for t = 1:horizon]
    num_actions = [1 for t = 1:horizon-1] 

    # ## dynamics
    function pendulum_continuous(x, u)
        mass = 1.0
        length_com = 0.5
        gravity = 9.81
        damping = 0.1

        [
            x[2],
            (u[1] / ((mass * length_com * length_com))
                - gravity * sin(x[1]) / length_com
                - damping * x[2] / (mass * length_com * length_com))
        ]
    end

    function pendulum_discrete(y, x, u)
        h = 0.05 # timestep 
        y - (x + h * pendulum_continuous(0.5 * (x + y), u))
    end

    dynamics = [pendulum_discrete for t = 1:horizon-1] 

    # ## states
    state_initial = [0.0; 0.0] 
    state_goal = [π; 0.0] 

    # ## objective 
    objective = [
        [(x, u) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
        (x, u) -> 0.1 * dot(x[1:2], x[1:2]),
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
    action_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
    initialize_states!(solver, state_guess) 
    initialize_controls!(solver, action_guess)

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
