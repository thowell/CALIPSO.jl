@testset "Examples: Rocket landing" begin 
    # ## horizon 
    horizon = 101

    # ## dimensions 
    num_states = [6 for t = 1:horizon]
    num_actions = [3 for t = 1:horizon-1] 

    # ## dynamics
    function rocket_continuous(x, u)
        mass = 1.0
        gravity = -9.81

        p = x[1:3] 
        v = x[4:6] 

        f = u[1:3]

        [
            v; 
            [0.0; 0.0; gravity] + 1.0 / mass * f;
        ]
    end

    function rocket_discrete(y, x, u)
        h = 0.05 # timestep 
        y - (x + h * rocket_continuous(0.5 * (x + y), u))
    end

    dynamics = [rocket_discrete for t = 1:horizon-1] 

    # ## states
    state_initial = [3.0; 2.0; 1.0; 0.0; 0.0; 0.0] 
    state_goal = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 

    # ## objective 
    objective = [
        [(x, u) -> 1.0 * dot(x[1:3] - state_goal[1:3], x[1:3] - state_goal[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
        (x, u) -> 1.0 * dot(x[1:3] - state_goal[1:3], x[1:3] - state_goal[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]),
    ];

    # ## constraints 
    equality = [
            (x, u) -> x - state_initial, 
            [empty_constraint for t = 2:horizon-1]..., 
            (x, u) -> x - state_goal,
    ];

    function thrust_cone(x, u) 
        [
            u[3]; 
            u[1]; 
            u[2];
        ]
    end

    second_order = [
            [[thrust_cone] for t = 1:horizon-1]..., 
            [empty_constraint],
    ];

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        second_order=second_order);

    # ## initialize
    state_guess = linear_interpolation(state_initial, state_goal, horizon)
    action_guess = [1.0e-3 * randn(num_actions[t]) for t = 1:horizon-1]
    initialize_states!(solver, state_guess) 
    initialize_controls!(solver, action_guess)

    # ## solve 
    solve!(solver)

    # ## solution
    x_sol, u_sol = get_trajectory(solver)

    # ## tests
    @test all([norm(u[1:2]) < u[3] for u in u_sol])

    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end