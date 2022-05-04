@testset "Examples: Box move (nonlinear friction cone)" begin 
    # ## horizon
    horizon = 11
    timestep = 0.1

    # ## RoboDojo dynamics 
    include("robodojo.jl")
    
    model = RoboDojo.box
    sim = RoboDojo.Simulator(model, 1, 
        h=timestep)

    # ## dimensions
    num_states, num_actions = state_action_dimensions(sim, horizon)

    # ## dynamics
    dynamics = [(y, x, u) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon-1]

    # ## states
    q1 = [0.0; 0.5; 0.0]
    qT = [1.0; 0.5; 0.0]

    state_initial = [q1; q1]
    state_goal = [qT; qT]
    state_reference = [qT; qT]

    # ## objective
    function obj1(x, u)
        J = 0.0
        J += 0.5 * transpose(x - state_reference) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_reference)
        J += 0.5 * transpose(u) * Diagonal(1.0e-2 * ones(3)) * u
        return J
    end

    function objt(x, u)
        J = 0.0
        J += 0.5 * transpose(x[1:6] - state_reference) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:6] - state_reference)
        J += 0.5 * transpose(u) * Diagonal(1.0e-2 * ones(3)) * u
        return J
    end

    function objT(x, u)
        J = 0.0
        J += 0.5 * transpose(x[1:6] - state_reference) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:6] - state_reference)
        return J
    end

    objective = [
            obj1, 
            [objt for t = 2:horizon-1]..., 
            objT,
    ];

    # ## constraints
    equality = [
            (x, u) -> x - state_initial, 
            [empty_constraint for t = 2:horizon-1]..., 
            (x, u) -> x[1:6] - state_goal,
    ];

    nonnegative = robodojo_nonnegative(sim, horizon);
    second_order = robodojo_second_order(sim, horizon);

    # ## options 
    options = Options(
        constraint_tensor=true,
        penalty_initial=1.0,
    )

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        nonnegative=nonnegative,
        second_order=second_order,
        options=options);

    # ## initialize
    configurations = CALIPSO.linear_interpolation(q1, q1, horizon+1)
    state_guess = robodojo_state_initialization(sim, configurations, horizon)
    action_guess = [0.0 * randn(sim.model.nu) for t = 1:horizon-1] # may need to run more than once to get good trajectory
    initialize_states!(solver, state_guess) 
    initialize_controls!(solver, action_guess)

    # ## solve
    solve!(solver)

    # ## solution
    x_sol, u_sol = CALIPSO.get_trajectory(solver)

    # test solution
    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
    # end

    # ## visualize
    vis = Visualizer()
    render(vis)
    RoboDojo.visualize!(vis, RoboDojo.box, x_sol, Δt=timestep, r=0.5);


end

# # ## visualize
# vis = Visualizer()
# render(vis)
# RoboDojo.visualize!(vis, RoboDojo.box, x_sol, Δt=timestep, r=0.5);

