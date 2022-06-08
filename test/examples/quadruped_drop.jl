@testset "Examples: Quadruped drop" begin
    # ## horizon
    horizon = 6
    timestep = 0.1

    # ## RoboDojo dynamics 
    include("robodojo.jl")

    model = RoboDojo.quadruped4 # NOTE: all contacts turned on
    sim = RoboDojo.Simulator(model, 1, 
        h=timestep)

    # ## dimensions
    num_states, num_actions = state_action_dimensions(sim, horizon)

    # ## dynamics
    dynamics = [(y, x, u) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon-1]

    # ## states
    function initial_configuration(model::RoboDojo.Quadruped4, θ1, θ2, θ3)
        q1 = zeros(11)
        q1[3] = 0.0
        q1[4] = -θ1
        q1[5] = θ2

        q1[8] = -θ1
        q1[9] = θ2

        q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

        q1[10] = -θ3
        q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

        q1[6] = -θ3
        q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

        return q1
    end

    θ1 = pi / 4.0
    θ2 = pi / 4.0
    θ3 = pi / 3.0

    q1 = initial_configuration(model, θ1, θ2, θ3)
    q1[2] += 0.25
    q1[3] += 0.2
    qT = initial_configuration(model, θ1, θ2, θ3)

    # ## objective
    function obj1(x, u)
        u_ctrl = u[1:model.nu]
        q = x[model.nq .+ (1:model.nq)]

        J = 0.0 
        v = q - x[collect(1:model.nq)]
        J += 1.0 * dot(v, v)
        J += 1.0e-3 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end

    function objt(x, u)
        u_ctrl = u[1:model.nu]
        q = x[model.nq .+ (1:model.nq)]

        J = 0.0 
        v = q - x[collect(1:model.nq)]
        J += 1.0 * dot(v, v)
        J += 1.0e-3 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end

    function objT(x, u)
        q = x[model.nq .+ (1:model.nq)]

        J = 0.0 
        v = q - x[collect(1:model.nq)]
        J += 1.0 * dot(v, v)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end

    objective = [
        obj1, 
        [objt for t = 2:horizon-1]..., 
        objT,
    ];

    # control limits
    equality = [
        (x, u) -> [x[1:11] - q1; x[11 .+ (1:11)] - q1; u[1:3]], 
        [(x, u) -> u[1:3] for t = 2:horizon-1]..., 
        empty_constraint,
    ];

    nonnegative = robodojo_nonnegative(sim, horizon) 
    second_order = robodojo_second_order(sim, horizon)

    # ## options 
    options = Options(
            verbose=true,        
            constraint_tensor=true,
            update_factorization=false,
    )

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        nonnegative=nonnegative,
        second_order=second_order,
        options=options,
    );

    # ## initialize
    configurations = CALIPSO.linear_interpolation(q1, q1, horizon+1)
    state_guess = robodojo_configuration_initialization(sim, configurations, horizon)
    action_guess = [1.0e-3 * randn(model.nu) for t = 1:horizon-1]
    initialize_states!(solver, state_guess)
    initialize_actions!(solver, action_guess)

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
end

# ## visualize 
# vis = Visualizer() 
# render(vis)
# RoboDojo.visualize!(vis, model, x_sol, Δt=timestep);