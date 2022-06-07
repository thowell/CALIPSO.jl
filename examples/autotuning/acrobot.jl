# ## dependencies 
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO
include("models/acrobot.jl")
include("autotuning.jl")

###############
## Reference ##
###############

# ## horizon
horizon = 101
timestep = 0.025

# ## dimensions 
num_states = [4 for t = 1:horizon] 
num_actions = [1 for t = 1:horizon-1] 

# ## dynamics
dynamics = [acrobot_discrete for t = 1:horizon-1]

# ## states
state_initial = [0.0; 0.0; 0.0; 0.0] 
state_goal = [π; 0.0; 0.0; 0.0] 

# ## objective 
objective = [
    [(x, u) -> 1.0 * dot(x[3:4], x[3:4]) + 1.0 * dot(u, u) for t = 1:horizon-1]..., 
    (x, u) -> 1.0 * dot(x[3:4], x[3:4]),
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
action_guess = [0.11 * ones(num_actions[t]) for t = 1:horizon-1]
initialize_states!(solver, state_guess) 
initialize_controls!(solver, action_guess)

# ## options 
solver.options.residual_tolerance=1.0e-6
solver.options.optimality_tolerance=1.0e-6 
solver.options.equality_tolerance=1.0e-6 
solver.options.complementarity_tolerance=1.0e-6
solver.options.slack_tolerance=1.0e-6

# ## solve 
solve!(solver)

# ## solution
state_solution, action_solution = get_trajectory(solver);

# ## visualizer
vis = Visualizer() 
render(vis)

# ## visualize
visualize_acrobot!(vis, nothing, 
    [[state_solution[1] for t = 1:50]..., state_solution..., [state_solution[end] for t = 1:50]...], 
    Δt=timestep)

# ## simulate
state_openloop = [state_initial]
action_openloop = []
for t = 1:horizon-1
    push!(action_openloop, max.(min.(10.0, action_solution[1]), -10.0))
    push!(state_openloop, max.(min.(10.0, acrobot_discrete(state_openloop[end], action_openloop[end])), -10.0))
end
visualize_acrobot!(vis, nothing, 
    [[state_openloop[1] for t = 1:50]..., state_openloop..., [state_openloop[end] for t = 1:50]...], 
    Δt=timestep)

###########
### MPC ###
###########

# ## mpc 
horizon_mpc = 10

# ## dynamics 
acrobot_discrete(y, x, u, w)  = acrobot_discrete(y, x, u)
dynamics_mpc = [acrobot_discrete for t = 1:horizon_mpc-1]

# ## objective 
objective_mpc = [
    [(x, u, w) -> begin 
        x̄ = w[1:4]
        ū = w[4 .+ (1:1)] 
        Q = Diagonal(w[5 .+ (1:4)].^2) 
        R = Diagonal(w[9 .+ (1:1)].^2)
        J = 0.5 * transpose(x - x̄) * Q * (x - x̄) + 0.5 * transpose(u - ū) * R * (u - ū) 
        return J
    end for t = 1:horizon_mpc-1]..., 
    (x, u, w) -> begin 
        x̄ = w[1:4]
        Q = Diagonal(w[4 .+ (1:4)].^2) 
        J = 0.5 * transpose(x - x̄) * Q * (x - x̄) 
    return J
    end,
];

# ## constraints
equality_mpc = [
    (x, u, w) -> x - w[10 .+ (1:4)], 
    [empty_constraint for t = 2:horizon_mpc-1]..., 
    empty_constraint,
];

# ## parameters 
q_cost = ones(4) 
r_cost = ones(1)
qT_cost = ones(4) 

parameters_mpc = [
    [[state_solution[t]; action_solution[t]; q_cost; r_cost; t == 1 ? state_initial : zeros(0)] for t = 1:horizon_mpc-1]...,
    [state_solution[horizon]; qT_cost],
]

# ## options 
options_mpc = Options(
    constraint_tensor=true,
    residual_tolerance=1.0e-3,
    optimality_tolerance=1.0e-3, 
    equality_tolerance=1.0e-3, 
    complementarity_tolerance=1.0e-3,
    slack_tolerance=1.0e-3,
    update_factorization=false,
    differentiate=true,
)

# ## solver 
solver_mpc = Solver(objective_mpc, dynamics_mpc, num_states[1:horizon_mpc], num_actions[1:horizon_mpc-1]; 
    equality=equality_mpc,
    parameters=parameters_mpc,
    options=options_mpc,
);

# ## solve
solve!(solver_mpc)

# ## policy 
state_reference = [state_solution..., [state_solution[horizon] for t = 1:horizon]...]
action_reference = [action_solution..., [action_solution[horizon-1] for t = 1:horizon]...]

initial_state_index = collect(10 .+ (1:4)) 
weight_index = [[collect((t > 1 ? 4 : 0) + (t - 1) * 10 + 5 .+ (1:5)) for t = 1:horizon_mpc-1]..., collect(4 + 10 * (horizon_mpc-1) + 4 .+ (1:4))]

function policy(θ, x, τ)
    # costs
    stage_cost = [θ[(t - 1) * 5 .+ (1:5)] for t = 1:horizon_mpc-1] 
    final_cost = θ[5 * (horizon_mpc - 1) .+ (1:4)]

    # set parameters
    solver_mpc.parameters .= vcat([
        [[state_reference[t + τ - 1]; action_reference[t + τ - 1]; stage_cost[t]; t == 1 ? x : zeros(0)] for t = 1:horizon_mpc-1]...,
        [state_reference[horizon_mpc + τ - 1]; final_cost],
    ]...)

    state_guess = deepcopy(state_reference[τ - 1 .+ (1:horizon_mpc)])
    action_guess = deepcopy(action_reference[τ - 1 .+ (1:horizon_mpc-1)])
    initialize_states!(solver_mpc, state_guess) 
    initialize_controls!(solver_mpc, action_guess)

    solver_mpc.options.verbose = false
    solver_mpc.options.differentiate = false
    solve!(solver_mpc)

    xs, us = get_trajectory(solver_mpc)
    return us[1]
end

function policy_jacobian_parameters(θ, x, τ)
    # costs
    stage_cost = [θ[(t - 1) * 5 .+ (1:5)] for t = 1:horizon_mpc-1] 
    final_cost = θ[5 * (horizon_mpc - 1) .+ (1:4)]

    # set parameters
    solver_mpc.parameters .= vcat([
        [[state_reference[t + τ - 1]; action_reference[t + τ - 1]; stage_cost[t]; t == 1 ? x : zeros(0)] for t = 1:horizon_mpc-1]...,
        [state_reference[horizon_mpc + τ - 1]; final_cost],
    ]...)

    state_guess = deepcopy(state_reference[τ - 1 .+ (1:horizon_mpc)])
    action_guess = deepcopy(action_reference[τ - 1 .+ (1:horizon_mpc-1)])
    initialize_states!(solver_mpc, state_guess) 
    initialize_controls!(solver_mpc, action_guess)

    solver_mpc.options.verbose = false
    solver_mpc.options.differentiate = true

    solve!(solver_mpc)

    return solver_mpc.data.solution_sensitivity[solver_mpc.problem.custom.indices.actions[1], vcat(weight_index...)]
    # xs, us = get_trajectory(solver_mpc)
    # return us[1]
end

function policy_jacobian_state(θ, x, τ)
    # costs
    stage_cost = [θ[(t - 1) * 5 .+ (1:5)] for t = 1:horizon_mpc-1] 
    final_cost = θ[5 * (horizon_mpc - 1) .+ (1:4)]

    # set parameters
    solver_mpc.parameters .= vcat([
        [[state_reference[t + τ - 1]; action_reference[t + τ - 1]; stage_cost[t]; t == 1 ? x : zeros(0)] for t = 1:horizon_mpc-1]...,
        [state_reference[horizon_mpc + τ - 1]; final_cost],
    ]...)

    state_guess = deepcopy(state_reference[τ - 1 .+ (1:horizon_mpc)])
    action_guess = deepcopy(action_reference[τ - 1 .+ (1:horizon_mpc-1)])
    initialize_states!(solver_mpc, state_guess) 
    initialize_controls!(solver_mpc, action_guess)

    solver_mpc.options.verbose = false
    solver_mpc.options.differentiate = true

    solve!(solver_mpc)

    return solver_mpc.data.solution_sensitivity[solver_mpc.problem.custom.indices.actions[1], initial_state_index]
end


# ## dynamics 
rollout_dynamics(x, u, t) = acrobot_discrete(x, u)
dynamics_jacobian_state(x, u, t) = FiniteDiff.finite_difference_jacobian(a -> rollout_dynamics(a, u, t), x)
dynamics_jacobian_action(x, u, t) = FiniteDiff.finite_difference_jacobian(a -> rollout_dynamics(x, a, t), u)

# ## tuning metric
state_cost = [t == horizon ? Diagonal([10.0; 10.0; 10.0; 10.0]) : Diagonal([1.0; 1.0; 1.0; 1.0]) for t = 1:horizon]
action_cost = [Diagonal([1.0e-1]) for t = 1:horizon-1]

# ## policy initialization
parameters_cost = vcat([[[1.0; 1.0; 1.0; 1.0; 1.0] for t = 1:horizon_mpc-1]..., [1.0; 1.0; 1.0; 1.0]]...)

# ## untuned policy rollout
state_untuned, action_untuned = rollout(state_initial, parameters_cost, horizon)
@show total_loss(state_untuned, action_untuned, state_reference, action_reference, state_cost, action_cost)

# ## visualize untuned
visualize_acrobot!(vis, nothing, 
    [[state_untuned[1] for t = 1:50]..., state_untuned..., [state_untuned[end] for t = 1:50]...], 
    Δt=timestep)

# ## autotune!
autotune!(parameters_cost, state_reference, action_reference, state_cost, action_cost, horizon)

# ## tuned policy rollout
state_tuned, action_tuned = rollout(state_initial, parameters_cost, horizon)
@show total_loss(state_tuned, action_tuned, state_reference, action_reference, state_cost, action_cost)

# ## visualize tuned
visualize_acrobot!(vis, nothing, 
    [[state_tuned[1] for t = 1:50]..., state_tuned..., [state_tuned[end] for t = 1:50]...], 
    Δt=timestep)
