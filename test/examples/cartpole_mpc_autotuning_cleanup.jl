# @testset "Examples: Cart-pole" begin 
# ## horizon 
horizon = 51
timestep = 0.05

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
    h = timestep # timestep 
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
    [(x, u) -> 0.5 * 1.0e-2 * dot(x - state_goal, x - state_goal) + 0.5 * 1.0e-1 * dot(x[3:4], x[3:4]) + 0.5 * 1.0e-1  * dot(u, u) for t = 1:horizon-1]..., 
    (x, u) -> 0.5 * 1.0e2 * dot(x - state_goal, x - state_goal) + 0.5 * 1.0e-1 * dot(x[3:4], x[3:4]),
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
initialize_controls!(solver, action_guess)

# ## solve 
solve!(solver)

# ## solution
state_reference, action_reference = get_trajectory(solver);

@test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

slack_norm = max(
                norm(solver.data.residual.equality_dual, Inf),
                norm(solver.data.residual.cone_dual, Inf),
)
@test slack_norm < solver.options.slack_tolerance

@test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
@test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# end

using RoboDojo
function cable_transform(y, z)
    v1 = [0.0, 0.0, 1.0]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1, v2)
    ang = acos(v1'*v2)
    R = RoboDojo.AngleAxis(ang, ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    RoboDojo.compose(RoboDojo.RoboDojo.Translation(z), RoboDojo.RoboDojo.LinearMap(R))
end

function default_background!(vis)
    RoboDojo.RoboDojo.setvisible!(vis["/Background"], true)
    RoboDojo.setprop!(vis["/Background"], "top_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setprop!(vis["/Background"], "bottom_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.RoboDojo.setvisible!(vis["/Axes"], false)
end

function _create_cartpole!(vis, model;
    i = 0,
    tl = 1.0,
    color = RoboDojo.Colors.RGBA(0, 0, 0, tl))

    l2 = RoboDojo.Cylinder(RoboDojo.Point3f0(-0.5 * 10.0, 0.0, 0.0),
        RoboDojo.Point3f0(0.5 * 10.0, 0.0, 0.0),
        convert(Float32, 0.0125))

    RoboDojo.setobject!(vis["slider_$i"], l2, RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))

    l1 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0, 0.0, 0.0),
        RoboDojo.Point3f0(0.0, 0.0, 0.5),
        convert(Float32, 0.025))

    RoboDojo.setobject!(vis["arm_$i"], l1,
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))

    RoboDojo.setobject!(vis["base_$i"], RoboDojo.HyperSphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.1)),
        RoboDojo.MeshPhongMaterial(color = color))

    RoboDojo.setobject!(vis["ee_$i"], RoboDojo.HyperSphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = color))
end

function _set_cartpole!(vis, model, x;
    i = 0)

    px = x[1] + 0.5 * sin(x[2])
    pz = -0.5 * cos(x[2])
    RoboDojo.settransform!(vis["arm_$i"], cable_transform([x[1]; 0;0], [px; 0.0; pz]))
    RoboDojo.settransform!(vis["base_$i"], RoboDojo.Translation([x[1]; 0.0; 0.0]))
    RoboDojo.settransform!(vis["ee_$i"], RoboDojo.Translation([px; 0.0; pz]))
end

function visualize_cartpole!(vis, model, q;
    i = 0,
    tl = 1.0,
    Δt = 0.1,
    color = RoboDojo.Colors.RGBA(0,0,0,1.0))

    default_background!(vis)
    _create_cartpole!(vis, model, i = i, color = color, tl = tl)

    anim = RoboDojo.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        RoboDojo.MeshCat.atframe(anim,t) do
            _set_cartpole!(vis, model, q[t], i = i)
        end
    end

    RoboDojo.settransform!(vis["/Cameras/default"],
        RoboDojo.compose(RoboDojo.Translation(0.0, 0.0, -1.0), RoboDojo.LinearMap(RoboDojo.RotZ(- pi / 2))))
    RoboDojo.RoboDojo.setvisible!(vis["/Grid"], false)

    RoboDojo.MeshCat.setanimation!(vis,anim)
end

vis = Visualizer() 
render(vis)
visualize_cartpole!(vis, nothing, state_reference, Δt=timestep)

# ## simulate
x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1
    push!(u_hist, action_reference[1])
    push!(x_hist, cartpole_discrete(x_hist[end], u_hist[end]))
end
visualize_cartpole!(vis, nothing, x_hist, Δt=timestep)
cost_initial = cartpole_cost(x_hist, u_hist, Q, R)

[cartpole_discrete(state_reference[t], action_reference[t]) - state_reference[t+1] for t = 1:horizon-1]

# LQR 
using FiniteDiff
A = [FiniteDiff.finite_difference_jacobian(a -> cartpole_discrete(a, action_reference[t]), state_reference[t]) for t = 1:horizon-1]
B = [FiniteDiff.finite_difference_jacobian(a -> cartpole_discrete(state_reference[t], a), action_reference[t]) for t = 1:horizon-1]
Q = [t == horizon ? Diagonal([1000.0; 1000.0; 1000.0; 1000.0]) : Diagonal([10.0; 10.0; 1.0; 1.0]) for t = 1:horizon]
R = [Diagonal([1.0e-1]) for t = 1:horizon-1]

# Q = [t == horizon ? Diagonal([1.0; 1.0; 1.0; 1.0]) : Diagonal([1.0; 1.0; 1.0; 1.0]) for t = 1:horizon]
# R = [Diagonal([1.0]) for t = 1:horizon-1]


function tvlqr(A, B, Q, R)
    T = length(Q)

    P = [zero(A[1]) for t = 1:T]
    K = [zero(B[1]') for t = 1:T-1]
    P[T] = Q[T]

    for t = T-1:-1:1
        K[t] = (R[t] + B[t]' * P[t+1] *  B[t]) \ (B[t]' * P[t+1] * A[t])
        P[t] = (Q[t] + K[t]' * R[t] * K[t]
                + (A[t] - B[t] * K[t])' * P[t+1] * (A[t] - B[t] * K[t]))
    end

    return K, P
end

K, P = tvlqr(A, B, Q, R)

x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1
    push!(u_hist, action_reference[1] - K[t] * (x_hist[t] - state_reference[t]))
    push!(x_hist, cartpole_discrete(x_hist[end], u_hist[end]))
end
visualize_cartpole!(vis, nothing, x_hist, Δt=timestep)
cost_initial = cartpole_cost(x_hist, u_hist, Q, R)

# ## mpc 
horizon_mpc = 10

# dynamics 
cartpole_discrete(y, x, u, w)  = cartpole_discrete(y, x, u)
dynamics_mpc = [cartpole_discrete for t = 1:horizon_mpc-1]

# objective 
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

# constraints
equality_mpc = [
    (x, u, w) -> x - w[10 .+ (1:4)], 
    [empty_constraint for t = 2:horizon_mpc-1]..., 
    empty_constraint,
];

# parameters 
q_cost = ones(4) 
r_cost = ones(1)
qT_cost = ones(4) 

parameters_mpc = [
    [[state_reference[t]; action_reference[t]; q_cost; r_cost; t == 1 ? state_initial : zeros(0)] for t = 1:horizon_mpc-1]...,
    [state_reference[horizon]; qT_cost],
]

# options 
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

solve!(solver_mpc)

# policy 
state_reference = [state_reference..., [state_reference[horizon] for t = 1:horizon]...]
action_reference = [action_reference..., [action_reference[horizon-1] for t = 1:horizon]...]

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
    # xs, us = get_trajectory(solver_mpc)
    # return us[1]
end

t = 1
parameters_cost = vcat([[[1.0; 1.0; 1.0; 1.0; 1.0] for t = 1:horizon_mpc-1]..., [1.0; 1.0; 1.0; 1.0]]...)
@show policy(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)
@show policy_jacobian_parameters(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)
@show policy_jacobian_state(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)

@show action_reference[t]

x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1 
    push!(u_hist, policy(parameters_cost, x_hist[end], t))
    push!(x_hist, cartpole_discrete(x_hist[end], u_hist[end]))
end
visualize_cartpole!(vis, nothing, x_hist, Δt=timestep)

# function cartpole_cost(X, U, Q, R) 
#     J = 0.0 
#     H = length(X)
#     for t = 1:H-1 
#         J += 0.5 * (X[t] - state_reference[t])' * Q[t] * (X[t] - state_reference[t])
#         J += 0.5 * (U[t] - action_reference[t])' * R[t] * (U[t] - action_reference[t])
#     end 
#     J += 0.5 * (X[H] - state_reference[H])' * Q[H] * (X[H] - state_reference[H])

#     return J 
# end

# cost_initial = cartpole_cost(x_hist, u_hist, Q, R)

# ## learning
function loss(state, action, state_reference, action_reference, state_cost, action_cost)
    J = 0.0 
    J += 0.5 * (state - state_reference)' * state_cost * (state - state_reference)
    J += 0.5 * (action - action_reference)' * action_cost * (action - action_reference)
    return J 
end

function loss_gradient_state(state, action, state_reference, action_reference, state_cost, action_cost)
    Jx = state_cost * (state - state_reference)
    return Jx
end

function loss_gradient_action(state, action, state_reference, action_reference, state_cost, action_cost)
    Ju = action_cost * (action - action_reference)
    return Ju
end

function cost(states, actions, state_references, action_references, state_costs, action_costs)
    horizon = length(states)

    J = 0.0

    for t = 1:horizon-1
        J += loss(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t])
    end

    J += loss(states[horizon], zeros(0), state_references[horizon], zeros(0), state_costs[horizon], zeros(0, 0))

    return J ./ horizon
end

loss(x_hist[1], u_hist[1], state_reference[1], action_reference[1], Q[1], R[1])
loss_gradient_state(x_hist[1], u_hist[1], state_reference[1], action_reference[1], Q[1], R[1])
loss_gradient_action(x_hist[1], u_hist[1], state_reference[1], action_reference[1], Q[1], R[1])
cost(x_hist, u_hist, state_reference, action_reference, Q, R)
loss(x_hist[horizon], zeros(0), state_reference[horizon], zeros(0), Q[horizon], zeros(0, 0))

function loss_gradient_parameters(states, actions, parameters, state_references, action_references, state_costs, action_costs) 
    # dimensions
    state_dim = length(states[1])
    parameters_dim = length(vcat(parameters...))
    horizon = length(states) 

    # initialize gradient
    Jθ = zeros(parameters_dime)
    
    # initialize Jacobians
    ∂x∂θ = [zeros(state_dim, parameters_dim)]
    ∂u∂θ = []
    ∂π∂x = [] 
    ∂π∂θ = []

    for t = 1:horizon-1
        # ∂u∂θ
        push!(∂π∂x, policy_jacobian_state(parameters[t], states[t], t))
        push!(∂π∂θ, policy_jacobian_parameters(parameters[t], states[t], t)) 
        
        # ∂f∂x, ∂f∂u
        ∂f∂x = dynamics_jacobian_state(states[t], actions[t])
        ∂f∂u = dynamics_jacobian_action(states[t], actions[t])

        push!(∂u∂θ, ∂π∂x[end] * ∂x∂θ[end] + ∂π∂θ[end])
        push!(∂x∂θ, ∂f∂x * ∂x∂θ[end] + ∂f∂u * ∂u∂θ[end])
    end

    for t = 1:horizon-1
        Jθ += ∂x∂θ[t]' * loss_gradient_state(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t]) 
        Jθ += ∂u∂θ[t]' * loss_gradient_action(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t])
    end 

    Jθ += ∂x∂θ[horizon]' * loss_gradient_state(states[horizon], zeros(0), state_reference[horizon], zeros(0), state_costs[horizon], zeros(0, 0))

    return Jθ ./ horizon
end

ψθ(x_hist, u_hist, parameters_cost)

function simulate(x0, θ, T) 
    X = [x0] 
    U = [] 
    for t = 1:T-1 
        push!(U, policy(θ, X[end], t))
        # push!(W, noise * randn(n)) 
        push!(X, cartpole_discrete(X[end], U[end]))
    end 
    return X, U
end

simulate(state_initial, parameters_cost, horizon)

# number of evaluations 
# initial policy
θ = parameters_cost
J_opt = 0.0
for i = 1:1
    x0 = state_initial #noise * randn(n)
    X, U = simulate(x0, θ, horizon)
    J_opt += ψ(X, U)
end
@show J_opt

X, U = simulate(state_initial, θ, horizon)

# @test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < noise
# plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
ψθ(X, U, θ)
c = [J_opt] 

for i = 1:10
    if i == 1 
        println("iteration: $(i)") 
        println("cost: $(c[end])") 
    end
    # i == 250 && (α *= 0.1) 
    # i == 500 && (α *= 0.1) 


    # train
    J = 0.0 
    Jθ = zeros(length(θ))
    for j = 1:1
        X, U = simulate(state_initial, θ, horizon)
        J += ψ(X, U)
        Jθ += ψθ(X, U, θ)
    end
     
    J_cand = Inf
    θ_cand = zero(θ)
    α = 1.0
    iter = 0

    while J_cand >= J 
        θ_cand = θ - α * Jθ
        J_cand = 0.0
        for j = 1:1
            X, U = simulate(state_initial, θ_cand, horizon)
            J_cand += ψ(X, U)
        end
        α = 0.5 * α
        iter += 1 
        iter > 25 && error("failure") 
    end

    J = J_cand 
    θ = θ_cand

    norm(Jθ, Inf) < 1.0e-2 && break

    # evaluate
    if i % 10 == 0
        J_eval = 0.0 
        for k = 1:1 
            X, U = simulate(state_initial, θ, horizon) 
            J_eval += ψ(X, U)
        end
        push!(c, J_eval)
        println("iteration: $(i)") 
        println("cost: $(c[end])")
    end
end

using Plots
plot(c, xlabel="iteration", ylabel="cost")

x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1 
    push!(u_hist, policy(θ, x_hist[end], t))
    push!(x_hist, cartpole_discrete(x_hist[end], u_hist[end]))
end

vis = Visualizer() 
render(vis)
visualize_cartpole!(vis, nothing, x_hist, Δt=timestep)
cost_initial = cartpole_cost(x_hist, u_hist, Q, R)
