# @testset "Examples: Acrobot" begin 
# ## horizon
horizon = 101
timestep = 0.025

# ## dimensions 
num_states = [4 for t = 1:horizon] 
num_actions = [1 for t = 1:horizon-1] 

# ## dynamics
function acrobot_continuous(x, u)
    mass1 = 1.0  
    inertia1 = 0.33  
    length1 = 1.0 
    lengthcom1 = 0.5 

    mass2 = 1.0  
    inertia2 = 0.33  
    length2 = 1.0 
    lengthcom2 = 0.5 

    gravity = 9.81 
    friction1 = 0.25
    friction2 = 0.25

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

    function B(x)
        [0.0; 1.0]
    end

    q = x[1:2]
    v = x[3:4]

    qdd = Minv(q) * (-1.0 * C(x) * v
            + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

function acrobot_discrete(x, u)
    h = timestep # timestep 
    x + h * acrobot_continuous(x + 0.5 * h * acrobot_continuous(x, u), u)
end

function acrobot_discrete(y, x, u)
    y - acrobot_discrete(x, u)
end

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
initialize_actions!(solver, action_guess)

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

    RoboDojo.compose(RoboDojo.Translation(z), RoboDojo.LinearMap(R))
end

function default_background!(vis)
    RoboDojo.setvisible!(vis["/Background"], true)
    RoboDojo.setprop!(vis["/Background"], "top_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setprop!(vis["/Background"], "bottom_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setvisible!(vis["/Axes"], false)
end

# visualization
function _create_acrobot!(vis, model;
    tl = 1.0,
    limit_color = RoboDojo.Colors.RGBA(0.0, 1.0, 0.0, tl),
    color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl),
    i = 0,
    r = 0.1)

    l1 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0, 0.0, 0.0), RoboDojo.Point3f0(0.0, 0.0, 1.0),
        convert(Float32, 0.025))
    RoboDojo.setobject!(vis["l1_$i"], l1, RoboDojo.MeshPhongMaterial(color = color))
    l2 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0,0.0,0.0), RoboDojo.Point3f0(0.0, 0.0, 1.0),
        convert(Float32, 0.025))
    RoboDojo.setobject!(vis["l2_$i"], l2, RoboDojo.MeshPhongMaterial(color = color))

    RoboDojo.setobject!(vis["elbow_nominal_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))
    RoboDojo.setobject!(vis["elbow_limit_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = limit_color))
    RoboDojo.setobject!(vis["ee_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))
end

function kinematics(model, x)
    [1.0 * sin(x[1]) + 1.0 * sin(x[1] + x[2]),
     -1.0 * 1.0 * cos(x[1]) - 1.0 * cos(x[1] + x[2])]
end

function _set_acrobot!(vis, model, x;
    i = 0, ϵ = 1.0e-1)

    p_mid = [kinematics_elbow(model, x)[1], 0.0, kinematics_elbow(model, x)[2]]
    p_ee = [kinematics(model, x)[1], 0.0, kinematics(model, x)[2]]

    RoboDojo.settransform!(vis["l1_$i"], cable_transform(zeros(3), p_mid))
    RoboDojo.settransform!(vis["l2_$i"], cable_transform(p_mid, p_ee))

    RoboDojo.settransform!(vis["elbow_nominal_$i"], RoboDojo.Translation(p_mid))
    RoboDojo.settransform!(vis["elbow_limit_$i"], RoboDojo.Translation(p_mid))
    RoboDojo.settransform!(vis["ee_$i"], RoboDojo.Translation(p_ee))

    if x[2] <= -0.5 * π + ϵ || x[2] >= 0.5 * π - ϵ
        RoboDojo.setvisible!(vis["elbow_nominal_$i"], false)
        RoboDojo.setvisible!(vis["elbow_limit_$i"], true)
    else
        RoboDojo.setvisible!(vis["elbow_nominal_$i"], true)
        RoboDojo.setvisible!(vis["elbow_limit_$i"], false)
    end
end

function kinematics_elbow(model, x)
    [1.0 * sin(x[1]),
     -1.0 * 1.0 * cos(x[1])]
end

# visualization
function visualize_elbow!(vis, model, x;
    tl = 1.0,
    i = 0,
    color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, 1.0),
    limit_color = RoboDojo.Colors.RGBA(0.0, 1.0, 0.0, tl),
    r = 0.1, Δt = 0.1,
    ϵ = 1.0e-1)

    default_background!(vis)
    _create_acrobot!(vis, model,
        tl = tl,
        color = color,
        limit_color = limit_color,
        i = i,
        r = r)

    anim = RoboDojo.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T
        RoboDojo.MeshCat.atframe(anim,t) do
            _set_acrobot!(vis, model, x[t], i = i, ϵ = ϵ)
        end
    end

    RoboDojo.settransform!(vis["/Cameras/default"],
       RoboDojo.compose(RoboDojo.Translation(0.0 , 0.0 , 0.0), RoboDojo.LinearMap(RoboDojo.RotZ(pi / 2.0))))

    RoboDojo.MeshCat.setanimation!(vis, anim)
end

vis = Visualizer() 
render(vis)
visualize_elbow!(vis, nothing, state_solution, Δt=timestep)

# ## simulate
x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1
    push!(u_hist, max.(min.(10.0, action_solution[1]), -10.0))
    push!(x_hist, max.(min.(10.0, acrobot_discrete(x_hist[end], u_hist[end])), -10.0))
end
visualize_elbow!(vis, nothing, x_hist, Δt=timestep)
cost_initial = acrobot_cost(x_hist, u_hist, Q, R)

[acrobot_discrete(state_solution[t], action_solution[t]) - state_solution[t+1] for t = 1:horizon-1]

# LQR 
using FiniteDiff
A = [FiniteDiff.finite_difference_jacobian(a -> acrobot_discrete(a, action_solution[t]), state_solution[t]) for t = 1:horizon-1]
B = [FiniteDiff.finite_difference_jacobian(a -> acrobot_discrete(state_solution[t], a), action_solution[t]) for t = 1:horizon-1]
Q = [t == horizon ? Diagonal([10.0; 10.0; 10.0; 10.0]) : Diagonal([1.0; 1.0; 1.0; 1.0]) for t = 1:horizon]
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
    push!(u_hist, max.(min.(10.0, action_solution[1] - K[t] * (x_hist[t] - state_solution[t])), -10.0))
    push!(x_hist, max.(min.(10.0, acrobot_discrete(x_hist[end], u_hist[end])), -10.0))
end
visualize_elbow!(vis, nothing, x_hist, Δt=timestep)

cost_initial = acrobot_cost(x_hist, u_hist, Q, R)

# ## mpc 
horizon_mpc = 10

# dynamics 
acrobot_discrete(y, x, u, w)  = acrobot_discrete(y, x, u)
dynamics_mpc = [acrobot_discrete for t = 1:horizon_mpc-1]

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
    [[state_solution[t]; action_solution[t]; q_cost; r_cost; t == 1 ? state_initial : zeros(0)] for t = 1:horizon_mpc-1]...,
    [state_solution[horizon]; qT_cost],
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
    initialize_actions!(solver_mpc, action_guess)

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
    initialize_actions!(solver_mpc, action_guess)

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
    initialize_actions!(solver_mpc, action_guess)

    solver_mpc.options.verbose = false
    solver_mpc.options.differentiate = true

    solve!(solver_mpc)

    return solver_mpc.data.solution_sensitivity[solver_mpc.problem.custom.indices.actions[1], initial_state_index]
    # xs, us = get_trajectory(solver_mpc)
    # return us[1]
end

parameters_cost = vcat([[[1.0; 1.0; 1.0; 1.0; 1.0] for t = 1:horizon_mpc-1]..., [1.0; 1.0; 1.0; 1.0]]...)
@show policy(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)
@show policy_jacobian_parameters(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)
@show policy_jacobian_state(parameters_cost, state_reference[t] + 1.0e-3 * randn(4), t)

@show action_reference[t]

x_hist = [state_initial]
u_hist = []
for t = 1:horizon-1 
    push!(u_hist, policy(parameters_cost, x_hist[end], t))
    push!(x_hist, acrobot_discrete(x_hist[end], u_hist[end]))
end
visualize_elbow!(vis, nothing, x_hist, Δt=timestep)

cost_initial = acrobot_cost(x_hist, u_hist, Q, R)


function acrobot_cost(X, U, Q, R) 
    J = 0.0 
    H = length(X)
    for t = 1:H-1 
        J += 0.5 * (X[t] - state_reference[t])' * Q[t] * (X[t] - state_reference[t])
        J += 0.5 * (U[t] - action_reference[t])' * R[t] * (U[t] - action_reference[t])
    end 
    J += 0.5 * (X[H] - state_reference[H])' * Q[H] * (X[H] - state_reference[H])

    return J 
end

cost_initial = acrobot_cost(x_hist, u_hist, Q, R)

# ## learning
function ψt(x, u, xr, ur, q, r)
    J = 0.0 
    J += 0.5 * (x - xr)' * q * (x - xr)
    J += 0.5 * (u - ur)' * r * (u - ur)
    return J 
end

ψtx(x, u, xr, ur, q, r) = FiniteDiff.finite_difference_gradient(z -> ψt(z, u, xr, ur, q, r), x)
ψtu(x, u, xr, ur, q, r) = FiniteDiff.finite_difference_gradient(z -> ψt(x, z, xr, ur, q, r), u)

function ψT(x, xr, q)
    J = 0.0 
    J += 0.5 * (x - xr)' * q * (x - xr)
    return J 
end

ψTx(x, xr, q) = FiniteDiff.finite_difference_gradient(z -> ψT(z, xr, q), x)

function ψ(X, U)
    J = 0.0 
    for t = 1:horizon-1
        J += ψt(X[t], U[t], state_reference[t], action_reference[t], Q[t], R[t])
    end
    J += ψT(X[horizon], state_reference[horizon], Q[horizon])
    return J ./ horizon
end

ψ(x_hist, u_hist)

fx(x, u) = FiniteDiff.finite_difference_jacobian(z -> acrobot_discrete(z, u), x)
fu(x, u) = FiniteDiff.finite_difference_jacobian(z -> acrobot_discrete(x, z), u)

function ψθ(X, U, θ) 
    Jθ = zeros(length(θ)) 

    ∂x∂θ = [zeros(4, length(vcat(parameters_cost...)))]
    ∂u∂θ = []
    ∂ϕ∂x = [] 
    ∂ϕ∂θ = []
    for t = 1:horizon-1
        # ∂u∂θ
        push!(∂ϕ∂x, policy_jacobian_state(θ, X[t], t))
        push!(∂ϕ∂θ, policy_jacobian_parameters(θ, X[t], t)) 
        # ∂f∂x, ∂f∂u
        ∂f∂x = fx(X[t], U[t])
        ∂f∂u = fu(X[t], U[t])
        push!(∂u∂θ, ∂ϕ∂x[end] * ∂x∂θ[end] + ∂ϕ∂θ[end])
        push!(∂x∂θ, ∂f∂x * ∂x∂θ[end] + ∂f∂u * ∂u∂θ[end])
    end

    for t = 1:horizon-1
        Jθ += ∂x∂θ[t]' * ψtx(X[t], U[t], state_reference[t], action_reference[t], Q[t], R[t]) 
        Jθ += ∂u∂θ[t]' * ψtu(X[t], U[t], state_reference[t], action_reference[t], Q[t], R[t])
    end 
    Jθ += ∂x∂θ[horizon]' * ψTx(X[horizon], state_reference[horizon], Q[horizon])
    return Jθ ./ horizon
end

ψθ(x_hist, u_hist, parameters_cost)

function simulate(x0, θ, T) 
    X = [x0] 
    U = [] 
    for t = 1:T-1 
        push!(U, policy(θ, X[end], t))
        # push!(W, noise * randn(n)) 
        push!(X, acrobot_discrete(X[end], U[end]))
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

    norm(Jθ, Inf) < 1.0e-4 && break

    # evaluate
    if i % 1 == 0
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
    push!(x_hist, acrobot_discrete(x_hist[end], u_hist[end]))
end

vis = Visualizer() 
render(vis)
visualize_elbow!(vis, nothing, x_hist, Δt=timestep)
cost_initial = acrobot_cost(x_hist, u_hist, Q, R)
