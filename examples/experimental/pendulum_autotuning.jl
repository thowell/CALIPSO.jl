# @testset "Examples: Pendulum" begin 
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
# end

# ## policy 
# ## (1-layer) multi-layer perceptron policy
l_input = num_states[1]
l1 = 20
l2 = num_actions[1]
nθ = l1 * l_input + l1 + l2 * l1 + l2

function policy(θ, x)
    shift = 0
    # input
    input = x

    # layer 1
    W1 = reshape(θ[shift .+ (1:(l1 * l_input))], l1, l_input)
    shift += l1 * l_input
    b1 = θ[shift .+ (1:l1)]
    shift += l1
    z1 = W1 * input + b1

    # output
    o1 = tanh.(z1)

    # layer 2
    W2 = reshape(θ[shift .+ (1:(l2 * l1))], l2, l1)
    shift += l2 * l1 
    b2 = θ[shift .+ (1:l2)] 
    z2 = W2 * o1 + b2
    shift += l2 

    o2 = z2
    return o2
end

function policy_jacobian_state(θ, x) 
    FiniteDiff.finite_difference_jacobian(a -> policy(θ, a), x)
end

function policy_jacobian_parameters(θ, x) 
    FiniteDiff.finite_difference_jacobian(a -> policy(a, x), θ)
end

function cost(θ, x, u) 
    u_policy = policy(θ, x)

    Δ = u_policy - u
    
    return dot(Δ, Δ) 
end

function cost_gradient(θ, x, u) 
    u_policy = policy(θ, x)
    Δ = u_policy - u
    2.0 * policy_jacobian_parameters(θ, x)' * Δ
end

norm(cost_gradient_fd(θ, state_solution[1], action_solution[1]) - cost_gradient(θ, state_solution[1], action_solution[1]), Inf)

function loss(θ, X, U) 
    T = length(U) 
     
    J = 0.0 

    for t = 1:T 
        J += cost(θ, X[t], U[t]) 
    end

    return J ./ T
end

function loss_gradient(θ, X, U) 
    T = length(U) 
     
    Jθ = zeros(nθ) 

    for t = 1:T 
        Jθ .+= cost_gradient(θ, X[t], U[t]) 
    end

    return Jθ ./ T
end

θ = randn(nθ)
J_prev = loss(θ, state_solution, action_solution)

T_learn = 5
for i = 1:10000
    α = 1.0 
    Δ = loss_gradient(θ, state_solution[1:T_learn], action_solution[1:T_learn]) 
    (norm(Δ) < 1.0e-3 || abs(J_prev) < 1.0e-3) && (println("* success *"); break)

    θ_candidate = θ - α * Δ
    J_cand = loss(θ_candidate, state_solution[1:T_learn], action_solution[1:T_learn])
    ls_iter = 0 
    while J_cand >= J_prev
        α *= 0.5 
        θ_candidate = θ - α * Δ
        J_cand = loss(θ_candidate, state_solution[1:T_learn], action_solution[1:T_learn])
        ls_iter += 1
        ls_iter > 100 && (println("line search failure"); break)
    end

    θ = θ_candidate
    J_prev = J_cand

    i % 100 == 0 && println("iter: $i \ncost: $J_prev\n α = $α \n")
end
