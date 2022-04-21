# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()

using LinearAlgebra 
using Plots
using Random
using Symbolics
using FiniteDiff
using Test

# pendulum
n = 2 
m = 1 
T = 100 
N = 10

# goal 
xT = [π; 0.0]

# noise
noise = 0.0e-2

function pendulum(x, u, w)
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

# function midpoint_implicit(y, x, u, w)
#     h = 0.05 # timestep 
#     y - (x + h * pendulum(0.5 * (x + y), u, w))
# end

function f(x, u, w)
    h = 0.05 # timestep 
    x + h * pendulum(x + 0.5 * h * pendulum(x, u, w), u, w)
end

fx(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(z, u, w), x) 
fu(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(x, z, w), u) 
    
# function ϕ(x, θ)
#     K = reshape(θ, m, n)
#     return -K * (x - xT)
# end

# function ϕx(x, θ) 
#     FiniteDiff.finite_difference_jacobian(z -> ϕ(z, θ), x) 
# end

# function ϕθ(x, θ) 
#     FiniteDiff.finite_difference_jacobian(p -> ϕ(x, p), θ) 
# end

### MPC policy 
# ## horizon 
H = 3

# ## acrobot 
num_state = 2
num_action = 1 
num_parameters = [2 + 1, [2 + 1 for t = 2:H-1]..., 2]

p = sum(num_parameters) - 2


function midpoint_implicit(y, x, u, w)
    h = 0.05 # timestep 
    y - (x + h * pendulum(0.5 * (x + y), u, w))
end

# ## model
dyn = [
        Dynamics(
        midpoint_implicit, 
        num_state, 
        num_state, 
        num_action, 
        num_parameter=num_parameters[t]) for t = 1:H-1
    ]

# ## initialization
# ## objective 
o1 = (x, u, w) -> w[3] * dot(u, u)
ot = (x, u, w) -> transpose(x[1:2] - xT) * Diagonal(w[1:2]) * (x[1:2] - xT) + w[3] * dot(u, u)
oT = (x, u, w) -> transpose(x[1:2] - xT) * Diagonal(w[1:2]) * (x[1:2] - xT)

obj = [
        Cost(o1, num_state, num_action, 
            num_parameter=num_parameters[1]),
        [Cost(ot, num_state, num_action,
            num_parameter=num_parameters[t]) for t = 2:H-1]..., 
        Cost(oT, num_state, 0,
            num_parameter=num_parameters[H])
      ]

# ## constraints 
eq = [
        Constraint((x, u, w) -> x - w[1:2], num_state, num_action, 
            num_parameter=num_parameters[1]),
        [Constraint() for t = 2:H-1]..., 
        Constraint()
    ]

ineq = [Constraint() for t = 1:H]
so = [[Constraint()] for t = 1:H]

# ## problem 
parameters = [
    [zeros(num_state); 1.0], 
    [[1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0]
]
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so;
    parameters=parameters,
)

# ## solver
methods = ProblemMethods(trajopt)
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    parameters=vcat(parameters...),   
    options=Options(verbose=false, penalty_initial=1.0e4))

function ϕ_calipso(x, θ)
    # initialize
    x_guess = [x for t = 1:H]
    u_guess = [randn(num_action) for t = 1:H-1]
    initialize_states!(solver, trajopt, x_guess) 
    initialize_controls!(solver, trajopt, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    # control
    return solver.variables[trajopt.indices.actions[1]]
end

function ϕx_calipso(x, θ)
    # initialize
    x_guess = [x for t = 1:H]
    u_guess = [1.0 * randn(num_action) for t = 1:H-1]
    initialize_states!(solver, trajopt, x_guess) 
    initialize_controls!(solver, trajopt, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[trajopt.indices.actions[1], 1:num_state]
end

function ϕθ_calipso(x, θ)
    # initialize
    x_guess = [x for t = 1:H]
    u_guess = [1.0 * randn(num_action) for t = 1:H-1]
    initialize_states!(solver, trajopt, x_guess) 
    initialize_controls!(solver, trajopt, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[trajopt.indices.actions[1], num_state .+ (1:(solver.dimensions.parameters-num_state))]
end

x0 = zeros(num_state)
parameters_obj = [
    [1.0], 
    [[1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0]
]
θ0 = vcat(parameters_obj...)
ϕ_calipso(x0, θ0)
ϕx_calipso(x0, θ0)
ϕθ_calipso(x0, θ0)

### 

function ψt(x, u)
    J = 0.0 
    Q = Diagonal(ones(n)) 
    R = Diagonal(0.1 * ones(m))
    J += transpose(x - xT) * Q * (x - xT)
    J += transpose(u) * R * u
    return J 
end

ψtx(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(z, u), x)
ψtu(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(x, z), u)

function ψT(x)
    J = 0.0 
    Q = 10.0 * Diagonal(ones(n)) 
    J += transpose(x - xT) * Q * (x - xT)
    return J 
end

ψTx(x) = FiniteDiff.finite_difference_gradient(ψT, x)

function ψ(X, U, W)
    J = 0.0 
    for t = 1:T 
        J += ψt(X[t], U[t])
    end
    J += ψT(X[T+1])
    return J / (T + 1) 
end

function ψθ(X, U, W, θ) 
    Jθ = zeros(length(θ)) 

    ∂x∂θ = [zeros(n, p)]
    ∂u∂θ = []
    ∂ϕ∂x = [] 
    ∂ϕ∂θ = []
    for t = 1:T
        # ∂u∂θ
        push!(∂ϕ∂x, ϕx_calipso(X[t], θ))
        push!(∂ϕ∂θ, ϕθ_calipso(X[t], θ)) 
        # ∂f∂x, ∂f∂u
        ∂f∂x = fx(X[t], U[t], W[t])
        ∂f∂u = fu(X[t], U[t], W[t])
        push!(∂u∂θ, ∂ϕ∂x[end] * ∂x∂θ[end] + ∂ϕ∂θ[end])
        push!(∂x∂θ, ∂f∂x * ∂x∂θ[end] + ∂f∂u * ∂u∂θ[end])
    end

    for t = 1:T
        Jθ += ∂x∂θ[t]' * ψtx(X[t], U[t]) 
        Jθ += ∂u∂θ[t]' * ψtu(X[t], U[t])
    end 
    Jθ += ∂x∂θ[T+1]' * ψTx(X[T+1])
    return Jθ ./ (T + 1) 
end

# function ψθ_fd(X, U, W, θ) 
#     function eval_ψ(x0, θ) 
#         u = [] 
#         x = [x0] 
#         for t = 1:T 
#             push!(u, ϕ(x[end], θ))
#             push!(x, f(x[end], u[end], W[t])) 
#         end
#         return ψ(x, u, W) 
#     end

#     FiniteDiff.finite_difference_gradient(p -> eval_ψ(X[1], p), θ) 
# end

function simulate(x0, T, policy) 
    X = [x0] 
    U = [] 
    W = [] 
    for t = 1:T 
        push!(U, policy(X[end]))
        push!(W, noise * randn(n)) 
        push!(X, f(X[end], U[end], W[end]))
    end 
    return X, U, W 
end

# number of evaluations 
E = 5

# initial policy
θ = rand(p)
J_opt = 0.0
for i = 1:E
    x0 = [1.0]
    X, U, W = simulate(x0, T, 
        # x -> ϕ(x, θ),
        x -> ϕ_calipso(x, θ),
    )
    J_opt += ψ(X, U, W)
end
@show J_opt / E

X, U, W = simulate([1.0], T, 
        x -> ϕ_calipso(x, θ),
    )

# @test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < noise
# plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
ψθ(X, U, W, θ)
N = 4
α = 0.5
c = [J_opt / E]
for i = 1:100
    if i == 1 
        println("iteration: $(i)") 
        println("cost: $(c[end])") 
    end
    i == 25 && (α = 0.1) 

    # train
    J = 0.0 
    Jθ = zeros(p)
    for j = 1:N 
        x0 = [1.0]
        X, U, W = simulate(x0, T, 
            # z -> ϕ(z, θ),
            z -> ϕ_calipso(z, θ),
        ) 
        J += ψ(X, U, W)
        Jθ += ψθ(X, U, W, θ)
    end
    θ = θ - α * Jθ ./ N 


    # evaluate
    if i % 10 == 0
        J_eval = 0.0 
        for k = 1:E 
            x0 = noise * randn(n)
            X, U, W = simulate(x0, T, 
                # z -> ϕ(z, θ),
                z -> ϕ_calipso(z, θ),
            ) 
            J_eval += ψ(X, U, W)
        end
        push!(c, J_eval / E)
        println("iteration: $(i)") 
        println("cost: $(c[end])")
    end
end

plot(c, xlabel="iteration", ylabel="cost")
# plot!(J_lqr / E * ones(length(cost)), color=:black)

# evaluate policy
X, U, W = simulate(noise * randn(n), T, 
        x -> ϕ_calipso(x, θ),
    )

plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
plot(hcat(U...)', xlabel="time step", ylabel="control")

@show X[end]

