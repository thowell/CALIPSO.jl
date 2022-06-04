using LinearAlgebra 
using Plots
using Random
using FiniteDiff
using Test

# scalar dynamics
n = 1
m = 1 
T = 10
N = 2

# goal 
xT = [1.0]

# noise
noise = 1.0e-2

function f(x, u, w)
    2.0 * x + 1.0 * u #+ w
end

function f_noise(x, u, w)
    2.0 * x + 1.0 * u + w
end

f(ones(n), ones(m), ones(0))

fx(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(z, u, w), x) 
fu(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(x, z, w), u) 
    
### MPC policy 
# ## horizon 
H = 5#T

# ## acrobot 
num_state = 1
num_action = 1 
num_parameters = [1 + 1, [1 + 1 for t = 2:H-1]..., 1]

p = sum(num_parameters) - 1

# ## model
dyn = [
        Dynamics(
        (y, x, u, w) -> y - f(x, u, w), 
        num_state, 
        num_state, 
        num_action, 
        num_parameter=num_parameters[t]) for t = 1:H-1
    ]

# ## objective 
o1 = (x, u, w) -> w[2] * dot(u, u)
ot = (x, u, w) -> transpose(x - xT) * Diagonal(w[1:1]) * (x - xT) + w[2] * dot(u, u)
oT = (x, u, w) -> transpose(x - xT) * Diagonal(w[1:1]) * (x - xT)

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
        Constraint((x, u, w) -> x - w[1:1], num_state, num_action, 
            num_parameter=num_parameters[1]),
        [Constraint() for t = 2:H-1]..., 
        Constraint()
    ]

ineq = [Constraint() for t = 1:H]
so = [[Constraint()] for t = 1:H]

# ## problem 
parameters = [
    [zeros(num_state); 1.0], 
    [[1.0; 1.0] for t = 2:H-1]..., 
    [100.0]
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
    return solver.solution.variables[trajopt.indices.actions[1]]
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

x0 = [0.0]#rand(num_state)
parameters_obj = [
    [1.0], 
    [[1.0; 1.0] for t = 2:H-1]..., 
    [1.0]
]
θ0 = vcat(parameters_obj...)
ϕ_calipso(x0, θ0)

solver.parameters
trajopt.data.parameters
ϕx_calipso(x0, θ0)
ϕθ_calipso(x0, θ0)

x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)
solver.parameters
solver.problem.equality_constraint

### 

function ψt(x, u)
    J = 0.0 
    Q = Diagonal([0.1]) 
    R = Diagonal(0.0 * ones(m))
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
        push!(X, f_noise(X[end], U[end], W[end]))
    end 
    return X, U, W 
end

# number of evaluations 
E = 5

# initial policy
θ = 1.0 * rand(p)
J_opt = 0.0
for i = 1:E
    x0 = [0.0]#noise * randn(n)
    X, U, W = simulate(x0, T, 
        # x -> ϕ(x, θ),
        x -> ϕ_calipso(x, θ),
    )
    J_opt += ψ(X, U, W)
end
@show J_opt / E

X, U, W = simulate([0.0], T, 
        x -> ϕ_calipso(x, θ),
    )

# @test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < noise
plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
ψθ(X, U, W, θ)
N = 4
α = 0.1
c = [J_opt / E]
for i = 1:1000
    if i == 1 
        println("iteration: $(i)") 
        println("cost: $(c[end])") 
    end
    i == 500 && (α = 0.1) 

    # train
    J = 0.0 
    Jθ = zeros(p)
    for j = 1:N 
        x0 = [0.0]#noise * randn(n)
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
            x0 = [0.0]#noise * randn(n)
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
X, U, W = simulate([0.0], T, 
        x -> ϕ_calipso(x, θ),
    )
ψ(X, U, W)
plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
plot(hcat(U...)', xlabel="time step", ylabel="control", linetype=:steppost)

@show X[end]

