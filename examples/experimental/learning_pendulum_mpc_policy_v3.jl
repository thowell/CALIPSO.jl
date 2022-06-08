# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
using Pkg
Pkg.activate(@__DIR__) 
Pkg.instantiate()

using LinearAlgebra 
using Plots
using Random
using MatrixEquations
using Symbolics
using FiniteDiff
using Test

# pendulum
n = 2 
m = 1 
T = 11
N = 1

# goal 
xT = [π; 0.0]

# noise
noise = 1.0e-3

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
    h = 0.1 # timestep 
    x + h * pendulum(x + 0.5 * h * pendulum(x, u, w), u, w)
end

f(ones(n), ones(m), ones(0))

fx(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(z, u, w), x) 
fu(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(x, z, w), u) 
    
### MPC policy 

# ## horizon 
H = 5

# ## pendulum 
num_states = [2 for t = 1:H]
num_actions = [1 for t = 1:H-1] 
num_parameters = [2 + 1, [2 + 1 for t = 2:H-1]..., 2]

p = sum(num_parameters) - 2

# ## model
dynamics = [(y, x, u, w) -> y - f(x, u, w) for t = 1:H-1]

# ## objective 
objective = [
        (x, u, w) -> w[3]^2 * dot(u, u),
        [(x, u, w) -> transpose(x - xT) * Diagonal(w[1:2].^2) * (x - xT) + w[3].^2 * dot(u, u) for t = 2:H-1]..., 
        (x, u, w) -> transpose(x - xT) * Diagonal(w[1:2].^2) * (x - xT),
]

# ## constraints 
equality = [
            (x, u, w) -> x - w[1:2], 
            [empty_constraint for t = 2:H-1]..., 
            empty_constraint,
]

nonnegative = [
        [(x, u, w) -> [10.0 .- u; u .+ 10.0] for t = 1:H-1]..., 
        empty_constraint,
]

# ## parameters 
parameters = [
    [zeros(num_states[1]); 1.0], 
    [[1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0]
]

# ## options 
options = Options(
    verbose=false
)

# ## solver
solver = Solver(objective, dynamics, num_states, num_actions,
    parameters=parameters,
    equality=equality,
    options=options);

function ϕ_calipso(x, θ)
    # initialize
    fill!(solver.solution.variables, 0.0)
    x_guess = [x for t = 1:H]
    u_guess = [1.0e-1 * randn(num_actions[t]) for t = 1:H-1]
    initialize_states!(solver, x_guess) 
    initialize_actions!(solver, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.solution.variables[solver.problem.custom.indices.actions[1]]
end

function ϕx_calipso(x, θ)
    # initialize
    fill!(solver.solution.variables, 0.0)

    x_guess = [x for t = 1:H]
    u_guess = [1.0e-1 * randn(num_actions[t]) for t = 1:H-1]
    initialize_states!(solver, x_guess) 
    initialize_actions!(solver, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], 1:num_states[1]]
end

function ϕθ_calipso(x, θ)
    # initialize
    fill!(solver.solution.variables, 0.0)

    x_guess = [x for t = 1:H]
    u_guess = [1.0e-1 * randn(num_actions[t]) for t = 1:H-1]
    initialize_states!(solver, x_guess) 
    initialize_actions!(solver, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], num_states[1] .+ (1:(solver.dimensions.parameters-num_states[1]))]
end

x0 = noise * randn(num_states[1])
parameters_obj = [
    [1.0], 
    [[1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0]
]
p = sum([length(par) for par in parameters_obj])
θ0 = vcat(parameters_obj...)
ϕ_calipso(x0, θ0)
ϕx_calipso(x0, θ0)
ϕθ_calipso(x0, θ0)

x_sol, u_sol = CALIPSO.get_trajectory(solver)
solver.parameters
solver.problem.equality_constraint

### 

function ψt(x, u)
    J = 0.0 
    Q = 0.0 * Diagonal([1.0; 1.0e-1]) 
    R = Diagonal(1.0e-2 * ones(m))
    J += transpose(x - xT) * Q * (x - xT)
    J += transpose(u) * R * u
    return J 
end

ψtx(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(z, u), x)
ψtu(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(x, z), u)

function ψT(x)
    J = 0.0 
    Q = Diagonal([100.0; 100.0]) 
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
E = 1

# initial policy
θ = 1.0 * rand(p)
J_opt = 0.0
for i = 1:E
    x0 = noise * randn(n)
    X, U, W = simulate(x0, T, 
        # x -> ϕ(x, θ),
        x -> ϕ_calipso(x, θ),
    )
    J_opt += ψ(X, U, W)
end
@show J_opt / E

X, U, W = simulate([0.0; 0.0], T, 
        x -> ϕ_calipso(x, θ),
    )

# @test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < noise
# plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
ψθ(X, U, W, θ)
N = 1
c = [J_opt / E]

for i = 1:1000
    if i == 1 
        println("iteration: $(i)") 
        println("cost: $(c[end])") 
    end
    # i == 250 && (α *= 0.1) 
    # i == 500 && (α *= 0.1) 


    # train
    J = 0.0 
    Jθ = zeros(p)
    for j = 1:N 
        x0 = noise * randn(n)
        X, U, W = simulate(x0, T, 
            z -> ϕ_calipso(z, θ),
        ) 
        J += ψ(X, U, W)
        Jθ += ψθ(X, U, W, θ)
    end
    
    J_cand = Inf
    θ_cand = zero(θ)
    α = 1.0
    iter = 0
    while J_cand >= J 
        θ_cand = θ - α * Jθ ./ N
        J_cand = 0.0
        for j = 1:N 
            x0 = noise * randn(n)
            X, U, W = simulate(x0, T, 
                z -> ϕ_calipso(z, θ_cand),
            ) 
            J_cand += ψ(X, U, W)
            # Jθ += ψθ(X, U, W, θ)
        end
        α = 0.5 * α
        iter += 1 
        iter > 100 && error("failure") 
    end

    J = J_cand 
    θ = θ_cand

    norm(Jθ, Inf) < 1.0e-2 && break

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

# ϕ_calipso(ones(n), θ)
θ
# evaluate policy
X, U, W = simulate([0.0; 0.0], T, 
        x -> ϕ_calipso(x, θ),
    )
ψ(X, U, W)
plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
plot(hcat(U...)', xlabel="time step", ylabel="control", linetype=:steppost)

@show X[end]

