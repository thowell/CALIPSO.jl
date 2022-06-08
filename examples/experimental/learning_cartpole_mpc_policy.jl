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

# cartpole 
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

function cartpole_discrete(x, u, w)
    h = 0.05 # timestep 
    x + h * cartpole_continuous(x + 0.5 * h * cartpole_continuous(x, u), u)
end

function cartpole_discrete(y, x, u, w)
    y - cartpole_discrete(x, u, w)
end

n = 4 
m = 1 
T = 26
N = 1

# goal 
xT = [0.0; π; 0.0; 0.0]

# noise
noise = 1.0e-3

f = cartpole_discrete

f(ones(n), ones(m), zeros(0))

fx(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(z, u, w), x) 
fu(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(x, z, w), u) 
    
### MPC policy 

# ## horizon 
H = 5

# ## pendulum 
num_states = [4 for t = 1:H]
num_actions = [1 for t = 1:H-1] 
num_parameters = [4 + 1, [4 + 1 for t = 2:H-1]..., 4]

p = sum(num_parameters) - 4

# ## model
dynamics = [(y, x, u, w) -> y - f(x, u, w) for t = 1:H-1]

# ## objective 
objective = [
        (x, u, w) -> w[3]^2 * dot(u, u),
        [(x, u, w) -> transpose(x - xT) * Diagonal(w[1:4].^2) * (x - xT) + w[5].^2 * dot(u, u) for t = 2:H-1]..., 
        (x, u, w) -> transpose(x - xT) * Diagonal(w[1:4].^2) * (x - xT),
]

# ## constraints 
equality = [
            (x, u, w) -> x - w[1:4], 
            [empty_constraint for t = 2:H-1]..., 
            empty_constraint,
]

nonnegative = [
        [(x, u, w) -> [10.0 .- u; u .+ 10.0] for t = 1:H-1]..., 
        empty_constraint,
]

# ## parameters 
parameters = [
    [zeros(4); 1.0], 
    [[1.0; 1.0; 1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0; 1.0; 1.0]
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
    u_guess = [1.0e-1 * randn(1) for t = 1:H-1]
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
    u_guess = [1.0e-1 * randn(1) for t = 1:H-1]
    initialize_states!(solver, x_guess) 
    initialize_actions!(solver, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], 1:4]
end

function ϕθ_calipso(x, θ)
    # initialize
    fill!(solver.solution.variables, 0.0)

    x_guess = [x for t = 1:H]
    u_guess = [1.0e-1 * randn(1) for t = 1:H-1]
    initialize_states!(solver, x_guess) 
    initialize_actions!(solver, u_guess)
    solver.parameters .= [x; θ]
    
    # solve
    solve!(solver)

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], 4 .+ (1:(solver.dimensions.parameters-4))]
end

x0 = noise * randn(4)
parameters_obj = [
    [1.0], 
    [[1.0; 1.0; 1.0; 1.0; 1.0] for t = 2:H-1]..., 
    [1.0; 1.0; 1.0; 1.0;]
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
    Q = 0.0 * Diagonal([1.0; 1.0; 1.0e-1; 1.0e-1]) 
    R = Diagonal(1.0e-2 * ones(m))
    J += transpose(x - xT) * Q * (x - xT)
    J += transpose(u) * R * u
    return J 
end

ψtx(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(z, u), x)
ψtu(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(x, z), u)

function ψT(x)
    J = 0.0 
    Q = Diagonal([100.0; 100.0; 100.0; 100.0]) 
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

X, U, W = simulate([0.0; 0.0; 0.0; 0.0], T, 
        x -> ϕ_calipso(x, θ),
    )

# @test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < noise
plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "pos." "vel." "vel."])
ψθ(X, U, W, θ)
N = 1
α = 0.1
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
            # z -> ϕ(z, θ),
            z -> ϕ_calipso(z, θ),
        ) 
        J += ψ(X, U, W)
        Jθ += ψθ(X, U, W, θ)
    end
    θ = θ - α * Jθ ./ N 
    θ = max.(0.0, θ)

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
X, U, W = simulate([0.0; 0.0; 0.0; 0.0], T, 
        x -> ϕ_calipso(x, θ),
    )
ψ(X, U, W)
plot(hcat(X...)', xlabel="time step", ylabel="states", labels=["pos." "vel."])
plot(hcat(U...)', xlabel="time step", ylabel="control", linetype=:steppost)

@show X[end]

