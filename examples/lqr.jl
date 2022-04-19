# Learning Convex Optimization Control Policies
# http://proceedings.mlr.press/v120/agrawal20a/agrawal20a.pdf

using LinearAlgebra 
using Plots
using Random, Distributions
using ControlSystems
using Symbolics
using FiniteDiff

n = 4 
m = 2 
T = 100 
noise = sqrt(0.25)
A = randn(n, n)
A ./= maximum(abs.(eigen(A).values))
B = randn(n, m)
Q = 1.0 * Array(I(n)) 
R = 1.0 * Array(I(m))
K = ControlSystems.lqr(ControlSystems.Discrete, A, B, Q, R)
N = 6 

f(x, u, w) = A * x + B * u + w
    
function lqr(x) 
    -K * x
end

function ϕ(x, θ)
    P = reshape(θ, n, n)' * reshape(θ, n, n) 
    K = (R + B' * P * B) \ (B' * P * A) 
    return -K * x
end

function ϕθ(x, θ) 
    FiniteDiff.finite_difference_jacobian(p -> ϕ(x, p), θ) 
end

function ψ(X, U, W)
    J = 0.0 
    for t = 1:T 
        J += transpose(X[t+1]) * Q * X[t+1]
        J += transpose(U[t]) * R * U[t] 
    end
    return J / T
end

function ψθ_fd(X, U, W, θ) 
    function eval_ψ(x0, θ) 
        u = [] 
        x = [x0] 
        for t = 1:T 
            push!(u, ϕ(x[end], θ))
            push!(x, f(x[end], u[end], W[t])) 
        end
        return ψ(x, u, W) 
    end

    FiniteDiff.finite_difference_gradient(p -> eval_ψ(X[1], p), θ) 
end

function ψθ(X, U, W, θ) 
    Jθ = zeros(length(θ)) 
    for t = 1:T 
        Jθ += 2.0 * ϕθ(X[t], θ)' * B' * Q * A * X[t] + 2.0 * ϕθ(X[t], θ)' * (B' * Q * B + R) * ϕ(X[t], θ) + 2.0 * ϕθ(X[t], θ)' * B' * Q * W[t]
    end 
    return Jθ ./ T 
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

# random initial state
x0 = randn(n)

# LQR baseline
X, U, W = simulate(x0, T, 
    lqr,
)
J = ψ(X, U, W)

θ = vec(1.0 * Diagonal(ones(n)))
X, U, W = simulate(x0, T, 
    x -> ϕ(x, θ),
    # lqr,
)
J = ψ(X, U, W)
Jθ = ψθ(X, U, W, θ)
Jθ = ψθ_fd(X, U, W, θ)

# plot(hcat(X...)', xlabel="time step", ylabel="states")

α = 0.5
cost = []
for i = 1:50
    i == 25 && (α = 0.1) 

    J = 0.0 
    Jθ = zeros(n^2)
    for j = 1:N 
        x0 = noise * randn(n)
        X, U, W = simulate(x0, T, 
            z -> ϕ(z, θ),
        ) 
        J += ψ(X, U, W)
        Jθ += ψθ(X, U, W, θ)
    end
    push!(cost, J / N) 
    θ = θ - α * Jθ ./ N 

    println("iteration: $i") 
    println("cost: $(cost[end])")
end

plot(cost)




