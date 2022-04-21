# Learning Convex Optimization Control Policies
# http://proceedings.mlr.press/v120/agrawal20a/agrawal20a.pdf

using LinearAlgebra 
using Plots
using Random
using Symbolics
using FiniteDiff
using Test
using MatrixEquations
# Riccati solution
P, _, K = MatrixEquations.ared(A, B, R, Q)

n = 4 
m = 2 
T = 100 
noise = 0.25
A = randn(n, n)
A ./= maximum(abs.(eigen(A).values))
B = randn(n, m)
Q = 1.0 * Array(I(n)) 
R = 1.0 * Array(I(m))
K = ControlSystems.lqr(ControlSystems.Discrete, A, B, Q, R)
N = 10

f(x, u, w) = A * x + B * u + w
fx(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(z, u, w), x) 
fu(x, u, w) = FiniteDiff.finite_difference_jacobian(z -> f(x, z, w), u) 
    
function lqr(x) 
    -K * x
end

function ϕ(x, θ)
    P = reshape(θ, n, n)' * reshape(θ, n, n) 
    K = (R + B' * P * B) \ (B' * P * A) 
    return -K * x
end

function ϕx(x, θ) 
    FiniteDiff.finite_difference_jacobian(z -> ϕ(z, θ), x) 
end

function ϕθ(x, θ) 
    FiniteDiff.finite_difference_jacobian(p -> ϕ(x, p), θ) 
end

function ψt(x, u)
    J = 0.0 
    J += transpose(x) * Q * x
    J += transpose(u) * R * u
    return J 
end

ψtx(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(z, u), x)
ψtu(x, u) = FiniteDiff.finite_difference_gradient(z -> ψt(x, z), u)

function ψ(X, U, W)
    J = 0.0 
    for t = 1:T 
        J += ψt(X[t], U[t])
    end
    J += ψt(X[T+1], zeros(m))
    return J / (T + 1) 
end

function ψθ(X, U, W, θ) 
    Jθ = zeros(length(θ)) 

    ∂x∂θ = [zeros(n, n^2)]
    ∂u∂θ = []
    ∂ϕ∂x = [] 
    ∂ϕ∂θ = []
    for t = 1:T
        # ∂u∂θ
        push!(∂ϕ∂x, ϕx(X[t], θ))
        push!(∂ϕ∂θ, ϕθ(X[t], θ)) 
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
    Jθ += ∂x∂θ[T+1]' * ψtx(X[T+1], zeros(m))
    return Jθ ./ (T + 1) 
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
E = 100 

# LQR baseline
J_lqr = 0.0
for i = 1:E
    x0 = noise * randn(n)
    X, U, W = simulate(x0, T, 
        lqr,
    )
    J_lqr += ψ(X, U, W)
end
@show J_lqr / E

# initial policy
θ = vec(1.0 * Diagonal(ones(n)))
J_opt = 0.0
for i = 1:E
    x0 = noise * randn(n)
    X, U, W = simulate(x0, T, 
        x -> ϕ(x, θ),
    )
    J_opt += ψ(X, U, W)
end
@show J_opt / E

# test gradients
X, U, W = simulate(x0, T, 
        x -> ϕ(x, θ),
    )
@test norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf) < 1.0e-5

# plot(hcat(X...)', xlabel="time step", ylabel="states")

α = 0.5
cost = [J_opt / E]
for i = 1:50
    if i == 1 
        println("iteration: $(i)") 
        println("cost: $(cost[end])") 
    end
    i == 25 && (α = 0.1) 

    # train
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
    θ = θ - α * Jθ ./ N 

    # evaluate
    if i % 1 == 0
        J_eval = 0.0 
        for k = 1:E 
            x0 = noise * randn(n)
            X, U, W = simulate(x0, T, 
                z -> ϕ(z, θ),
            ) 
            J_eval += ψ(X, U, W)
        end
        push!(cost, J_eval / E)
        println("iteration: $(i)") 
        println("cost: $(cost[end])")
    end
end

plot(cost);
plot!(J_lqr / E * ones(length(cost)), color=:black)

function simulate_no_noise(x0, T, policy) 
    X = [x0] 
    U = [] 
    W = [] 
    for t = 1:T 
        push!(U, policy(X[end]))
        push!(W, 1.0 * randn(n)) 
        push!(X, f(X[end], U[end], W[end]))
    end 
    return X, U, W 
end

θ = ones(n^2)
X, U, W = simulate_no_noise(randn(n), T, z -> ϕ(z, θ))

norm(ψθ(X, U, W, θ) - ψθ_fd(X, U, W, θ), Inf)
