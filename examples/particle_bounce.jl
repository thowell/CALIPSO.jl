using FiniteDiff
using LinearAlgebra 
using Plots 

m = 1.0 
g = 10.0 
h = 1.0e-2

function dynamics(q0, q1, q2, γ, h) 
    m * (q2 - 2 * q1 + q0) / h + h * m * [0.0; g] - [0.0; 1.0] * γ 
end

function residual(z, θ, κ) 
    q2 = z[1:2] 
    s1 = z[3] 
    γ1 = z[4] 
    # q3 = z[5:6]
    # s2 = z[7] 
    # γ2 = z[8]

    q0 = θ[1:2] 
    q1 = θ[3:4] 
    # γ0 = θ[5]

    [
        dynamics(q0, q1, q2, γ1, h);
        # dynamics(q1, q2, q3, γ2, h);
        s1 - q2[2]; 
        s1 * γ1 - κ;
        # s2 - q3[2]; 
        # s2 * γ2 - κ;
        # t - q1[2]; 
        # t * ψ - κ;
    ]
end

function step(q0, q1)
    z = ones(8) 
    z[1:2] = copy(q[end]) 
    # z[5:6] = copy(q[end])
    θ = [copy(q[end-1]); copy(q[end])]
    κ = 0.1

    for j = 1:100
        for i = 1:100
            r = residual(z, θ, κ)
            norm(r, 1) < 1.0e-12 && break 
            R = FiniteDiff.finite_difference_jacobian(w -> residual(w, θ, κ), z)

            Δ = R \ r 

            α = 1.0 

            iter = 0
            while any((z - α * Δ)[3:4] .<= 0.0)
                α = 0.5 * α 
                iter += 1 
                iter > 100 && error("line search failure")
            end

            while norm(residual(z - α * Δ, θ, κ), 1) > norm(r, 1)  
                α = 0.5 * α 
                iter += 1
                iter > 100 && error("line search failure")
            end

            z = z - α * Δ
        end
        if κ < 1.0e-10
            break 
        else
            κ = 0.1 * κ 
        end
    end

    return z[1:2]
end

q0 = [0.0; 1.0] 
q1 = [0.0; 1.0]
q = [q0, q1] 
for t = 1:100
    push!(q, step(q[end-1], q[end]))
    # push!(q, q2) 
    # push!(γ, γ1)
end

plot(hcat(q...)[1:2, :]', xlabel="time", ylabel="q")