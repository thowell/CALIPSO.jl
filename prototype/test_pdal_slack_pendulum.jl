using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 

include("pendulum.jl")
n = trajopt.num_variables 
m = trajopt.num_equality

# augmented Lagrangian 
al(x, r, λ, ρ) = f(x) + dot(λ, r) + 0.5 * ρ * dot(r, r) 

r = g(x)
y = zeros(m) 
λ = zeros(m) 
ρ = 1.0 

total_iter = 0
for j = 1:10
    for i = 1:100
        println("iter: ($j, $i, $total_iter)")

        M = al(x, r, λ, ρ)

        Lx = fx(x) + transpose(gx(x)) * y 
        Lr = λ + ρ * r - y
        Ly = g(x) - r 
        Lxx = fxx(x) + gyxx(x, y)
        Lxr = zeros(n, m) 
        Lxy = transpose(gx(x)) 
        Lrx = zeros(m, n) 
        Lrr = ρ * I(m) 
        Lry = -1.0 * I(m) 
        Lyx = gx(x) 
        Lyr = -1.0 * I(m) 
        Lyy = zeros(m, m)
        H = [Lxx Lxr Lxy; Lrx Lrr Lry; Lyx Lyr Lyy] 
        h = [Lx; Lr; Ly] 
        h_norm = norm(h, Inf)

        println("res: $(h_norm)")
        println("cond: $(cond(H))")

        # check convergence
        if h_norm < (norm(g(x), Inf) < 1.0e-3 ? 1.0e-4 : 1.0e-3)
            break 
        end

        # search direction 
        Δ = H \ h 

        # line search
        ls_iter = 0
        c1 = 1.0e-4 
        c2 = 0.9
        α = 1.0  
        x̂ = x - α * Δ[1:n] 
        r̂ = r - α * Δ[n .+ (1:m)]
        ŷ = y - α * Δ[n + m .+ (1:m)]

        while al(x̂, r̂, λ, ρ) + dot(ŷ, g(x̂) - r̂) > M + dot(y, g(x) - r) + c1 * α * dot(Δ, [Lx; Lr; Ly]) || -dot(Δ, [fx(x̂) + transpose(gx(x̂)) * ŷ; λ + ρ * r̂ - ŷ; g(x̂) - r̂]) > -c2 * dot(Δ, [Lx; Lr; Ly])
        # while al(x̂, r̂, λ, ρ) > M 
            α = 0.5 * α
            x̂ = x - α * Δ[1:n]
            r̂ = r - α * Δ[n .+ (1:m)]
            ŷ = y - α * Δ[n + m .+ (1:m)]
            ls_iter += 1 
            ls_iter > 25 && error("line search failure")
        end
        println("α = $α")

        # update
        x = x̂
        r = r̂
        y = ŷ

        total_iter += 1
        println("con: $(norm(g(x), Inf))")
        println("")
    end
  
    if norm(g(x), Inf) < 1.0e-3 #&& norm(fx(x) + transpose(gx(x)) * y, Inf) < 1.0e-3
        println("solve success!")
        break 
    end
    # update
    λ = λ + ρ * r
    ρ = 10 * ρ 
end

using Plots
x_sol = [x[idx] for idx in trajopt.indices.states]
u_sol = [x[idx] for idx in trajopt.indices.actions]

plot(hcat(x_sol...)', xlabel="time step", ylabel="state")
plot(hcat(u_sol...)', xlabel="time step", ylabel="action")