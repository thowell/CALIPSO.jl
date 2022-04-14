using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 

include("cartpole.jl")
n = trajopt.dimensions.total_variables 
m = trajopt.dimensions.equality

# augmented Lagrangian 
al(x, λ, ρ) = f(x) + dot(λ, g(x)) + 0.5 * ρ * dot(g(x), g(x)) 
alx(x, λ, ρ) = fx(x) + transpose(gx(x)) * (λ + ρ * g(x))
alxx(x, λ, ρ) = fxx(x) + 0.0 * gyxx(x, λ) + ρ * transpose(gx(x)) * gx(x) 

# x = randn(n) 
y = zeros(m) 
λ = zeros(m) 
ρ = 10000.0

M = al(x, λ, ρ)
Mx = alx(x, λ, ρ) 
Mxx = alxx(x, λ, ρ) 
eigen(Mxx).values
cond(Mxx)

Lxx = fxx(x) 
Lxy = transpose(gx(x)) 
Lyx = gx(x) 
Lyy = -1.0 / ρ * I(m) 
H = [Lxx Lxy; Lyx Lyy]
eigen(H).values
cond(H)

total_iter = 0
for j = 1:10
    for i = 1:100
        println("iter: ($j, $i, $total_iter)")
        # evaluate 
        M = al(x, λ, ρ)
        Mx = alx(x, λ, ρ)
        Mxx = alxx(x, λ, ρ) 
        Mx_norm = norm(Mx) 

        Lx = fx(x) + transpose(gx(x)) * y 
        Ly = g(x) + 1.0 / ρ * (λ - y) 
        Lxx = fxx(x) 
        Lxy = transpose(gx(x)) 
        Lyx = gx(x) 
        Lyy = -1.0 / ρ * I(m) 
        H = [Lxx Lxy; Lyx Lyy] 
        h = [Lx; Ly] 
        h_norm = norm(h, Inf)

        println("res: $(h_norm)")
        println("cond: $(cond(H))")
        # println(sum(eigen(H).values .> 0.0))

        # check convergence
        if h_norm < (norm(g(x), Inf) < 1.0e-3 ? 1.0e-4 : 1.0e-3)
            break 
        end

        # search direction 
        Δ = H \ h 

        # line search
        ls_iter = 0
        
        α = 1.0  
        x̂ = x - α * Δ[1:n] 
        while al(x̂, λ, ρ) > M 
            α = 0.5 * α
            x̂ = x - α * Δ[1:n]
            ls_iter += 1 
            ls_iter > 25 && error("line search failure")
        end
        println("α = $α")

        # update
        x = x̂
        y = y - α * Δ[n .+ (1:m)]

        total_iter += 1
        println("con: $(norm(g(x), Inf))")
        println("")
    end
  
    if norm(g(x), Inf) < 1.0e-3 && norm(fx(x) + transpose(gx(x)) * y, Inf) < 1.0e-3
        println("solve success!")
        break 
    end
    # update
    λ = λ + ρ * g(x) 
    ρ = 10 * ρ 
end


using Plots
x_sol = [x[idx] for idx in trajopt.indices.states]
u_sol = [x[idx] for idx in trajopt.indices.actions]

plot(hcat(x_sol...)', xlabel="time step", ylabel="state")
plot(hcat(u_sol...)', xlabel="time step", ylabel="action")