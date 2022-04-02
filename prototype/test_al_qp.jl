using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 

include("qp.jl")

# augmented Lagrangian 
al(x, λ, ρ) = f(x) + dot(λ, g(x)) + 0.5 * ρ * dot(g(x), g(x)) 
alx(x, λ, ρ) = fx(x) + transpose(gx(x)) * (λ + ρ * g(x))
alxx(x, λ, ρ) = fxx(x) + gyxx(x, λ) + ρ * transpose(gx(x)) * gx(x) 

x = randn(n) 
λ = zeros(m) 
ρ = 100.0 

M = al(x, λ, ρ)
Mx = alx(x, λ, ρ) 
Mxx = alxx(x, λ, ρ) 
eigen(Mxx).values
cond(Mxx)

total_iter = 0
for j = 1:10
    for i = 1:100
        println("iter: ($j, $i, $total_iter)")
        # evaluate 
        M = al(x, λ, ρ)
        Mx = alx(x, λ, ρ)
        Mxx = alxx(x, λ, ρ) 
        Mx_norm = norm(Mx) 
        println("res: $(Mx_norm)")
        println("cond: $(cond(Mxx))")

        # check convergence
        if Mx_norm < (norm(g(x), Inf) < 1.0e-3 ? 1.0e-4 : 1.0e-3)
            break 
        end

        # search direction 
        Δ = Mxx \ Mx 

        # line search
        ls_iter = 0
        c1 = 1.0e-4 
        c2 = 0.1
        α = 1.0  
        x̂ = x - α * Δ
            # wolfe conditions
        while al(x̂, λ, ρ) > M + c1 * α * dot(Δ, Mx) || -dot(Δ, alx(x̂, λ, ρ)) > -c2 * dot(Δ, Mx)
            α = 0.5 * α
            x̂ = x - α * Δ
            ls_iter += 1 
            ls_iter > 25 && error("line search failure")
        end
        println("α = $α")

        # update
        x = x̂
        total_iter += 1
        println("con: $(norm(g(x), Inf))")
        println("")
    end
    
    if norm(g(x), Inf) < 1.0e-3 && norm(alx(x, λ, ρ), Inf) < 1.0e-3
        break 
    end
    # update
    λ = λ + ρ * g(x) 
    ρ = 10 * ρ 
end


