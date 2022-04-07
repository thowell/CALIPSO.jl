# merit 
function merit(f, x, r, s, κ, λ, ρ)
    M = 0.0
    
    # objective
    M += f

    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ[1] * dot(r, r)

    # barrier  
    M -= κ[1] * sum(log.(s))

    return M
end

function merit_gradient(fx, x, r, s, κ, λ, ρ)
    Mx = fx
    Mr = λ + ρ[1] * r 
    Ms = - κ[1] ./ s

    return Mx, Mr, Ms
end