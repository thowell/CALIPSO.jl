# merit 
function merit(f, x, r, s, κ, λ, ρ, indices)
    M = 0.0
    
    # objective
    M += f

    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ[1] * dot(r, r)

    # barrier  
    M -= κ[1] * cone_barrier(s, indices.cone_nonnegative, indices.cone_second_order)

    return M
end

function merit_gradient(fx, x, r, s, κ, λ, ρ, indices)
    Mx = fx
    Mr = λ + ρ[1] * r 
    Ms = - κ[1] * cone_barrier_gradient(s, indices.cone_nonnegative, indices.cone_second_order)

    return Mx, Mr, Ms
end