# merit 
function merit(f, r, Φ, κ, λ, ρ, indices)
    M = 0.0
    
    # objective
    M += f

    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ[1] * dot(r, r)

    # barrier  
    M -= κ[1] * Φ

    return M
end

function merit_gradient!(grad, fx, r, Φs, κ, λ, ρ, indices)
    # Mx = fx
    # Mr = λ + ρ[1] * r 
    # Ms = - κ[1] * Φs
    for i in indices.variables 
        grad[i] = fx[i] 
    end 

    for (i, ii) in enumerate(indices.symmetric_equality) 
        grad[ii] = λ[i] + ρ[1] * r[i] 
    end

    for (i, ii) in enumerate(indices.symmetric_cone)
        grad[ii] = -1.0 * κ[1] * Φs[i] 
    end

    return 
end