function problem!(data::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T};
    gradient=true,
    constraint=true,
    jacobian=true,
    hessian=true) where T

    x = @views variables[idx.variables]
    y = @views variables[idx.equality]
    z = @views variables[idx.inequality]

    # TODO: remove final allocations
    gradient && methods.objective_gradient(data.objective_gradient, x)
    hessian && methods.objective_hessian(data.objective_hessian, x)

    constraint && methods.equality(data.equality, x)
    jacobian && methods.equality_jacobian(data.equality_jacobian, x)
    hessian && methods.equality_hessian(data.equality_hessian, x, y)

    constraint && methods.inequality(data.inequality, x)
    jacobian && methods.inequality_jacobian(data.inequality_jacobian, x)
    hessian && methods.inequality_hessian(data.inequality_hessian, x, z)

    return
end

function matrix!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ, ϵp, ϵd)
    # slacks 
    s = @views w[idx.slack_primal]
    t = @views w[idx.slack_dual]

    # reset
    H = s_data.matrix 
    fill!(H, 0.0)

    # Hessian of Lagrangian
    for i in idx.variables 
        for j in idx.variables 
            H[i, j]  = p_data.objective_hessian[i, j] 
            H[i, j] += p_data.equality_hessian[i, j]
            H[i, j] += p_data.inequality_hessian[i, j]
        end
    end

    # equality Jacobian 
    for (i, ii) in enumerate(idx.equality) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.equality_jacobian[i, j]
            H[jj, ii] = p_data.equality_jacobian[i, j]
        end
    end

    # augmented Lagrangian 
    for (i, ii) in enumerate(idx.equality)
        H[ii, ii] -= 1.0 / ρ[1]
    end

    # inequality Jacobian 
    for (i, ii) in enumerate(idx.inequality) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.inequality_jacobian[i, j]
            H[jj, ii] = p_data.inequality_jacobian[i, j]
        end
    end

    # identity blocks 
    for (i, ii) in enumerate(idx.inequality)
        for (j, jj) in enumerate(idx.slack_primal)
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = -1.0
            end
        end
    end

    # identity and T blocks 
    for (i, ii) in enumerate(idx.slack_primal)
        for (j, jj) in enumerate(idx.slack_dual)
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = t[i] 
            end
        end
    end

    # S block 
    for (i, ii) in enumerate(idx.slack_dual)
        H[ii, ii] = s[i] 
    end

    # regularization 
    for i in idx.variables 
        H[i, i] += ϵp 
    end

    for i in idx.slack_primal 
        H[i, i] += ϵp
    end 

    for i in idx.equality 
        H[i, i] -= ϵd
    end

    for i in idx.inequality 
        H[i, i] -= ϵd 
    end

    for i in idx.slack_dual 
        H[i, i] -= ϵd 
    end 

    return
end

function matrix_symmetric!(matrix_symmetric, matrix, idx::Indices)
    # reset
    fill!(matrix_symmetric, 0.0)

    # Hessian of Lagrangian
    for i in idx.variables 
        for j in idx.variables 
            matrix_symmetric[i, j] = matrix[i, j]
        end
    end

    # equality Jacobian 
    for (i, ii) in enumerate(idx.equality) 
        for (j, jj) in enumerate(idx.variables)
            matrix_symmetric[idx.symmetric_equality[i], jj] = matrix[ii, jj]
            matrix_symmetric[jj, idx.symmetric_equality[i]] = matrix[ii, jj]
        end
    end

    # augmented Lagrangian 
    for (i, ii) in enumerate(idx.equality)
        matrix_symmetric[idx.symmetric_equality[i], idx.symmetric_equality[i]] = matrix[ii, ii]
    end

    # inequality Jacobian 
    for (i, ii) in enumerate(idx.inequality) 
        for (j, jj) in enumerate(idx.variables)
            matrix_symmetric[idx.symmetric_inequality[i], jj] = matrix[ii, jj]
            matrix_symmetric[jj, idx.symmetric_inequality[i]] = matrix[ii, jj]
        end
    end

    # -T \ S block | -(T + S̄P) \ S̄ + D
    for (i, ii) in enumerate(idx.slack_dual)
        S̄i = matrix[ii, ii] 
        Ti = matrix[ii, idx.slack_primal[i]]
        Pi = matrix[idx.slack_primal[i], idx.slack_primal[i]] 
        Di = matrix[idx.inequality[i], idx.inequality[i]]
        matrix_symmetric[idx.symmetric_inequality[i], idx.symmetric_inequality[i]] += -1.0 * S̄i / (Ti + S̄i * Pi) + Di
    end
   
    return
end

function residual!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ)
    # duals 
    y = @views w[idx.equality]
    z = @views w[idx.inequality]
    num_equality = length(y) 
    num_inequality = length(z)

    # slacks 
    s = @views w[idx.slack_primal]
    t = @views w[idx.slack_dual]

    # reset
    r = s_data.residual 
    fill!(r, 0.0)

    # gradient of Lagrangian 
    r[idx.variables] = p_data.objective_gradient 

    for (i, ii) in enumerate(idx.variables)
        cy = 0.0
        for j = 1:num_equality 
            cy += p_data.equality_jacobian[j, i] * y[j]
        end
        r[ii] += cy 

        cz = 0.0
        for k = 1:num_inequality 
            cz += p_data.inequality_jacobian[k, i] * z[k]
        end
        r[ii] += cz
    end

    # equality 
    r[idx.equality] = p_data.equality
    for (i, ii) in enumerate(idx.equality)
        r[ii] -= 1.0 / ρ[1] * (λ[i] - y[i])
    end

    # inequality 
    r[idx.inequality] = p_data.inequality 
    for (i, ii) in enumerate(idx.inequality) 
        r[ii] -= s[i]
    end

    # # -z - t
    for (i, ii) in enumerate(idx.slack_primal)
        r[ii] = -z[i] - t[i]
    end
    
    # s ∘ t 
    for (i, ii) in enumerate(idx.slack_dual) 
        r[ii] = s[i] * t[i] - κ[1]
    end

    return 
end

function residual_symmetric!(residual_symmetric, residual, matrix, idx::Indices)
    # reset
    fill!(residual_symmetric, 0.0)

    rx = @views residual[idx.variables]
    rs = @views residual[idx.slack_primal]
    ry = @views residual[idx.equality]
    rz = @views residual[idx.inequality]
    rt = @views residual[idx.slack_dual]

    residual_symmetric[idx.variables] = rx
    residual_symmetric[idx.symmetric_equality] = ry
    residual_symmetric[idx.symmetric_inequality] = rz
 
    # inequality correction
    for (i, ii) in enumerate(idx.slack_dual) 
        S̄i = matrix[ii, ii] 
        Ti = matrix[ii, idx.slack_primal[i]]
        Pi = matrix[idx.slack_primal[i], idx.slack_primal[i]] 
        residual_symmetric[idx.symmetric_inequality[i]] += (rt[i] + S̄i * rs[i]) / (Ti + S̄i * Pi)
    end

    return 
end

function step!(step, data::SolverData)
    fill!(step, 0.0)
    step .= data.matrix \ data.residual
    return 
end

function step_symmetric!(step, residual, matrix, step_symmetric, residual_symmetric, matrix_symmetric, idx::Indices, solver::LinearSolver)
    # reset
    fill!(step, 0.0) 
    fill!(step_symmetric, 0.0)
    
    # solve symmetric system
    residual_symmetric!(residual_symmetric, residual, matrix, idx) 
    matrix_symmetric!(matrix_symmetric, matrix, idx) 
    linear_solve!(solver, step_symmetric, matrix_symmetric, residual_symmetric)
    
    # set Δx, Δy, Δz
    Δx = @views step_symmetric[idx.variables]
    Δy = @views step_symmetric[idx.symmetric_equality]
    Δz = @views step_symmetric[idx.symmetric_inequality]
    step[idx.variables] = Δx
    step[idx.equality] = Δy
    step[idx.inequality] = Δz

    # recover Δs, Δt
    Δs = @views step[idx.slack_primal]
    Δt = @views step[idx.slack_dual]
    rs = @views residual[idx.slack_primal] 
    rt = @views residual[idx.slack_dual]
    num_inequality = length(idx.inequality) 

    # Δt = z + t - Δz | -rs - Δz
    # Δs = s - κ[1] ./ t - s.* Δt ./ t | rt / t + rs * s / t + Δz  * s / t
    for i = 1:num_inequality
        S̄i = matrix[idx.slack_dual[i], idx.slack_dual[i]]
        Ti = matrix[idx.slack_dual[i], idx.slack_primal[i]]
        Pi = matrix[idx.slack_primal[i], idx.slack_primal[i]] 
        Di = matrix[idx.inequality[i], idx.inequality[i]]
        Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
        Δt[i] = Pi * Δs[i] -rs[i] - Δz[i]
    end
    
    return 
end