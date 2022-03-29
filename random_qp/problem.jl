function problem!(data::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T};
    gradient=true,
    constraint=true,
    jacobian=true,
    hessian=true) where T

    x = @views variables[idx.variables]
    y = @views variables[idx.equality_dual]
    z = @views variables[idx.inequality_dual]

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
    r = @views w[idx.equality_slack]
    s = @views w[idx.inequality_slack]
    t = @views w[idx.inequality_slack_dual]

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

    for (i, ii) in enumerate(idx.equality_slack) 
        for (j, jj) in enumerate(idx.equality_dual) 
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = -1.0
            end 
        end
    end

    for (i, ii) in enumerate(idx.inequality_slack) 
        for (j, jj) in enumerate(idx.inequality_dual) 
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = -1.0 
            end
        end
    end

    for (i, ii) in enumerate(idx.inequality_slack) 
        for (j, jj) in enumerate(idx.inequality_slack_dual) 
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = t[i]
            end
        end
    end

    # equality Jacobian 
    for (i, ii) in enumerate(idx.equality_dual) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.equality_jacobian[i, j]
            H[jj, ii] = p_data.equality_jacobian[i, j]
        end
    end

    # inequality Jacobian 
    for (i, ii) in enumerate(idx.inequality_dual) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.inequality_jacobian[i, j]
            H[jj, ii] = p_data.inequality_jacobian[i, j]
        end
    end

    # augmented Lagrangian block 
    for (i, ii) in enumerate(idx.equality_slack) 
        H[ii, ii] = ρ[1]
    end

    # S block 
    for (i, ii) in enumerate(idx.inequality_slack_dual)
        H[ii, ii] = s[i] 
    end

    # regularization 
    for i in idx.variables 
        H[i, i] += ϵp 
    end

    for i in idx.equality_slack
        H[i, i] += ϵp
    end 

    for i in idx.inequality_slack
        H[i, i] += ϵp
    end 

    for i in idx.equality_dual
        H[i, i] -= ϵd
    end

    for i in idx.inequality_dual
        H[i, i] -= ϵd 
    end

    for i in idx.inequality_slack_dual 
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
    for (i, ii) in enumerate(idx.equality_dual) 
        for (j, jj) in enumerate(idx.variables)
            matrix_symmetric[idx.symmetric_equality[i], jj] = matrix[ii, jj]
            matrix_symmetric[jj, idx.symmetric_equality[i]] = matrix[ii, jj]
        end
    end

    # augmented Lagrangian 
    for (i, ii) in enumerate(idx.equality_slack)
        matrix_symmetric[idx.symmetric_equality[i], idx.symmetric_equality[i]] = -1.0 / matrix[ii, ii] + matrix[idx.equality_dual[i], idx.equality_dual[i]]
    end

    # inequality Jacobian 
    for (i, ii) in enumerate(idx.inequality_dual) 
        for (j, jj) in enumerate(idx.variables)
            matrix_symmetric[idx.symmetric_inequality[i], jj] = matrix[ii, jj]
            matrix_symmetric[jj, idx.symmetric_inequality[i]] = matrix[ii, jj]
        end
    end

    # -T \ S block | -(T + S̄P) \ S̄ + D
    for (i, ii) in enumerate(idx.inequality_slack_dual)
        S̄i = matrix[ii, ii] 
        Ti = matrix[ii, idx.inequality_slack[i]]
        Pi = matrix[idx.inequality_slack[i], idx.inequality_slack[i]] 
        Di = matrix[idx.inequality_dual[i], idx.inequality_dual[i]]
        matrix_symmetric[idx.symmetric_inequality[i], idx.symmetric_inequality[i]] += -1.0 * S̄i / (Ti + S̄i * Pi) + Di
    end
   
    return
end

function residual!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ)
    # duals 
    y = @views w[idx.equality_dual]
    z = @views w[idx.inequality_dual]
    num_equality = length(y) 
    num_inequality = length(z)

    # slacks 
    r = @views w[idx.equality_slack]
    s = @views w[idx.inequality_slack]
    t = @views w[idx.inequality_slack_dual]

    # reset
    res = s_data.residual 
    fill!(res, 0.0)

    # gradient of Lagrangian 
    res[idx.variables] = p_data.objective_gradient 

    for (i, ii) in enumerate(idx.variables)
        cy = 0.0
        for j = 1:num_equality 
            cy += p_data.equality_jacobian[j, i] * y[j]
        end
        res[ii] += cy 

        cz = 0.0
        for k = 1:num_inequality 
            cz += p_data.inequality_jacobian[k, i] * z[k]
        end
        res[ii] += cz
    end

    # λ + ρr - y 
    for (i, ii) in enumerate(idx.equality_slack) 
        res[ii] = λ[i] + ρ[1] * r[i] - y[i]
    end

    # -z - t
    for (i, ii) in enumerate(idx.inequality_slack)
        res[ii] = -z[i] - t[i]
    end

    # equality 
    res[idx.equality_dual] = p_data.equality
    for (i, ii) in enumerate(idx.equality_dual)
        res[ii] -= r[i]
    end

    # inequality 
    res[idx.inequality_dual] = p_data.inequality 
    for (i, ii) in enumerate(idx.inequality_dual) 
        res[ii] -= s[i]
    end

    # s ∘ t 
    for (i, ii) in enumerate(idx.inequality_slack_dual) 
        res[ii] = s[i] * t[i] - κ[1]
    end

    return 
end

function residual_symmetric!(residual_symmetric, residual, matrix, idx::Indices)
    # reset
    fill!(residual_symmetric, 0.0)

    rx = @views residual[idx.variables]
    rr = @views residual[idx.equality_slack]
    rs = @views residual[idx.inequality_slack]
    ry = @views residual[idx.equality_dual]
    rz = @views residual[idx.inequality_dual]
    rt = @views residual[idx.inequality_slack_dual]

    residual_symmetric[idx.variables] = rx
    residual_symmetric[idx.symmetric_equality] = ry
    residual_symmetric[idx.symmetric_inequality] = rz

    # equality correction 
    for (i, ii) in enumerate(idx.symmetric_equality)
        residual_symmetric[ii] += rr[i] / matrix[idx.equality_slack[i], idx.equality_slack[i]]
    end
 
    # inequality correction
    for (i, ii) in enumerate(idx.symmetric_inequality) 
        S̄i = matrix[idx.inequality_slack_dual[i], idx.inequality_slack_dual[i]] 
        Ti = matrix[idx.inequality_slack_dual[i], idx.inequality_slack[i]]
        Pi = matrix[idx.inequality_slack[i], idx.inequality_slack[i]] 
        residual_symmetric[ii] += (rt[i] + S̄i * rs[i]) / (Ti + S̄i * Pi)
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
    step[idx.equality_dual] = Δy
    step[idx.inequality_dual] = Δz

    # recover Δr, Δs, Δt
    Δr = @views step[idx.equality_slack]
    Δs = @views step[idx.inequality_slack]
    Δt = @views step[idx.inequality_slack_dual]
    rr = @views residual[idx.equality_slack] 
    rs = @views residual[idx.inequality_slack] 
    rt = @views residual[idx.inequality_slack_dual]

    # Δr 
    for (i, ii) in enumerate(idx.equality_slack)
        Δr[i] = (rr[i] + Δy[i]) / matrix[idx.equality_slack[i], idx.equality_slack[i]]
    end
   
    # Δs, Δt
    for (i, ii) in enumerate(idx.inequality_slack)
        S̄i = matrix[idx.inequality_slack_dual[i], idx.inequality_slack_dual[i]] 
        Ti = matrix[idx.inequality_slack_dual[i], idx.inequality_slack[i]]
        Pi = matrix[idx.inequality_slack[i], idx.inequality_slack[i]]  
        Di = matrix[idx.inequality_dual[i], idx.inequality_dual[i]]
        
        Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
        Δt[i] = (rt[i] - t[i] * Δs[i]) / S̄i
    end
    
    return 
end