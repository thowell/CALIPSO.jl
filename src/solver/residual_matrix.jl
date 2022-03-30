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