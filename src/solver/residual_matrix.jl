function matrix!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ, ϵp, ϵd)
    # slacks 
    r = @views w[idx.equality_slack]
    s = @views w[idx.cone_slack]
    t = @views w[idx.cone_slack_dual]

    # reset
    H = s_data.matrix 
    fill!(H, 0.0)

    # Hessian of Lagrangian
    for i in idx.variables 
        for j in idx.variables 
            H[i, j]  = p_data.objective_hessian[i, j] 
            H[i, j] += p_data.equality_hessian[i, j]
            H[i, j] += p_data.cone_hessian[i, j]
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

    for (i, ii) in enumerate(idx.cone_slack) 
        for (j, jj) in enumerate(idx.cone_dual) 
            if i == j
                H[ii, jj] = -1.0 
                H[jj, ii] = -1.0 
            end
        end
    end

    for (i, ii) in enumerate(idx.cone_slack) 
        for (j, jj) in enumerate(idx.cone_slack_dual) 
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

    # cone Jacobian 
    for (i, ii) in enumerate(idx.cone_dual) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.cone_jacobian[i, j]
            H[jj, ii] = p_data.cone_jacobian[i, j]
        end
    end

    # augmented Lagrangian block 
    for (i, ii) in enumerate(idx.equality_slack) 
        H[ii, ii] = ρ[1]
    end

    # S block 
    for (i, ii) in enumerate(idx.cone_slack_dual)
        H[ii, ii] = s[i] 
    end

    # regularization 
    for i in idx.variables 
        H[i, i] += ϵp 
    end

    for i in idx.equality_slack
        H[i, i] += ϵp
    end 

    for i in idx.cone_slack
        H[i, i] += ϵp
    end 

    for i in idx.equality_dual
        H[i, i] -= ϵd
    end

    for i in idx.cone_dual
        H[i, i] -= ϵd 
    end

    for i in idx.cone_slack_dual 
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

    # cone Jacobian 
    for (i, ii) in enumerate(idx.cone_dual) 
        for (j, jj) in enumerate(idx.variables)
            matrix_symmetric[idx.symmetric_cone[i], jj] = matrix[ii, jj]
            matrix_symmetric[jj, idx.symmetric_cone[i]] = matrix[ii, jj]
        end
    end

    # -T \ S block | -(T + S̄P) \ S̄ + D
    for (i, ii) in enumerate(idx.cone_slack_dual)
        S̄i = matrix[ii, ii] 
        Ti = matrix[ii, idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]] 
        Di = matrix[idx.cone_dual[i], idx.cone_dual[i]]
        matrix_symmetric[idx.symmetric_cone[i], idx.symmetric_cone[i]] += -1.0 * S̄i / (Ti + S̄i * Pi) + Di
    end
   
    return
end