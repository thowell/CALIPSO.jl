function matrix!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ, ϵp, ϵd;
    constraint_hessian=true)
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
            constraint_hessian && (H[i, j] += p_data.equality_hessian[i, j])
            constraint_hessian && (H[i, j] += p_data.cone_hessian[i, j])
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
    
    # cone block (non-negative)
    for i in idx.cone_nonnegative
        H[idx.cone_slack_dual[i], idx.cone_slack[i]] = p_data.cone_product_jacobian_primal[i, i] 
        H[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] = p_data.cone_product_jacobian_dual[i, i]  
    end

    # cone block (second-order)
    for idx_soc in idx.cone_second_order
        Cs = @views p_data.cone_product_jacobian_primal[idx_soc, idx_soc] 
        Ct = @views p_data.cone_product_jacobian_dual[idx_soc, idx_soc] 
        H[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]] = Cs 
        H[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] = Ct 
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

    # cone correction (non-negative)
    for i in idx.cone_nonnegative
        S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] 
        Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]] 
        Di = matrix[idx.cone_dual[i], idx.cone_dual[i]]
        matrix_symmetric[idx.symmetric_cone[i], idx.symmetric_cone[i]] += -1.0 * S̄i / (Ti + S̄i * Pi) + Di
    end

    # cone correction (second-order)
    for idx_soc in idx.cone_second_order
        C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] 
        Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
        P = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]] 
        D = @views matrix[idx.cone_dual[idx_soc], idx.cone_dual[idx_soc]]
        matrix_symmetric[idx.symmetric_cone[idx_soc], idx.symmetric_cone[idx_soc]] += -1.0 * (Cs + C̄t * P) \ C̄t  + D
    end

    return
end