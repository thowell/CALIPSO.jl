function residual_jacobian_variables!(data::SolverData, problem::ProblemData, idx::Indices, κ, ρ, λ, ϵp, ϵd;
    constraint_tensor=true)
     
    # reset
    H = data.jacobian_variables 
    fill!(H.nzval, 0.0)

    # Hessian of Lagrangian
    for i in idx.variables 
        for j in idx.variables 
            H[i, j]  = problem.objective_jacobian_variables_variables[i, j] 
            constraint_tensor && (H[i, j] += problem.equality_dual_jacobian_variables_variables[i, j])
            constraint_tensor && (H[i, j] += problem.cone_dual_jacobian_variables_variables[i, j])
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
            H[ii, jj] = problem.equality_jacobian_variables[i, j]
            H[jj, ii] = problem.equality_jacobian_variables[i, j]
        end
    end

    # cone Jacobian 
    for (i, ii) in enumerate(idx.cone_dual) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = problem.cone_jacobian_variables[i, j]
            H[jj, ii] = problem.cone_jacobian_variables[i, j]
        end
    end

    # augmented Lagrangian block 
    for (i, ii) in enumerate(idx.equality_slack) 
        H[ii, ii] = ρ[1]
    end
    
    # cone block (nonnegative)
    for i in idx.cone_nonnegative
        H[idx.cone_slack_dual[i], idx.cone_slack[i]] = problem.cone_product_jacobian_primal[i, i] 
        H[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] = problem.cone_product_jacobian_dual[i, i]  
    end

    # cone block (second-order)
    for idx_soc in idx.cone_second_order
        if !isempty(idx_soc)
            for i in idx_soc 
                for j in idx_soc 
                    H[idx.cone_slack_dual[i], idx.cone_slack[j]] = problem.cone_product_jacobian_primal[i, j]
                    H[idx.cone_slack_dual[i], idx.cone_slack_dual[j]] = problem.cone_product_jacobian_dual[i, j] 
                end 
            end
        end
    end

    # regularization 
    for i in idx.variables 
        H[i, i] += ϵp[1] 
    end

    for i in idx.equality_slack
        H[i, i] += ϵp[1]
    end 

    for i in idx.cone_slack
        H[i, i] += ϵp[1]
    end 

    for i in idx.equality_dual
        H[i, i] -= ϵd[1]
    end

    for i in idx.cone_dual
        H[i, i] -= ϵd[1] 
    end

    for i in idx.cone_slack_dual 
        H[i, i] -= ϵd[1] 
    end 

    return
end

function residual_jacobian_variables_symmetric!(matrix_symmetric, matrix, idx::Indices, second_order_jacobians, second_order_jacobians_inverse)
    # reset
    fill!(matrix_symmetric.nzval, 0.0)

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

    # cone correction (nonnegative)
    for i in idx.cone_nonnegative
        S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] 
        Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]] 
        Di = matrix[idx.cone_dual[i], idx.cone_dual[i]]
        matrix_symmetric[idx.symmetric_cone[i], idx.symmetric_cone[i]] += -1.0 * S̄i / (Ti + S̄i * Pi) + Di
    end

    # cone correction (second-order)
    for (i, idx_soc) in enumerate(idx.cone_second_order)
        if !isempty(idx_soc)
            C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] 
            Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
            P = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]] 
            D = @views matrix[idx.cone_dual[idx_soc], idx.cone_dual[idx_soc]]
            
            for (i, ii) in enumerate(idx_soc) 
                matrix_symmetric[idx.symmetric_cone[idx_soc], idx.symmetric_cone[ii]] -= second_order_matrix_inverse((Cs + C̄t * P), C̄t[:, i])
            end
            matrix_symmetric[idx.symmetric_cone[idx_soc], idx.symmetric_cone[idx_soc]] += D
        end
    end

    return
end