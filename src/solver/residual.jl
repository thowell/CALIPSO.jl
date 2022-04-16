function residual!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ)
    # duals 
    y = @views w[idx.equality_dual]
    z = @views w[idx.cone_dual]
    num_equality = length(y) 
    num_cone = length(z)

    # slacks 
    r = @views w[idx.equality_slack]
    s = @views w[idx.cone_slack]
    t = @views w[idx.cone_slack_dual]

    # reset
    res = s_data.residual 
    fill!(res, 0.0)

    # gradient of Lagrangian 
    res[idx.variables] = p_data.objective_gradient_variables

    for (i, ii) in enumerate(idx.variables)
        cy = 0.0
        # for j = 1:num_equality 
        #     cy += p_data.equality_jacobian_variables[j, i] * y[j]
        # end
        # res[ii] += cy 

        # cz = 0.0
        # for k = 1:num_cone 
        #     cz += p_data.cone_jacobian_variables[k, i] * z[k]
        # end
        # res[ii] += cz

        res[ii] += p_data.equality_dual_jacobian_variables[i] 
        res[ii] += p_data.cone_dual_jacobian_variables[i]
    end
    
    # λ + ρr - y 
    for (i, ii) in enumerate(idx.equality_slack) 
        res[ii] = λ[i] + ρ[1] * r[i] - y[i]
    end

    # -z - t
    for (i, ii) in enumerate(idx.cone_slack)
        res[ii] = -z[i] - t[i]
    end

    # equality 
    res[idx.equality_dual] = p_data.equality_constraint
    for (i, ii) in enumerate(idx.equality_dual)
        res[ii] -= r[i]
    end

    # cone 
    res[idx.cone_dual] = p_data.cone_constraint 
    for (i, ii) in enumerate(idx.cone_dual) 
        res[ii] -= s[i]
    end

    # s ∘ t - κ e
    for (i, ii) in enumerate(idx.cone_slack_dual) 
        res[ii] = p_data.cone_product[i] - κ[1] * p_data.cone_target[i]
    end

    return 
end

function residual_symmetric!(residual_symmetric, residual, matrix, idx::Indices)
    # reset
    fill!(residual_symmetric, 0.0)

    rx = @views residual[idx.variables]
    rr = @views residual[idx.equality_slack]
    rs = @views residual[idx.cone_slack]
    ry = @views residual[idx.equality_dual]
    rz = @views residual[idx.cone_dual]
    rt = @views residual[idx.cone_slack_dual]

    residual_symmetric[idx.variables] = rx
    residual_symmetric[idx.symmetric_equality] = ry
    residual_symmetric[idx.symmetric_cone] = rz

    # equality correction 
    for (i, ii) in enumerate(idx.symmetric_equality)
        residual_symmetric[ii] += rr[i] / matrix[idx.equality_slack[i], idx.equality_slack[i]]
    end
 
    # cone correction (nonnegative)
    for i in idx.cone_nonnegative
        S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] 
        Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]] 
        residual_symmetric[idx.symmetric_cone[i]] += (rt[i] + S̄i * rs[i]) / (Ti + S̄i * Pi)
    end

    # cone correction (second-order)
    for idx_soc in idx.cone_second_order 
        C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] 
        Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
        P  = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]] 
        residual_symmetric[idx.symmetric_cone[idx_soc]] += (Cs + C̄t * P) \ (C̄t * rt[idx_soc] + rt[idx_soc])
    end

    return 
end
