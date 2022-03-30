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