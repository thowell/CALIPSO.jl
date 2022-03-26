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

function matrix!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ)
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

            if i == j 
                # regularization 
                # H[i, j] += 
            end
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

    return
end

function matrix_symmetric!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ)
    # dimensions 
    num_variables = length(idx.variables)
    num_equality = length(idx.equality) 
    num_inequality = length(idx.inequality) 

    # slacks 
    s = @views w[idx.slack_primal]
    t = @views w[idx.slack_dual]

    # reset
    H = s_data.matrix_symmetric
    fill!(H, 0.0)

    # Hessian of Lagrangian
    for i in idx.variables 
        for j in idx.variables 
            H[i, j]  = p_data.objective_hessian[i, j] 
            H[i, j] += p_data.equality_hessian[i, j]
            H[i, j] += p_data.inequality_hessian[i, j]

            if i == j 
                # regularization 
                # H[i, j] += 
            end
        end
    end

    # equality Jacobian 
    for (i, ii) in enumerate(idx.symmetric_equality) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.equality_jacobian[i, j]
            H[jj, ii] = p_data.equality_jacobian[i, j]
        end
    end

    # augmented Lagrangian 
    for (i, ii) in enumerate(idx.symmetric_equality)
        H[ii, ii] -= 1.0 / ρ[1]
    end

    # inequality Jacobian 
    for (i, ii) in enumerate(idx.symmetric_inequality) 
        for (j, jj) in enumerate(idx.variables)
            H[ii, jj] = p_data.inequality_jacobian[i, j]
            H[jj, ii] = p_data.inequality_jacobian[i, j]
        end
    end

    # -T \ S block 
    for (i, ii) in enumerate(idx.symmetric_inequality)    
        H[ii, ii] -= 1.0 * s[i] / t[i]
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

function residual_symmetric!(s_data::SolverData, p_data::ProblemData, idx::Indices, w, κ, ρ, λ;
    mode=:recompute)

    # dimensions 
    num_variables = length(idx.variables)
    num_equality = length(idx.equality) 
    num_inequality = length(idx.inequality) 

    # duals 
    y = @views w[idx.equality]
    z = @views w[idx.inequality]

    # slacks 
    s = @views w[idx.slack_primal]
    t = @views w[idx.slack_dual]

    # reset
    r = s_data.residual_symmetric 
    fill!(r, 0.0)

    if mode == :recompute
        residual!(s_data, p_data, idx, w, κ, ρ, λ)
    end

    rp = @views s_data.residual[idx.variables]
    re = @views s_data.residual[idx.equality]
    ri = @views s_data.residual[idx.inequality]

    r[idx.variables] = rp 
    r[idx.symmetric_equality] = re
    r[idx.symmetric_inequality] = ri
 
    # inequality correction
    for (i, ii) in enumerate(idx.symmetric_inequality) 
        r[ii] -= s[i] * z[i] / t[i] 
        r[ii] -= κ[1] / t[i]
    end

    return 
end

function step!(step, data::SolverData)
    fill!(step, 0.0)
    step .= data.matrix \ data.residual
    return 
end

function step_symmetric!(step, solver::LinearSolver, data::SolverData, idx::Indices, w, κ)
    # reset
    fill!(step, 0.0) 
    fill!(data.step_symmetric, 0.0)
    
    # solve symmetric system
    linear_solve!(solver, data.step_symmetric, data.matrix_symmetric, data.residual_symmetric)
    
    # set Δx, Δy, Δz
    Δx = @views data.step_symmetric[idx.variables]
    Δy = @views data.step_symmetric[idx.symmetric_equality]
    Δz = @views data.step_symmetric[idx.symmetric_inequality]
    step[idx.variables] = Δx
    step[idx.equality] = Δy
    step[idx.inequality] = Δz

    # recover Δs, Δt
    z = @views w[idx.inequality]
    s = @views w[idx.slack_primal]
    t = @views w[idx.slack_dual]
    Δs = @views step[idx.slack_primal]
    Δt = @views step[idx.slack_dual]
    num_inequality = length(z) 

    # Δt = z + t - Δz
    # Δs = s - κ[1] ./ t - s.* Δt ./ t
    for i = 1:num_inequality
        Δt[i] = z[i] + t[i] - Δz[i]
        Δs[i] = s[i] - κ[1] / t[i] - s[i] * Δt[i] / t[i]
    end
    
    return 
end