function search_direction!(step, data::SolverData)
    fill!(step, 0.0)
    step .= data.matrix \ data.residual
    return 
end

function search_direction_symmetric!(step, residual, matrix, step_symmetric, residual_symmetric, matrix_symmetric, idx::Indices, solver::LinearSolver)
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
    Δz = @views step_symmetric[idx.symmetric_cone]
    step[idx.variables] = Δx
    step[idx.equality_dual] = Δy
    step[idx.cone_dual] = Δz

    # recover Δr, Δs, Δt
    Δr = @views step[idx.equality_slack]
    Δs = @views step[idx.cone_slack]
    Δt = @views step[idx.cone_slack_dual]
    rr = @views residual[idx.equality_slack] 
    rs = @views residual[idx.cone_slack] 
    rt = @views residual[idx.cone_slack_dual]

    # Δr 
    for (i, ii) in enumerate(idx.equality_slack)
        Δr[i] = (rr[i] + Δy[i]) / matrix[idx.equality_slack[i], idx.equality_slack[i]]
    end
   
    # Δs, Δt (non-negative)
    for i in idx.cone_nonnegative
        S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] 
        Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]]  
        
        Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
        Δt[i] = (rt[i] - Ti * Δs[i]) / S̄i
    end

    # Δs, Δt (second-order)
    for idx_soc in idx.cone_second_order
        C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] 
        Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
        P  = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]] 
        rs_soc = @views rs[idx_soc] 
        rt_soc = @views rt[idx_soc] 
        Δz_soc = @views Δz[idx_soc] 
        Δs_soc = @views Δs[idx_soc]

        Δs[idx_soc] = (Cs + C̄t * P) \ (rt_soc + C̄t * (rs_soc + Δz_soc))
        Δt[idx_soc] = C̄t \ (rt_soc - Cs * Δs_soc)
    end
    
    return 
end