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
        Δt[i] = (rt[i] - Ti * Δs[i]) / S̄i
    end
    
    return 
end