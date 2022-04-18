function search_direction!(solver) 
    # correct inertia
    inertia_correction!(solver)

    # compute search direction
    search_direction_symmetric!(solver.data.step, solver.data.residual, solver.data.jacobian_variables, 
        solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.jacobian_variables_symmetric, 
        solver.indices, solver.linear_solver;
        update=solver.options.update_factorization)

    # refine search direction
    solver.options.iterative_refinement && iterative_refinement!(solver.data.step, solver)
end

function search_direction_symmetric!(step, residual, matrix, step_symmetric, residual_symmetric, matrix_symmetric, idx::Indices, solver::LinearSolver;
    update=true)
    # reset
    # fill!(step, 0.0) 
    # fill!(step_symmetric, 0.0)
    
    # solve symmetric system
    residual_symmetric!(residual_symmetric, residual, matrix, idx) 
    # residual_jacobian_variables_symmetric!(matrix_symmetric, matrix, idx) 
    
    linear_solve!(solver, step_symmetric, matrix_symmetric, residual_symmetric;
        update=update)
    
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
   
    # Δs, Δt (nonnegative)
    for i in idx.cone_nonnegative
        S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] 
        Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
        Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]]  
        
        Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
        Δt[i] = (rt[i] - Ti * Δs[i]) / S̄i
    end

    # Δs, Δt (second-order)
    for idx_soc in idx.cone_second_order
        C̄t = matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] 
        Cs = matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
        P  = matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]] 
        rs_soc = @views rs[idx_soc] 
        rt_soc = @views rt[idx_soc] 
        Δz_soc = @views Δz[idx_soc] 
        Δs_soc = @views Δs[idx_soc]

        Δs[idx_soc] = (Cs + C̄t * P) \ (rt_soc + C̄t * (rs_soc + Δz_soc))
        Δt[idx_soc] = C̄t \ (rt_soc - Cs * Δs_soc)
    end
    
    return 
end

function search_direction_nonsymmetric!(step, data::SolverData)
    # fill!(step, 0.0)
    step .= data.jacobian_variables \ data.residual
    return 
end