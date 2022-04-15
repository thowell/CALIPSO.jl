function residual_jacobian_parameters!(data::SolverData, problem::ProblemData, idx::Indices)
  
    # reset
    H = data.jacobian_parameters 
    fill!(H, 0.0)

    # Lxθ
    for i in enumerate(idx.variables) 
        for j in enumerate(idx.parameters)
            H[i, j] = problem.objective_jacobian_variables_parameters[i, j] 
            H[i, j] += problem.equality_dual_jacobian_variables_parameters[i, j]
            H[i, j] += problem.cone_dual_jacobian_variables_parameters[i, j]
        end
    end

    # Lrθ 
    nothing 

    # Lsθ 
    nothing 

    # Lyθ 
    for (i, ii) in enumerate(idx.equality_dual)
        for (j, jj) in idx.parameters 
            H[ii, jj] = problem.equality_jacobian_parameters[i, j] 
        end
    end
    
    # Lzθ 
    for (i, ii) in enumerate(idx.cone_dual)
        for (j, jj) in idx.parameters 
            H[ii, jj] = problem.cone_jacobian_parameters[i, j] 
        end
    end

    # Ltθ 
    nothing

    return
end

