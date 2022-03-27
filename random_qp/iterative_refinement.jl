function iterative_refinement!(step, solver::Solver)
    # reset 
    fill!(solver.data.step_correction, 0.0) 
    fill!(solver.data.residual_error, 0.0)
    iteration = 0 

    # cache step (in case of failure)
    step_copy = deepcopy(step)

    # residual error
    solver.data.residual_error .= solver.data.residual - solver.data.matrix * step
    residual_norm = norm(solver.data.residual, Inf)

    while (iteration < solver.options.max_iterative_refinement && residual_norm > solver.options.iterative_refinement_tolerance) || iteration < solver.options.min_iterative_refinement
       
        # @show norm(solver.data.residual_error)
        norm(solver.data.residual_error) < solver.options.iterative_refinement_tolerance && return 
        # correction
        step_symmetric!(solver.data.step_correction, solver.data.residual_error, solver.data.matrix, 
            solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.matrix_symmetric, 
            solver.indices, solver.linear_solver)
        
        # update
        step .+= solver.data.step_correction 

         # residual error
        solver.data.residual_error .= solver.data.residual - solver.data.matrix * step
        residual_norm = norm(solver.data.residual, Inf)

        iteration += 1
    end

    if residual_norm < solver.options.iterative_refinement_tolerance
        return true
    else
        step .= step_copy
        return false
    end
end