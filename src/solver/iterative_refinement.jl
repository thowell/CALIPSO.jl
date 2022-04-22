function iterative_refinement!(step, solver::Solver)
    # reset 
    fill!(solver.data.step_correction.all, 0.0) 
    fill!(solver.data.residual_error.all, 0.0)
    iteration = 0 

    # cache step (in case of failure)
    step_copy = deepcopy(step)

    # residual error
    solver.data.residual_error.all .= solver.data.residual.all - solver.data.jacobian_variables * step.all
    residual_norm = norm(solver.data.residual.all, Inf)

    while (iteration < solver.options.max_iterative_refinement && residual_norm > solver.options.iterative_refinement_tolerance) || iteration < solver.options.min_iterative_refinement
       
        # @show norm(solver.data.residual_error)
        norm(solver.data.residual_error.all) < solver.options.iterative_refinement_tolerance && return 
        # correction
        search_direction_symmetric!(solver.data.step_correction, solver.data.residual_error, solver.data.jacobian_variables, 
            solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.jacobian_variables_symmetric, 
            solver.indices, solver.linear_solver)
        
        # update
        step.all .+= solver.data.step_correction.all 

         # residual error
        solver.data.residual_error.all .= solver.data.residual.all - solver.data.jacobian_variables * step.all
        residual_norm = norm(solver.data.residual.all, Inf)

        iteration += 1
    end

    if residual_norm < solver.options.iterative_refinement_tolerance
        return true
    else
        # solver.options.verbose && println("iterative refinement failure")
        search_direction_nonsymmetric!(solver.data.step, solver.data)
        # step .= step_copy
        return false
    end
end