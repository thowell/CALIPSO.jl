function iterative_refinement!(step, solver::Solver)
    # reset 
    fill!(solver.data.step_correction, 0.0) 
    fill!(solver.data.residual_error, 0.0)
    
    # residual error
    solver.data.residual_error .= solver.data.residual - solver.data.matrix * step

    for i = solver.options.max_iterative_refinement
        step_symmetric!(solver.data.step_correction, solver.data.residual_error, solver.data.matrix, 
            solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.matrix_symmetric, 
            solver.indices, solver.linear_solver)
        
        step .+= solver.data.step_correction 
        solver.data.residual_error .= solver.data.residual - solver.data.matrix * step

        @show norm(solver.data.residual_error)
    
        if norm(solver.data.residual_error) < solver.options.iterative_refinement_tolerance
            return 
        end
    end

    @warn "iterative refinement failure"
    return 
end