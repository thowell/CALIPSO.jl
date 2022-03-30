function solve!(solver::Solver)
    # initialize 
    initialize_duals!(solver.variables, solver.indices)
    initialize_interior_point!(solver.central_path, solver.options)
    initialize_augmented_lagrangian!(solver.penalty, solver.dual, solver.options)

    # compute residual 
    problem!(solver.problem, solver.methods, solver.indices, solver.variables,
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=false)

    for i = 1:solver.options.max_outer_iterations
        residual!(solver.data, solver.problem, solver.indices, solver.variables, 
            solver.central_path, solver.penalty, solver.dual)

        residual_norm = norm(solver.data.residual, solver.options.residual_norm)

        for j = 1:solver.options.max_residual_iterations
            solver.options.verbose && println("outer iteration: $i, residual iteration: $j, residual norm: $residual_norm")
            
            # check convergence 
            residual_norm <= solver.options.residual_tolerance && (print("solve!"); break)

            # compute step 
            problem!(solver.problem, solver.methods, solver.indices, solver.variables,
                gradient=false,
                constraint=false,
                jacobian=false,
                hessian=true)

            inertia_correction!(solver)

            search_direction_symmetric!(solver.data.step, solver.data.residual, solver.data.matrix, 
                solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.matrix_symmetric, 
                solver.indices, solver.linear_solver)

            solver.options.iterative_refinement && iterative_refinement!(solver.data.step, solver)

            # candidate
            step_size = 1.0 
            solver.candidate .= solver.variables - step_size * solver.data.step
            
            s_candidate = @views solver.candidate[solver.indices.inequality_slack] 
            t_candidate = @views solver.candidate[solver.indices.inequality_slack_dual]

            # cone line search
            for i = 1:solver.options.max_cone_line_search 
                if positive_check(s_candidate) && positive_check(t_candidate)
                    break 
                else
                    step_size *= solver.options.scaling_line_search
                    solver.candidate .= solver.variables - step_size * solver.data.step
                    s_candidate = @views solver.candidate[solver.indices.inequality_slack] 
                    t_candidate = @views solver.candidate[solver.indices.inequality_slack_dual]

                    i == solver.options.max_cone_line_search && error("cone line search failure")
                end
            end

            # residual line search
            for i = 1:solver.options.max_residual_line_search
                problem!(solver.problem, solver.methods, solver.indices, solver.candidate,
                    gradient=true,
                    constraint=true,
                    jacobian=true,
                    hessian=false)

                # compute residual
                residual!(solver.data, solver.problem, solver.indices, solver.candidate, 
                    solver.central_path, solver.penalty, solver.dual)
                residual_candidate_norm = norm(solver.data.residual, solver.options.residual_norm)
                
                if residual_candidate_norm < residual_norm 
                    solver.variables .= solver.candidate 
                    residual_norm = residual_candidate_norm 
                    break
                else
                    step_size *= solver.options.scaling_line_search
                    solver.candidate .= solver.variables - step_size * solver.data.step
                
                    # i == solver.options.max_residual_line_search && error("residual line search failure")
                end
            end
        end

        if solver.central_path[1] < solver.options.central_path_tolerance && norm(solver.problem.equality, 1) / max(1, solver.dimensions.equality_dual) < solver.options.dual_tolerance 
            return true 
        else
            # interior-point 
            solver.central_path[1] *= solver.options.scaling_central_path 

            # augmented Lagrangian 
            solver.dual .+= solver.penalty[1] .* solver.problem.equality 
            solver.penalty[1] *= solver.options.scaling_penalty
        end
    end
    
    return false
end