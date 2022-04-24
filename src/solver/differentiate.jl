function differentiate!(solver)  
    # evaluate derivatives wrt to parameters 
    CALIPSO.problem!(solver.problem, solver.methods, solver.indices, solver.solution, solver.parameters,
        objective_jacobian_variables_parameters=true,
        equality_jacobian_parameters=true,
        equality_dual_jacobian_variables_parameters=true,
        cone_jacobian_parameters=true,
        cone_dual_jacobian_variables_parameters=true,
    )

    # TODO: check if we can use current residual Jacobian w/o recomputing
    # residual Jacobian wrt variables 
    residual_jacobian_variables!(solver.data, solver.problem, solver.indices, 
        solver.central_path, solver.penalty, solver.dual,
        solver.primal_regularization, solver.dual_regularization,
        constraint_hessian=solver.options.constraint_hessian)
    residual_jacobian_variables_symmetric!(solver.data.jacobian_variables_symmetric, solver.data.jacobian_variables, solver.indices, 
        solver.problem.second_order_jacobians, solver.problem.second_order_jacobians_inverse) 
    factorize!(solver.linear_solver, solver.data.jacobian_variables_symmetric; 
        update=solver.options.update_factorization)
 
    # residual Jacobian wrt parameters
    residual_jacobian_parameters!(solver.data, solver.problem, solver.indices)

    # compute solution sensitivities
    fill!(solver.data.solution_sensitivity, 0.0)
 
    #TODO parallelize, make more efficient
    for i in solver.indices.parameters 
        for k = 1:solver.dimensions.total 
            solver.data.jacobian_parameters_vectors[i].all[k] = solver.data.jacobian_parameters[k, i]
        end
     
        search_direction_symmetric!(
            solver.data.solution_sensitivity_vectors[i],
            solver.data.jacobian_parameters_vectors[i],
            solver.data.jacobian_variables, 
            solver.data.step_symmetric, 
            solver.data.residual_symmetric, 
            solver.data.jacobian_variables_symmetric, 
            solver.indices, 
            solver.linear_solver)
 
        for (k, s) in enumerate(solver.data.solution_sensitivity_vectors[i].all) 
            solver.data.solution_sensitivity[k, i] = -1.0 * s
        end
    end

    return 
end

