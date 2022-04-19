function differentiate!(solver) 
    # evaluate derivatives wrt to parameters 
    CALIPSO.problem!(solver.problem, solver.methods, solver.indices, solver.variables, solver.parameters,
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
    residual_jacobian_variables_symmetric!(solver.data.jacobian_variables_symmetric, solver.data.jacobian_variables, solver.indices) 
    factorize!(solver.linear_solver, solver.data.jacobian_variables_symmetric; 
        update=solver.options.update_factorization)

    # residual Jacobian wrt parameters
    residual_jacobian_parameters!(solver.data, solver.problem, solver.indices)

    # compute solution sensitivities
    fill!(solver.data.solution_sensitivity, 0.0)
    for i in solver.indices.parameters 
        ∂z∂pi = @views solver.data.solution_sensitivity[:, i]
        Rpi   = @views solver.data.jacobian_parameters[:, i]
        search_direction_symmetric!(∂z∂pi, Rpi, 
            solver.data.jacobian_variables, 
            solver.data.step_symmetric, 
            solver.data.residual_symmetric, 
            solver.data.jacobian_variables_symmetric, 
            solver.indices, 
            solver.linear_solver)
        ∂z∂pi .*= -1.0
    end

    return 
end

