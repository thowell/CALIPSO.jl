# solver 
function initialize!(solver::Solver, guess)
    # variables 
    solver.solution.variables .= guess 
    return
end

function initialize_slacks!(solver)
    # set slacks to constraints
    problem!(solver.problem, solver.methods, solver.indices, solver.solution, solver.parameters,
        equality_constraint=true,
        cone_constraint=true,
    )

    for (i, idx) in enumerate(solver.indices.equality_slack)
        solver.solution.equality_slack[i] = solver.problem.equality_constraint[i]
    end

    initialize_cone!(solver.solution.cone_slack, solver.indices.cone_nonnegative, solver.indices.cone_second_order) 

    return 
end

function initialize_duals!(solver)
    solver.solution.equality_dual .= 0.0
    solver.solution.cone_dual .= 0.0
    initialize_cone!(solver.solution.cone_slack_dual, solver.indices.cone_nonnegative, solver.indices.cone_second_order) 
    return 
end

function initialize_interior_point!(solver)
    solver.central_path[1] = solver.options.central_path_initial
    solver.fraction_to_boundary[1] = max(0.99, 1.0 - solver.central_path[1])
    return 
end

function initialize_augmented_lagrangian!(solver)
    solver.penalty[1] = solver.options.penalty_initial 
    solver.dual .= solver.options.dual_initial
    return 
end
