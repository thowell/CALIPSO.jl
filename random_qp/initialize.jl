function initialize_variables!(variables, guess, idx::Indices)
    variables[idx.primal] = guess 
    return 
end

function initialize_duals!(variables, idx::Indices)
    variables[idx.equality] .= 0.0
    variables[idx.inequality] .= 0.0
    variables[idx.slack_primal] .= 1.0 
    variables[idx.slack_dual] .= 1.0 
    return 
end

function initialize_interior_point!(central_path, options::Options)
    central_path[1] = options.central_path_initial
    return 
end

function initialize_augmented_lagrangian!(penalty, dual, options::Options)
    penalty[1] = options.penalty_initial 
    dual .= options.dual_initial
    return 
end

# solver 
function initialize!(solver::Solver, x)
    # variables 
    initialize_variables!(solver.variables, x, solver.indices)
end
