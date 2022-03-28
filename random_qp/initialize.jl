function initialize_primals!(variables, guess, idx::Indices)
    variables[idx.variables] = guess 
    variables[idx.equality_slack] .= 0.0
    variables[idx.inequality_slack] .= 1.0 
    return 
end

function initialize_duals!(variables, idx::Indices)
    variables[idx.equality_dual] .= 0.0
    variables[idx.inequality_dual] .= 0.0
    variables[idx.inequality_slack_dual] .= 1.0 
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
    initialize_primals!(solver.variables, x, solver.indices)
end
