# methods
function ProblemMethods(trajopt::TrajectoryOptimizationProblem) 
    ProblemMethods(
        z -> objective!(trajopt, z),
        (g, z) -> objective_gradient!(g, trajopt, z),
        (h, z) -> objective_hessian!(h, trajopt, z),
        (c, z) -> equality!(c, trajopt, z), 
        (j, z) -> equality_jacobian!(j, trajopt, z),
        (h, z, y) -> equality_hessian!(h, trajopt, z, y),
        (c, z) -> cone!(c, trajopt, z),
        (j, z) -> cone_jacobian!(j, trajopt, z),
        (h, z, y) -> cone_hessian!(h, trajopt, z, y),
    )
end

function cone_indices(trajopt::TrajectoryOptimizationProblem) 
    idx_nonnegative = vcat(trajopt.indices.nonnegative_duals...)
    idx_second_order = [(trajopt.indices.second_order_duals...)...]
    return idx_nonnegative, idx_second_order
end

function initialize_states!(solver::Solver, trajopt::TrajectoryOptimizationProblem, states) 
    for (t, idx) in enumerate(trajopt.indices.states)
        solver.variables[solver.indices.variables[idx]] = states[t]
    end
end

function initialize_controls!(solver::Solver, trajopt::TrajectoryOptimizationProblem, actions) 
    for (t, idx) in enumerate(trajopt.indices.actions)
        solver.variables[solver.indices.variables[idx]] = actions[t]
    end
end

function get_trajectory(solver::Solver, trajopt::TrajectoryOptimizationProblem) 
    states = [solver.variables[solver.indices.variables[idx]] for (t, idx) in enumerate(trajopt.indices.states)]
    actions = [solver.variables[solver.indices.variables[idx]] for (t, idx) in enumerate(trajopt.indices.actions)] 
    return states, actions
end
