# methods
function ProblemMethods(trajopt::TrajectoryOptimizationProblem) 
    ProblemMethods(
        (c, z, θ) -> objective!(c, trajopt, z, θ),
        (g, z, θ) -> objective_gradient_variables!(g, trajopt, z, θ),
        (g, z, θ) -> objective_gradient_parameters!(g, trajopt, z, θ),
        (j, z, θ) -> objective_jacobian_variables_variables!(j, trajopt, z, θ),
        (j, z, θ) -> objective_jacobian_variables_parameters!(j, trajopt, z, θ),
        (c, z, θ) -> equality!(c, trajopt, z, θ), 
        (j, z, θ) -> equality_jacobian_variables!(j, trajopt, z, θ),
        (j, z, θ) -> equality_jacobian_parameters!(j, trajopt, z, θ),
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> equality_dual_jacobian_variables!(v, trajopt, z, y, θ),
        (h, z, θ, y) -> equality_jacobian_variables_variables!(h, trajopt, z, y, θ),
        (h, z, θ, y) -> equality_jacobian_variables_parameters!(h, trajopt, z, y, θ),
        (c, z, θ) -> cone!(c, trajopt, z, θ),
        (j, z, θ) -> cone_jacobian_variables!(j, trajopt, z, θ),
        (j, z, θ) -> cone_jacobian_parameters!(j, trajopt, z, θ),
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> cone_dual_jacobian_variables!(v, trajopt, z, y, θ),
        (h, z, θ, y) -> cone_jacobian_variables_variables!(h, trajopt, z, y, θ),
        (h, z, θ, y) -> cone_jacobian_variables_parameters!(h, trajopt, z, y, θ),
    )
end

function cone_indices(trajopt::TrajectoryOptimizationProblem) 
    idx_nonnegative = vcat(trajopt.indices.nonnegative_duals...)
    idx_second_order = [(trajopt.indices.second_order_duals...)...]
    return idx_nonnegative, idx_second_order
end

function initialize_states!(solver::Solver, trajopt::TrajectoryOptimizationProblem, states) 
    for (t, idx) in enumerate(trajopt.indices.states)
        solver.solution.variables[idx] = states[t]
    end
end

function initialize_controls!(solver::Solver, trajopt::TrajectoryOptimizationProblem, actions) 
    for (t, idx) in enumerate(trajopt.indices.actions)
        solver.solution.variables[idx] = actions[t]
    end
end

function get_trajectory(solver::Solver, trajopt::TrajectoryOptimizationProblem) 
    states = [solver.solution.variables[idx] for (t, idx) in enumerate(trajopt.indices.states)]
    actions = [solver.solution.variables[idx] for (t, idx) in enumerate(trajopt.indices.actions)] 
    return states, actions
end
