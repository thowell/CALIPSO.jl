# methods
function ProblemMethods(trajopt::TrajectoryOptimizationProblem) 
    ProblemMethods(
        (z, θ) -> objective!(trajopt, z),
        (g, z, θ) -> objective_gradient!(g, trajopt, z),
        (g, z, θ) -> nothing,
        (h, z, θ) -> objective_hessian!(h, trajopt, z),
        (h, z, θ) -> nothing,
        (c, z, θ) -> equality!(c, trajopt, z), 
        (j, z, θ) -> equality_jacobian!(j, trajopt, z),
        (j, z, θ) -> nothing,
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> begin 
            j = zeros(trajopt.dimensions.total_equality, trajopt.dimensions.total_variables); 
            equality_jacobian!(j, trajopt, z); 
            v .= j' * y; 
        end,
        (h, z, θ, y) -> equality_hessian!(h, trajopt, z, y),
        (h, z, θ, y) -> nothing,
        (c, z, θ) -> cone!(c, trajopt, z),
        (j, z, θ) -> cone_jacobian!(j, trajopt, z),
        (j, z, θ) -> nothing,
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> begin 
            j = zeros(trajopt.dimensions.total_cone, trajopt.dimensions.total_variables); 
            cone_jacobian!(j, trajopt, z); 
            v .= j' * y; 
        end,
        (h, z, θ, y) -> cone_hessian!(h, trajopt, z, y),
        (h, z, θ, y) -> nothing,
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
