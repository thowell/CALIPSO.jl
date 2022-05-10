# methods
function ProblemMethods(trajopt::TrajectoryOptimizationProblem) 
    ProblemMethods(
        (c, z, θ) -> objective!(c, trajopt, z, θ),
        (g, z, θ) -> objective_gradient_variables!(g, trajopt, z, θ),
        (g, z, θ) -> objective_gradient_parameters!(g, trajopt, z, θ),
        (j, z, θ) -> objective_jacobian_variables_variables!(j, trajopt, z, θ),
        (j, z, θ) -> objective_jacobian_variables_parameters!(j, trajopt, z, θ),
        zeros(length(vcat(trajopt.sparsity.objective_jacobian_variables_variables...))), 
        zeros(length(vcat(trajopt.sparsity.objective_jacobian_variables_parameters...))), 
        vcat(trajopt.sparsity.objective_jacobian_variables_variables...),
        vcat(trajopt.sparsity.objective_jacobian_variables_parameters...),
        (c, z, θ) -> equality!(c, trajopt, z, θ), 
        (j, z, θ) -> equality_jacobian_variables!(j, trajopt, z, θ),
        (j, z, θ) -> equality_jacobian_parameters!(j, trajopt, z, θ),
        zeros(length(vcat(trajopt.sparsity.dynamics_jacobian_variables...)) + length(vcat(trajopt.sparsity.equality_jacobian_variables...)) + length(trajopt.sparsity.equality_general_jacobian_variables)), 
        zeros(length(vcat(trajopt.sparsity.dynamics_jacobian_parameters...)) + length(vcat(trajopt.sparsity.equality_jacobian_parameters...)) + length(trajopt.sparsity.equality_general_jacobian_parameters)),
        vcat(trajopt.sparsity.dynamics_jacobian_variables..., trajopt.sparsity.equality_jacobian_variables..., trajopt.sparsity.equality_general_jacobian_variables), 
        vcat(trajopt.sparsity.dynamics_jacobian_parameters..., trajopt.sparsity.equality_jacobian_parameters..., trajopt.sparsity.equality_general_jacobian_parameters),
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> equality_dual_jacobian_variables!(v, trajopt, z, y, θ),
        (h, z, θ, y) -> equality_jacobian_variables_variables!(h, trajopt, z, y, θ),
        (h, z, θ, y) -> equality_jacobian_variables_parameters!(h, trajopt, z, y, θ),
        zeros(length(vcat(trajopt.sparsity.dynamics_jacobian_variables_variables...)) + length(vcat(trajopt.sparsity.equality_jacobian_variables_variables...)) + length(trajopt.sparsity.equality_general_jacobian_variables_variables)), 
        zeros(length(vcat(trajopt.sparsity.dynamics_jacobian_variables_parameters...)) + length(vcat(trajopt.sparsity.equality_jacobian_variables_parameters...)) + length(trajopt.sparsity.equality_general_jacobian_variables_parameters)),
        vcat(trajopt.sparsity.dynamics_jacobian_variables_variables..., trajopt.sparsity.equality_jacobian_variables_variables..., trajopt.sparsity.equality_general_jacobian_variables_variables...), 
        vcat(trajopt.sparsity.dynamics_jacobian_variables_parameters..., trajopt.sparsity.equality_jacobian_variables_parameters..., trajopt.sparsity.equality_general_jacobian_variables_parameters...),
        (c, z, θ) -> cone!(c, trajopt, z, θ),
        (j, z, θ) -> cone_jacobian_variables!(j, trajopt, z, θ),
        (j, z, θ) -> cone_jacobian_parameters!(j, trajopt, z, θ),
        zeros(length(vcat(trajopt.sparsity.nonnegative_jacobian_variables...)) + length(vcat((trajopt.sparsity.second_order_jacobian_variables...)...))), 
        zeros(length(vcat(trajopt.sparsity.nonnegative_jacobian_parameters...)) + length(vcat((trajopt.sparsity.second_order_jacobian_parameters...)...))),
        vcat(trajopt.sparsity.nonnegative_jacobian_variables..., (trajopt.sparsity.second_order_jacobian_variables...)...), 
        vcat(trajopt.sparsity.nonnegative_jacobian_parameters..., (trajopt.sparsity.second_order_jacobian_parameters...)...),
        (f, z, θ, y) -> nothing,
        (v, z, θ, y) -> cone_dual_jacobian_variables!(v, trajopt, z, y, θ),
        (h, z, θ, y) -> cone_jacobian_variables_variables!(h, trajopt, z, y, θ),
        (h, z, θ, y) -> cone_jacobian_variables_parameters!(h, trajopt, z, y, θ),
        zeros(length(vcat(trajopt.sparsity.nonnegative_jacobian_variables_variables...)) + length(vcat((trajopt.sparsity.second_order_jacobian_variables_variables...)...))), 
        zeros(length(vcat(trajopt.sparsity.nonnegative_jacobian_variables_parameters...)) + length(vcat((trajopt.sparsity.second_order_jacobian_variables_parameters...)...))),
        vcat(trajopt.sparsity.nonnegative_jacobian_variables_variables..., (trajopt.sparsity.second_order_jacobian_variables_variables...)...), 
        vcat(trajopt.sparsity.nonnegative_jacobian_variables_parameters..., (trajopt.sparsity.second_order_jacobian_variables_parameters...)...),
    )
end

function cone_indices(trajopt::TrajectoryOptimizationProblem) 
    idx_nonnegative = vcat(trajopt.indices.nonnegative_duals...)
    idx_second_order = [(trajopt.indices.second_order_duals...)...]
    return idx_nonnegative, idx_second_order
end