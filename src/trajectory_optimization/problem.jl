
struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    num_variables::Int                 
    num_constraint::Int 
    num_jacobian::Int 
    num_hessian_lagrangian::Int               
    indices::TrajectoryOptimizationIndices
    jacobian_sparsity
    hessian_lagrangian_sparsity
    hessian_lagrangian::Bool 
    general_constraint::GeneralConstraint{T}
    parameters::Vector{T}
    duals_general::Vector{T}
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint()) 

    # number of variables
    total_variables = sum(data.state_dimensions) + sum(data.action_dimensions)

    # number of constraints
    num_dynamics = num_constraint(data.dynamics)
    num_stage = num_constraint(data.constraints)  
    num_general = num_constraint(general_constraint)
    total_constraints = num_dynamics + num_stage + num_general

    # number of nonzeros in constraint Jacobian
    num_dynamics_jacobian = num_jacobian(data.dynamics)
    num_constraint_jacobian = num_jacobian(data.constraints)  
    num_general_jacobian = num_jacobian(general_constraint)
    total_jacobians = num_dynamics_jacobian + num_constraint_jacobian + num_general_jacobian

    # constraint Jacobian sparsity
    sparsity_dynamics = sparsity_jacobian(data.dynamics, data.state_dimensions, data.action_dimensions, 
        row_shift=0)
    sparsity_constraint = sparsity_jacobian(data.constraints, data.state_dimensions, data.action_dimensions, 
        row_shift=num_dynamics)
    sparsity_general = sparsity_jacobian(general_constraint, total_variables, row_shift=(num_dynamics + num_stage))
    jacobian_sparsity = collect([sparsity_dynamics..., sparsity_constraint..., sparsity_general...]) 

    # Hessian of Lagrangian sparsity 
    sparsity_objective_hessians = sparsity_hessian(data.objective, data.state_dimensions, data.action_dimensions)
    sparsity_dynamics_hessians = sparsity_hessian(data.dynamics, data.state_dimensions, data.action_dimensions)
    sparsity_constraint_hessian = sparsity_hessian(data.constraints, data.state_dimensions, data.action_dimensions)
    sparsity_general_hessian = sparsity_hessian(general_constraint, total_variables)
    hessian_lagrangian_sparsity = [sparsity_objective_hessians..., sparsity_dynamics_hessians..., sparsity_constraint_hessian..., sparsity_general_hessian...]
    hessian_lagrangian_sparsity = !isempty(hessian_lagrangian_sparsity) ? hessian_lagrangian_sparsity : Tuple{Int,Int}[]
    hessian_sparsity_key = sort(unique(hessian_lagrangian_sparsity))

    # number of nonzeros in Hessian of Lagrangian
    num_hessian_lagrangian = length(hessian_lagrangian_sparsity)

    # indices 
    idx = indices(
        data.objective, 
        data.dynamics, 
        data.constraints, 
        general_constraint, 
        hessian_sparsity_key, 
        data.state_dimensions, 
        data.action_dimensions, 
        total_variables)

    TrajectoryOptimizationProblem(data, 
        total_variables, 
        total_constraints, 
        total_jacobians, 
        num_hessian_lagrangian, 
        idx, 
        jacobian_sparsity, 
        hessian_sparsity_key,
        evaluate_hessian, 
        general_constraint,
        vcat(data.parameters...),
        zeros(general_constraint.num_constraint))
end

function TrajectoryOptimizationProblem(dynamics::Vector{Dynamics{T}}, objective::Objective{T}, constraints::Constraints{T}; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint(),
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    data = TrajectoryOptimizationData(dynamics, objective, constraints,
        parameters=parameters)

    trajopt = TrajectoryOptimizationProblem(data, 
        general_constraint=general_constraint, 
        evaluate_hessian=evaluate_hessian) 

    return trajopt 
end

function initialize_states!(trajopt::TrajectoryOptimizationProblem, states) 
    for (t, xt) in enumerate(states) 
        trajopt.data.states[t] .= xt
    end
end 

function initialize_controls!(trajopt::TrajectoryOptimizationProblem, actions)
    for (t, ut) in enumerate(actions) 
        trajopt.data.actions[t] .= ut
    end
end

function get_trajectory(trajopt::TrajectoryOptimizationProblem) 
    return trajopt.data.states, trajopt.data.actions[1:end-1]
end

function trajectory!(states::Vector{Vector{T}}, actions::Vector{Vector{T}}, 
    trajectory::Vector{T}, 
    state_indices::Vector{Vector{Int}}, action_indices::Vector{Vector{Int}}) where T
    for (t, idx) in enumerate(state_indices)
        states[t] .= @views trajectory[idx]
    end 
    for (t, idx) in enumerate(action_indices)
        actions[t] .= @views trajectory[idx]
    end
end

function duals!(duals_dynamics::Vector{Vector{T}}, duals_constraints::Vector{Vector{T}}, duals_general::Vector{T}, duals, 
    dynamics_indices::Vector{Vector{Int}}, constraint_indices::Vector{Vector{Int}}, general_indices::Vector{Int}) where T
    for (t, idx) in enumerate(dynamics_indices)
        duals_dynamics[t] .= @views duals[idx]
    end 
    for (t, idx) in enumerate(constraint_indices)
        duals_constraints[t] .= @views duals[idx]
    end
    duals_general .= duals[general_indices]
end