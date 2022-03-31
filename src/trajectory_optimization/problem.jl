
struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    num_variables::Int                 
    num_equality::Int 
    num_inequality::Int
    num_jacobian_equality::Int 
    num_jacobian_inequality::Int
    num_hessian_lagrangian::Int  

    indices::TrajectoryOptimizationIndices
    # TODO sparsity struct
    sparsity_dynamics_jacobian
    sparsity_equality_jacobian
    sparsity_inequality_jacobian

    hessian_sparsity_key
    sparsity_objective_hessian
    sparsity_dynamics_hessian
    sparsity_equality_hessian
    sparsity_inequality_hessian

    hessian_lagrangian::Bool 
    parameters::Vector{T}
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    evaluate_hessian=true) 

    # number of variables
    total_variables = sum(data.state_dimensions) + sum(data.action_dimensions)

    # number of constraints
    num_dynamics = num_constraint(data.dynamics)
    num_equality = num_constraint(data.equality) 
    num_inequality = num_constraint(data.inequality) 
    total_equality = num_dynamics + num_equality 
    total_inequality = num_inequality

    # number of nonzeros in constraint Jacobian
    num_dynamics_jacobian = num_jacobian(data.dynamics)
    num_equality_jacobian = num_jacobian(data.equality)  
    num_inequality_jacobian = num_jacobian(data.inequality)
    total_equality_jacobians = num_dynamics_jacobian + num_equality_jacobian 
    total_inequality_jacobians = num_inequality_jacobian

    # constraint Jacobian sparsity
    sparsity_dynamics_jacobian = sparsity_jacobian(data.dynamics, data.state_dimensions, data.action_dimensions, 
        row_shift=0)
    sparsity_equality_jacobian = sparsity_jacobian(data.equality, data.state_dimensions, data.action_dimensions, 
        row_shift=num_dynamics)
    sparsity_inequality_jacobian = sparsity_jacobian(data.inequality, data.state_dimensions, data.action_dimensions, 
        row_shift=(num_dynamics + num_equality))

    # Hessian of Lagrangian sparsity 
    sparsity_objective_hessian = sparsity_hessian(data.objective, data.state_dimensions, data.action_dimensions)
    sparsity_dynamics_hessian = sparsity_hessian(data.dynamics, data.state_dimensions, data.action_dimensions)
    sparsity_equality_hessian = sparsity_hessian(data.equality, data.state_dimensions, data.action_dimensions)
    sparsity_inequality_hessian = sparsity_hessian(data.inequality, data.state_dimensions, data.action_dimensions)
    hessian_lagrangian_sparsity = [(sparsity_objective_hessian...)..., 
        (sparsity_dynamics_hessian...)..., 
        (sparsity_equality_hessian...)..., 
        (sparsity_inequality_hessian...)...]
    hessian_lagrangian_sparsity = !isempty(hessian_lagrangian_sparsity) ? hessian_lagrangian_sparsity : Tuple{Int,Int}[]
    hessian_sparsity_key = sort(unique(hessian_lagrangian_sparsity))

    # number of nonzeros in Hessian of Lagrangian
    num_hessian_lagrangian = length(hessian_lagrangian_sparsity)

    # indices 
    idx = indices(
        data.objective, 
        data.dynamics, 
        data.equality, 
        data.inequality,
        hessian_sparsity_key, 
        data.state_dimensions, 
        data.action_dimensions, 
        total_variables)

    TrajectoryOptimizationProblem(
        data, 
        total_variables, 
        total_equality,
        total_inequality, 
        total_equality_jacobians, 
        total_inequality_jacobians,
        num_hessian_lagrangian, 
        idx, 
        sparsity_dynamics_jacobian,
        sparsity_equality_jacobian,
        sparsity_inequality_jacobian,
        hessian_sparsity_key,
        sparsity_objective_hessian,
        sparsity_dynamics_hessian,
        sparsity_equality_hessian,
        sparsity_inequality_hessian,
        evaluate_hessian, 
        vcat(data.parameters...),
        )
end

function TrajectoryOptimizationProblem(dynamics::Vector{Dynamics{T}}, objective::Objective{T}, equality::Constraints{T}, inequality::Constraints{T}; 
    evaluate_hessian=true, 
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    data = TrajectoryOptimizationData(dynamics, objective, equality, inequality,
        parameters=parameters)

    trajopt = TrajectoryOptimizationProblem(data, 
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
    return 
end

function duals!(duals_dynamics::Vector{Vector{T}}, duals_equality::Vector{Vector{T}}, duals_inequality::Vector{Vector{T}}, 
    duals, 
    dynamics_indices::Vector{Vector{Int}}, equality_indices::Vector{Vector{Int}}, inequality_indices::Vector{Vector{Int}}) where T
    
    for (t, idx) in enumerate(dynamics_indices)
        duals_dynamics[t] .= @views duals[idx]
    end 
    for (t, idx) in enumerate(equality_indices)
        duals_equality[t] .= @views duals[idx]
    end
    for (t, idx) in enumerate(inequality_indices)
        duals_inequality[t] .= @views duals[idx]
    end
    return 
end