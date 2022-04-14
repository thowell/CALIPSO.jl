struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    # num_variables::Int     
    # num_parameters::Int            
    # num_equality::Int 
    # num_equality_dynamics::Int 
    # num_equality_constraints::Int
    # num_cone::Int
    # num_cone_nonnegative::Int
    # num_cone_second_order::Int
    # num_jacobian_equality::Int 
    # num_jacobian_cone::Int
    # num_hessian_lagrangian::Int  

    indices::TrajectoryOptimizationIndices
    sparsity::TrajectoryOptimizationSparsity 
    dimensions::TrajectoryOptimizationDimensions

    # TODO sparsity struct
    # sparsity_dynamics_jacobian
    # sparsity_equality_jacobian
    # sparsity_nonnegative_jacobian
    # sparsity_second_order_jacobian

    # hessian_sparsity_key
    # sparsity_objective_hessian
    # sparsity_dynamics_hessian
    # sparsity_equality_hessian
    # sparsity_nonnegative_hessian
    # sparsity_second_order_hessian

    hessian_lagrangian::Bool 
    parameters::Vector{T}
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    evaluate_hessian=true) 

    # # number of variables
    # total_variables = sum(data.state_dimensions) + sum(data.action_dimensions)

    # # number of parameters = 
    # num_parameters = sum(data.parameter_dimensions)

    # # number of constraints
    # num_dynamics = num_constraint(data.dynamics)
    # num_equality = num_constraint(data.equality) 
    # num_nonnegative = num_constraint(data.nonnegative)
    # num_second_order = sum(num_constraint(data.second_order))
    # total_equality = num_dynamics + num_equality 
    # total_cone = num_nonnegative + num_second_order

    # # number of nonzeros in constraint Jacobian
    # num_dynamics_jacobian = num_jacobian(data.dynamics)
    # num_equality_jacobian = num_jacobian(data.equality)  
    # num_nonnegative_jacobian = num_jacobian(data.nonnegative)
    # num_second_order_jacobian = sum(num_jacobian(data.second_order))

    # total_equality_jacobians = num_dynamics_jacobian + num_equality_jacobian 
    # total_cone_jacobians = num_nonnegative_jacobian + num_second_order_jacobian

    # # constraint Jacobian sparsity
    # sparsity_dynamics_jacobian = sparsity_jacobian(data.dynamics, data.state_dimensions, data.action_dimensions, 
    #     row_shift=0)
    # sparsity_equality_jacobian = sparsity_jacobian(data.equality, data.state_dimensions, data.action_dimensions, 
    #     row_shift=num_dynamics)
    # sparsity_nonnegative_jacobian = sparsity_jacobian(data.nonnegative, data.state_dimensions, data.action_dimensions, 
    #     row_shift=0)
    # sparsity_second_order_jacobian = sparsity_jacobian(data.second_order, data.state_dimensions, data.action_dimensions, 
    #     row_shift=num_nonnegative)

    # # Hessian of Lagrangian sparsity 
    # sparsity_objective_hessian = sparsity_hessian(data.objective, data.state_dimensions, data.action_dimensions)
    # sparsity_dynamics_hessian = sparsity_hessian(data.dynamics, data.state_dimensions, data.action_dimensions)
    # sparsity_equality_hessian = sparsity_hessian(data.equality, data.state_dimensions, data.action_dimensions)
    # sparsity_nonnegative_hessian = sparsity_hessian(data.nonnegative, data.state_dimensions, data.action_dimensions)
    # sparsity_second_order_hessian = sparsity_hessian(data.second_order, data.state_dimensions, data.action_dimensions)

    # hessian_lagrangian_sparsity = [(sparsity_objective_hessian...)..., 
    #     (sparsity_dynamics_hessian...)..., 
    #     (sparsity_equality_hessian...)..., 
    #     (sparsity_nonnegative_hessian...)...,
    #     ((sparsity_second_order_hessian...)...)...,
    # ]
    
    # hessian_lagrangian_sparsity = !isempty(hessian_lagrangian_sparsity) ? hessian_lagrangian_sparsity : Tuple{Int,Int}[]
    # hessian_sparsity_key = sort(unique(hessian_lagrangian_sparsity))

    # # number of nonzeros in Hessian of Lagrangian
    # num_hessian_lagrangian = length(hessian_lagrangian_sparsity)

    spar = TrajectoryOptimizationSparsity(data)
    dims = TrajectoryOptimizationDimensions(data, num_hessian_lagrangian=length(spar.hessian_key))

    # indices 
    idx = indices(
        data.objective, 
        data.dynamics, 
        data.equality, 
        data.nonnegative,
        data.second_order,
        spar.hessian_key, 
        dims.states, 
        dims.actions, 
        dims.total_variables)

   

    TrajectoryOptimizationProblem(
        data, 
        # total_variables, 
        # num_parameters,
        # total_equality,
        # num_dynamics, 
        # num_equality,
        # total_cone,
        # num_nonnegative, 
        # num_second_order, 
        # total_equality_jacobians, 
        # total_cone_jacobians,
        # num_hessian_lagrangian, 
        idx, 
        spar, 
        dims,
        # sparsity_dynamics_jacobian,
        # sparsity_equality_jacobian,
        # sparsity_nonnegative_jacobian,
        # sparsity_second_order_jacobian,
        # hessian_sparsity_key,
        # sparsity_objective_hessian,
        # sparsity_dynamics_hessian,
        # sparsity_equality_hessian,
        # sparsity_nonnegative_hessian,
        # sparsity_second_order_hessian,
        evaluate_hessian, 
        vcat(data.parameters...),
        )
end

function TrajectoryOptimizationProblem(
    dynamics::Vector{Dynamics{T}}, 
    objective::Objective{T}, 
    equality::Constraints{T}, 
    nonnegative::Constraints{T},
    second_order::Vector{Constraints{T}}; 
    evaluate_hessian=true, 
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    data = TrajectoryOptimizationData(dynamics, objective, equality, nonnegative, second_order,
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

function trajectory!(states::Vector{Vector{T}}, actions::Vector{Vector{T}}, 
    trajectory, 
    state_indices::Vector{Vector{Int}}, action_indices::Vector{Vector{Int}}) where T
    for (t, idx) in enumerate(state_indices)
        states[t] .= @views trajectory[idx]
    end 
    for (t, idx) in enumerate(action_indices)
        actions[t] .= @views trajectory[idx]
    end
    return 
end

function equality_duals!(duals_dynamics::Vector{Vector{T}}, duals_equality::Vector{Vector{T}}, 
    duals, 
    dynamics_indices::Vector{Vector{Int}}, equality_indices::Vector{Vector{Int}}) where T
    
    for (t, idx) in enumerate(dynamics_indices)
        duals_dynamics[t] .= @views duals[idx]
    end 
    for (t, idx) in enumerate(equality_indices)
        duals_equality[t] .= @views duals[idx]
    end
    return 
end

function cone_duals!(duals_nonnegative::Vector{Vector{T}}, duals_second_order::Vector{Vector{Vector{T}}},
    duals, 
    nonnegative_indices::Vector{Vector{Int}}, second_order_indices::Vector{Vector{Vector{Int}}}) where T
    
    for (t, idx) in enumerate(nonnegative_indices)
        duals_nonnegative[t] .= @views duals[idx]
    end
    for (t, idx_soc) in enumerate(second_order_indices)
        for (i, idx) in enumerate(idx_soc)
            duals_second_order[t][i] .= @views duals[idx]
        end
    end
    return 
end