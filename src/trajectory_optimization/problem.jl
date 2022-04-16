struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    indices::TrajectoryOptimizationIndices
    sparsity::TrajectoryOptimizationSparsity 
    dimensions::TrajectoryOptimizationDimensions
    hessian_lagrangian::Bool 
    parameters::Vector{T}
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    evaluate_hessian=true) 
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
        idx, 
        spar, 
        dims,
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