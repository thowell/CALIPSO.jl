struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    indices::TrajectoryOptimizationIndices
    sparsity::TrajectoryOptimizationSparsity 
    dimensions::TrajectoryOptimizationDimensions
    hessian_lagrangian::Bool 
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    evaluate_hessian=true) 
    spar = TrajectoryOptimizationSparsity(data)
    dims = TrajectoryOptimizationDimensions(data, 
        num_jacobian_variables_variables=length(spar.jacobian_variables_variables_key),
        num_jacobian_variables_parameters=length(spar.jacobian_variables_parameters_key))

    # indices 
    idx = indices(
        data.objective, 
        data.dynamics, 
        data.equality, 
        data.nonnegative,
        data.second_order,
        spar.jacobian_variables_variables_key, 
        spar.jacobian_variables_parameters_key,
        dims.states, 
        dims.actions, 
        dims.parameters,
        dims.total_variables)

    TrajectoryOptimizationProblem(
        data, 
        idx, 
        spar, 
        dims,
        evaluate_hessian, 
    )
end

function TrajectoryOptimizationProblem(
    dynamics, 
    objective, 
    equality, 
    nonnegative,
    second_order; 
    evaluate_hessian=true, 
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)])

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
