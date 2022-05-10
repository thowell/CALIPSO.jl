struct TrajectoryOptimizationProblem{T}
    data::TrajectoryOptimizationData{T}
    indices::TrajectoryOptimizationIndices
    sparsity::TrajectoryOptimizationSparsity 
    dimensions::TrajectoryOptimizationDimensions
    hessian_lagrangian::Bool 
end

function TrajectoryOptimizationProblem(data::TrajectoryOptimizationData; 
    constraint_tensor=true) 
    spar = TrajectoryOptimizationSparsity(data)
    dims = TrajectoryOptimizationDimensions(data, 
        num_jacobian_variables_variables=length(spar.jacobian_variables_variables_key),
        num_jacobian_variables_parameters=length(spar.jacobian_variables_parameters_key))

    # indices 
    idx = indices(
        data.objective, 
        data.dynamics, 
        data.equality,
        data.equality_general, 
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
        constraint_tensor, 
    )
end

function TrajectoryOptimizationProblem(
    dynamics, 
    objective, 
    equality, 
    equality_general,
    nonnegative,
    second_order; 
    constraint_tensor=true, 
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)])

    data = TrajectoryOptimizationData(dynamics, objective, equality, equality_general, nonnegative, second_order,
        parameters=parameters)

    trajopt = TrajectoryOptimizationProblem(data, 
        constraint_tensor=constraint_tensor) 

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
