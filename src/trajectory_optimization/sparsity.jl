struct TrajectoryOptimizationSparsity 
    dynamics_jacobian_variables
    dynamics_jacobian_parameters
    equality_jacobian_variables
    equality_jacobian_parameters
    nonnegative_jacobian_variables
    nonnegative_jacobian_parameters
    second_order_jacobian_variables
    second_order_jacobian_parameters
    jacobian_variables_variables_key
    jacobian_variables_parameters_key
    objective_jacobian_variables_variables
    objective_jacobian_variables_parameters
    dynamics_jacobian_variables_variables
    dynamics_jacobian_variables_parameters
    equality_jacobian_variables_variables
    equality_jacobian_variables_parameters
    nonnegative_jacobian_variables_variables
    nonnegative_jacobian_variables_parameters
    second_order_jacobian_variables_variables
    second_order_jacobian_variables_parameters
end

function TrajectoryOptimizationSparsity(data::TrajectoryOptimizationData)
    # dimensions
    num_states, num_actions, num_parameters = dimensions(data.dynamics,
        parameters=[length(p) for p in data.parameters])
    num_dynamics = num_constraint(data.dynamics)
    num_nonnegative = num_constraint(data.nonnegative)

    # constraint Jacobian sparsity
    sparsity_dynamics_jacobian_variables = sparsity_jacobian_variables(data.dynamics, num_states, num_actions, 
        row_shift=0)
    sparsity_equality_jacobian_variables = sparsity_jacobian_variables(data.equality, num_states, num_actions, 
        row_shift=num_dynamics)
    sparsity_nonnegative_jacobian_variables = sparsity_jacobian_variables(data.nonnegative, num_states, num_actions, 
        row_shift=0)
    sparsity_second_order_jacobian_variables = sparsity_jacobian_variables(data.second_order, num_states, num_actions, 
        row_shift=num_nonnegative)

    sparsity_dynamics_jacobian_parameters = sparsity_jacobian_parameters(data.dynamics, num_states, num_actions, num_parameters, 
        row_shift=0)
    sparsity_equality_jacobian_parameters = sparsity_jacobian_parameters(data.equality, num_states, num_actions, num_parameters, 
        row_shift=num_dynamics)
    sparsity_nonnegative_jacobian_parameters = sparsity_jacobian_parameters(data.nonnegative, num_states, num_actions, num_parameters, 
        row_shift=0)
    sparsity_second_order_jacobian_parameters = sparsity_jacobian_parameters(data.second_order, num_states, num_actions, num_parameters, 
        row_shift=num_nonnegative)

    # Hessian of Lagrangian sparsity 
    sparsity_objective_jacobian_variables_variables = sparsity_jacobian_variables_variables(data.objective, num_states, num_actions)
    sparsity_dynamics_jacobian_variables_variables = sparsity_jacobian_variables_variables(data.dynamics, num_states, num_actions)
    sparsity_equality_jacobian_variables_variables = sparsity_jacobian_variables_variables(data.equality, num_states, num_actions)
    sparsity_nonnegative_jacobian_variables_variables = sparsity_jacobian_variables_variables(data.nonnegative, num_states, num_actions)
    sparsity_second_order_jacobian_variables_variables = sparsity_jacobian_variables_variables(data.second_order, num_states, num_actions)

    sparsity_objective_jacobian_variables_parameters = sparsity_jacobian_variables_parameters(data.objective, num_states, num_actions, num_parameters)
    sparsity_dynamics_jacobian_variables_parameters = sparsity_jacobian_variables_parameters(data.dynamics, num_states, num_actions, num_parameters)
    sparsity_equality_jacobian_variables_parameters = sparsity_jacobian_variables_parameters(data.equality, num_states, num_actions, num_parameters)
    sparsity_nonnegative_jacobian_variables_parameters = sparsity_jacobian_variables_parameters(data.nonnegative, num_states, num_actions, num_parameters)
    sparsity_second_order_jacobian_variables_parameters = sparsity_jacobian_variables_parameters(data.second_order, num_states, num_actions, num_parameters)

    hessian_lagrangian_variables_variables_sparsity = [
        (sparsity_objective_jacobian_variables_variables...)..., 
        (sparsity_dynamics_jacobian_variables_variables...)..., 
        (sparsity_equality_jacobian_variables_variables...)..., 
        (sparsity_nonnegative_jacobian_variables_variables...)...,
        ((sparsity_second_order_jacobian_variables_variables...)...)...,
    ]
    hessian_lagrangian_variables_variables_sparsity = !isempty(hessian_lagrangian_variables_variables_sparsity) ? hessian_lagrangian_variables_variables_sparsity : Tuple{Int,Int}[]
    hessian_lagrangian_variables_variables_sparsity_key = sort(unique(hessian_lagrangian_variables_variables_sparsity))

    hessian_lagrangian_variables_parameters_sparsity = [
        (sparsity_objective_jacobian_variables_parameters...)..., 
        (sparsity_dynamics_jacobian_variables_parameters...)..., 
        (sparsity_equality_jacobian_variables_parameters...)..., 
        (sparsity_nonnegative_jacobian_variables_parameters...)...,
        ((sparsity_second_order_jacobian_variables_parameters...)...)...,
    ]
    hessian_lagrangian_variables_parameters_sparsity = !isempty(hessian_lagrangian_variables_parameters_sparsity) ? hessian_lagrangian_variables_parameters_sparsity : Tuple{Int,Int}[]
    hessian_lagrangian_variables_parameters_sparsity_key = sort(unique(hessian_lagrangian_variables_parameters_sparsity))

    TrajectoryOptimizationSparsity(
        sparsity_dynamics_jacobian_variables,
        sparsity_dynamics_jacobian_parameters,
        sparsity_equality_jacobian_variables,
        sparsity_equality_jacobian_parameters,
        sparsity_nonnegative_jacobian_variables,
        sparsity_nonnegative_jacobian_parameters,
        sparsity_second_order_jacobian_variables,
        sparsity_second_order_jacobian_parameters,
        hessian_lagrangian_variables_variables_sparsity_key,
        hessian_lagrangian_variables_parameters_sparsity_key,
        sparsity_objective_jacobian_variables_variables,
        sparsity_objective_jacobian_variables_parameters,
        sparsity_dynamics_jacobian_variables_variables,
        sparsity_dynamics_jacobian_variables_parameters,
        sparsity_equality_jacobian_variables_variables,
        sparsity_equality_jacobian_variables_parameters,
        sparsity_nonnegative_jacobian_variables_variables,
        sparsity_nonnegative_jacobian_variables_parameters,
        sparsity_second_order_jacobian_variables_variables,
        sparsity_second_order_jacobian_variables_parameters,
    )
end