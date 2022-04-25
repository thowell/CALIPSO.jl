struct TrajectoryOptimizationIndices 
    objective_jacobians_variables_variables::Vector{Vector{Int}}
    objective_jacobians_variables_parameters::Vector{Vector{Int}}
    dynamics_constraints::Vector{Vector{Int}} 
    dynamics_jacobians_variables::Vector{Vector{Int}} 
    dynamics_jacobians_parameters::Vector{Vector{Int}} 
    dynamics_jacobian_variables_variables::Vector{Vector{Int}}
    dynamics_jacobian_variables_parameters::Vector{Vector{Int}}
    equality_constraints::Vector{Vector{Int}} 
    equality_jacobians_variables::Vector{Vector{Int}} 
    equality_jacobians_parameters::Vector{Vector{Int}} 
    equality_jacobians_variables_variables::Vector{Vector{Int}}
    equality_jacobians_variables_parameters::Vector{Vector{Int}}
    nonnegative_constraints::Vector{Vector{Int}} 
    nonnegative_jacobians_variables::Vector{Vector{Int}} 
    nonnegative_jacobians_parameters::Vector{Vector{Int}} 
    nonnegative_jacobians_variables_variables::Vector{Vector{Int}}
    nonnegative_jacobians_variables_parameters::Vector{Vector{Int}}
    second_order_constraints::Vector{Vector{Vector{Int}}}
    second_order_jacobians_variables::Vector{Vector{Vector{Int}}}
    second_order_jacobians_parameters::Vector{Vector{Vector{Int}}}
    second_order_jacobians_variables_variables::Vector{Vector{Vector{Int}}}
    second_order_jacobians_variables_parameters::Vector{Vector{Vector{Int}}}
    dynamics_duals::Vector{Vector{Int}}
    equality_duals::Vector{Vector{Int}} 
    nonnegative_duals::Vector{Vector{Int}}
    second_order_duals::Vector{Vector{Vector{Int}}}
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Int}}
    state_action::Vector{Vector{Int}}
    state_action_next_state::Vector{Vector{Int}}
    parameters::Vector{Vector{Int}}
end

function indices(
    objective, 
    dynamics, 
    equality,
    nonnegative, 
    second_order,
    jacobian_variables_variables_key::Vector{Tuple{Int,Int}}, 
    jacobian_variables_parameters_key::Vector{Tuple{Int,Int}}, 
    num_state::Vector{Int}, 
    num_action::Vector{Int}, 
    num_parameter::Vector{Int},
    num_trajectory::Int) 
    
    # parameters 
    parameters = [sum(num_parameter[1:(t-1)]) .+ collect(1:num_parameter[t]) for t = 1:length(num_parameter)]

    # dynamics
    dynamics_constraints = constraint_indices(dynamics, 
        shift=0)
    dynamics_jacobians_variables = jacobian_variables_indices(dynamics, 
        shift=0)
    dynamics_jacobians_parameters = jacobian_parameters_indices(dynamics, 
        shift=0)

    # equality constraints
    equality_constraints = constraint_indices(equality, 
        shift=num_constraint(dynamics))
    equality_jacobians_variables = jacobian_variables_indices(equality, 
        shift=num_jacobian_variables(dynamics))
    equality_jacobians_parameters = jacobian_parameters_indices(equality, 
        shift=num_jacobian_parameters(dynamics))
    
    # nonnegative constraints
    nonnegative_constraints = constraint_indices(nonnegative, 
        shift=0)
    nonnegative_jacobians_variables = jacobian_variables_indices(nonnegative, 
        shift=0) 
    nonnegative_jacobians_parameters = jacobian_parameters_indices(nonnegative, 
        shift=0) 

    # second-order constraints
    second_order_constraints = constraint_indices(second_order, 
        shift=num_constraint(nonnegative))
    second_order_jacobians_variables = jacobian_variables_indices(second_order, 
        shift=num_jacobian_variables(nonnegative))
    second_order_jacobians_parameters = jacobian_parameters_indices(second_order, 
        shift=num_jacobian_parameters(nonnegative))

    # equality duals 
    dynamics_duals = constraint_indices(dynamics)
    equality_duals = constraint_indices(equality,
        shift=num_constraint(dynamics))

    # nonnegative duals
    nonnegative_duals = constraint_indices(nonnegative)

    # second-order duals
    second_order_duals = constraint_indices(second_order, 
        shift=num_constraint(nonnegative))

    # objective Jacobians
    objective_jacobians_variables_variables = jacobian_variables_variables_indices(objective, jacobian_variables_variables_key, num_state, num_action)
    objective_jacobians_variables_parameters = jacobian_variables_parameters_indices(objective, jacobian_variables_parameters_key, num_state, num_action, num_parameter)

    dynamics_jacobians_variables_variables = jacobian_variables_variables_indices(dynamics, jacobian_variables_variables_key, num_state, num_action)
    dynamics_jacobians_variables_parameters = jacobian_variables_parameters_indices(dynamics, jacobian_variables_parameters_key, num_state, num_action, num_parameter)

    equality_jacobian_variables_variables = jacobian_variables_variables_indices(equality, jacobian_variables_variables_key, num_state, num_action)
    equality_jacobian_variables_parameters = jacobian_variables_parameters_indices(equality, jacobian_variables_parameters_key, num_state, num_action, num_parameter)

    nonnegative_jacobian_variables_variables = jacobian_variables_variables_indices(nonnegative, jacobian_variables_variables_key, num_state, num_action)
    nonnegative_jacobian_variables_parameters = jacobian_variables_parameters_indices(nonnegative, jacobian_variables_parameters_key, num_state, num_action, num_parameter)

    second_order_jacobian_variables_variables = jacobian_variables_variables_indices(second_order, jacobian_variables_variables_key, num_state, num_action)
    second_order_jacobian_variables_parameters = jacobian_variables_parameters_indices(second_order, jacobian_variables_parameters_key, num_state, num_action, num_parameter)

    # indices
    x_idx = state_indices(dynamics)
    u_idx = action_indices(dynamics)
    xu_idx = state_action_indices(dynamics)
    xuy_idx = state_action_next_state_indices(dynamics)

    return TrajectoryOptimizationIndices(
        objective_jacobians_variables_variables, 
        objective_jacobians_variables_parameters, 
        dynamics_constraints, 
        dynamics_jacobians_variables, 
        dynamics_jacobians_parameters, 
        dynamics_jacobians_variables_variables, 
        dynamics_jacobians_variables_parameters, 
        equality_constraints, 
        equality_jacobians_variables, 
        equality_jacobians_parameters, 
        equality_jacobian_variables_variables,
        equality_jacobian_variables_parameters,
        nonnegative_constraints, 
        nonnegative_jacobian_variables_variables, 
        nonnegative_jacobian_variables_parameters, 
        nonnegative_jacobian_variables_variables,
        nonnegative_jacobian_variables_parameters,
        second_order_constraints, 
        second_order_jacobian_variables_variables, 
        second_order_jacobian_variables_parameters, 
        second_order_jacobian_variables_variables,
        second_order_jacobian_variables_parameters,
        dynamics_duals,
        equality_duals, 
        nonnegative_duals,
        second_order_duals,
        x_idx, 
        u_idx, 
        xu_idx, 
        xuy_idx,
        parameters,
    ) 
end