struct TrajectoryOptimizationIndices 
    objective_jacobians_variables_variables::Vector{Vector{Int}}
    objective_jacobians_variables_parameters::Vector{Vector{Int}}
    dynamics_constraints::Vector{Vector{Int}} 
    dynamics_jacobians::Vector{Vector{Int}} 
    dynamics_hessians::Vector{Vector{Int}}
    equality_constraints::Vector{Vector{Int}} 
    equality_jacobians::Vector{Vector{Int}} 
    equality_hessians::Vector{Vector{Int}}
    nonnegative_constraints::Vector{Vector{Int}} 
    nonnegative_jacobians::Vector{Vector{Int}} 
    nonnegative_hessians::Vector{Vector{Int}}
    second_order_constraints::Vector{Vector{Vector{Int}}}
    second_order_jacobians::Vector{Vector{Vector{Int}}}
    second_order_hessians::Vector{Vector{Vector{Int}}}
    dynamics_duals::Vector{Vector{Int}}
    equality_duals::Vector{Vector{Int}} 
    nonnegative_duals::Vector{Vector{Int}}
    second_order_duals::Vector{Vector{Vector{Int}}}
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Int}}
    state_action::Vector{Vector{Int}}
    state_action_next_state::Vector{Vector{Int}}
end

function indices(
    objective::Objective{T}, 
    dynamics::Vector{Dynamics{T}}, 
    equality::Constraints{T},
    nonnegative::Constraints{T}, 
    second_order::Vector{Constraints{T}},
    key::Vector{Tuple{Int,Int}}, 
    num_state::Vector{Int}, 
    num_action::Vector{Int}, 
    num_trajectory::Int) where T 
    
    # dynamics
    dynamics_constraints = constraint_indices(dynamics, 
        shift=0)
    dynamics_jacobians = jacobian_indices(dynamics, 
        shift=0)

    # equality constraints
    equality_constraints = constraint_indices(equality, 
        shift=num_constraint(dynamics))
    equality_jacobians = jacobian_indices(equality, 
        shift=num_jacobian(dynamics))
    
    # non-negative constraints
    nonnegative_constraints = constraint_indices(nonnegative, 
        shift=0)
    nonnegative_jacobians = jacobian_indices(nonnegative, 
        shift=0) 

    # second-order constraints
    second_order_constraints = constraint_indices(second_order, 
        shift=num_constraint(nonnegative))
    second_order_jacobians = jacobian_indices(second_order, 
        shift=num_jacobian(nonnegative))

    # equality duals 
    dynamics_duals = constraint_indices(dynamics)
    equality_duals = constraint_indices(equality,
        shift=num_constraint(dynamics))

    # non-negative duals
    nonnegative_duals = constraint_indices(nonnegative)

    # second-order duals
    second_order_duals = constraint_indices(second_order, 
        shift=num_constraint(nonnegative))

    # objective Jacobians
    objective_jacobians_variables_variables = jacobian_variables_variables_indices(objective, key, num_state, num_action)
    objective_jacobians_variables_parameters = jacobian_variables_parameters_indices(objective, key, num_state, num_action)

    dynamics_hessians = hessian_indices(dynamics, key, num_state, num_action)
    equality_hessians = hessian_indices(equality, key, num_state, num_action)
    nonnegative_hessians = hessian_indices(nonnegative, key, num_state, num_action)
    second_order_hessians = hessian_indices(second_order, key, num_state, num_action)

    # indices
    x_idx = state_indices(dynamics)
    u_idx = action_indices(dynamics)
    xu_idx = state_action_indices(dynamics)
    xuy_idx = state_action_next_state_indices(dynamics)

    return TrajectoryOptimizationIndices(
        objective_jacobians_variables_variables, 
        objective_jacobians_variables_parameters, 
        dynamics_constraints, 
        dynamics_jacobians, 
        dynamics_hessians, 
        equality_constraints, 
        equality_jacobians, 
        equality_hessians,
        nonnegative_constraints, 
        nonnegative_jacobians, 
        nonnegative_hessians,
        second_order_constraints, 
        second_order_jacobians, 
        second_order_hessians,
        dynamics_duals,
        equality_duals, 
        nonnegative_duals,
        second_order_duals,
        x_idx, 
        u_idx, 
        xu_idx, 
        xuy_idx) 
end