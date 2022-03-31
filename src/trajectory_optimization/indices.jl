struct TrajectoryOptimizationIndices 
    objective_hessians::Vector{Vector{Int}}
    dynamics_constraints::Vector{Vector{Int}} 
    dynamics_jacobians::Vector{Vector{Int}} 
    dynamics_hessians::Vector{Vector{Int}}
    stage_constraints::Vector{Vector{Int}} 
    stage_jacobians::Vector{Vector{Int}} 
    stage_hessians::Vector{Vector{Int}}
    general_constraint::Vector{Int}
    general_jacobian::Vector{Int}
    general_hessian::Vector{Int}
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Int}}
    state_action::Vector{Vector{Int}}
    state_action_next_state::Vector{Vector{Int}}
end

function indices(objective::Objective{T}, dynamics::Vector{Dynamics{T}}, constraints::Constraints{T}, general::GeneralConstraint{T},
    key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_trajectory::Int) where T 
    # Jacobians
    dynamics_constraints = constraint_indices(dynamics, 
        shift=0)
    dynamics_jacobians = jacobian_indices(dynamics, 
        shift=0)
    stage_constraints = constraint_indices(constraints, 
        shift=num_constraint(dynamics))
    stage_jacobians = jacobian_indices(constraints, 
        shift=num_jacobian(dynamics)) 
    general_constraint = constraint_indices(general, 
        shift=(num_constraint(dynamics) + num_constraint(constraints)))
    general_jacobian = jacobian_indices(general, 
        shift=(num_jacobian(dynamics) + num_jacobian(constraints))) 

    # Hessian of Lagrangian 
    objective_hessians = hessian_indices(objective, key, num_state, num_action)
    dynamics_hessians = hessian_indices(dynamics, key, num_state, num_action)
    stage_hessians = hessian_indices(constraints, key, num_state, num_action)
    general_hessian = hessian_indices(general, key, num_trajectory)

    # indices
    x_idx = state_indices(dynamics)
    u_idx = action_indices(dynamics)
    xu_idx = state_action_indices(dynamics)
    xuy_idx = state_action_next_state_indices(dynamics)

    return TrajectoryOptimizationIndices(
        objective_hessians, 
        dynamics_constraints, 
        dynamics_jacobians, 
        dynamics_hessians, 
        stage_constraints, 
        stage_jacobians, 
        stage_hessians,
        general_constraint, 
        general_jacobian, 
        general_hessian,
        x_idx, 
        u_idx, 
        xu_idx, 
        xuy_idx) 
end