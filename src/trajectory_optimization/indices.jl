struct TrajectoryOptimizationIndices 
    objective_hessians::Vector{Vector{Int}}
    dynamics_constraints::Vector{Vector{Int}} 
    dynamics_jacobians::Vector{Vector{Int}} 
    dynamics_hessians::Vector{Vector{Int}}
    equality_constraints::Vector{Vector{Int}} 
    equality_jacobians::Vector{Vector{Int}} 
    equality_hessians::Vector{Vector{Int}}
    inequality_constraints::Vector{Vector{Int}} 
    inequality_jacobians::Vector{Vector{Int}} 
    inequality_hessians::Vector{Vector{Int}}
    dynamics_duals::Vector{Vector{Int}}
    equality_duals::Vector{Vector{Int}} 
    inequality_duals::Vector{Vector{Int}}
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Int}}
    state_action::Vector{Vector{Int}}
    state_action_next_state::Vector{Vector{Int}}
end

function indices(objective::Objective{T}, dynamics::Vector{Dynamics{T}}, equality::Constraints{T}, inequality::Constraints{T},
    key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_trajectory::Int) where T 
    
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
    
    # inequality constraints
    inequality_constraints = constraint_indices(inequality, 
        shift=num_constraint(dynamics) + num_constraint(equality))
    inequality_jacobians = jacobian_indices(inequality, 
        shift=num_jacobian(dynamics) + num_jacobian(equality)) 

    # equality duals 
    dynamics_duals = constraint_indices(dynamics)
    equality_duals = constraint_indices(equality,
        shift=num_constraint(dynamics))

    # inequality duals
    inequality_duals = constraint_indices(inequality)

    # Hessian of Lagrangian 
    objective_hessians = hessian_indices(objective, key, num_state, num_action)
    dynamics_hessians = hessian_indices(dynamics, key, num_state, num_action)
    equality_hessians = hessian_indices(equality, key, num_state, num_action)
    inequality_hessians = hessian_indices(inequality, key, num_state, num_action)

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
        equality_constraints, 
        equality_jacobians, 
        equality_hessians,
        inequality_constraints, 
        inequality_jacobians, 
        inequality_hessians,
        dynamics_duals,
        equality_duals, 
        inequality_duals,
        x_idx, 
        u_idx, 
        xu_idx, 
        xuy_idx) 
end