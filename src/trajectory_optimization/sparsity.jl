struct TrajectoryOptimizationSparsity 
    dynamics_jacobian
    equality_jacobian
    nonnegative_jacobian
    second_order_jacobian
    hessian_key
    objective_hessian
    dynamics_hessian
    equality_hessian
    nonnegative_hessian
    second_order_hessian
end

function TrajectoryOptimizationSparsity(data::TrajectoryOptimizationData)
    # dimensions
    num_states, num_actions, num_parameters = dimensions(data.dynamics)
    num_dynamics = num_constraint(data.dynamics)
    num_nonnegative = num_constraint(data.nonnegative)

    # constraint Jacobian sparsity
    sparsity_dynamics_jacobian = sparsity_jacobian(data.dynamics, num_states, num_actions, 
        row_shift=0)
    sparsity_equality_jacobian = sparsity_jacobian(data.equality, num_states, num_actions, 
        row_shift=num_dynamics)
    sparsity_nonnegative_jacobian = sparsity_jacobian(data.nonnegative, num_states, num_actions, 
        row_shift=0)
    sparsity_second_order_jacobian = sparsity_jacobian(data.second_order, num_states, num_actions, 
        row_shift=num_nonnegative)

    # Hessian of Lagrangian sparsity 
    sparsity_objective_hessian = sparsity_hessian(data.objective, num_states, num_actions)
    sparsity_dynamics_hessian = sparsity_hessian(data.dynamics, num_states, num_actions)
    sparsity_equality_hessian = sparsity_hessian(data.equality, num_states, num_actions)
    sparsity_nonnegative_hessian = sparsity_hessian(data.nonnegative, num_states, num_actions)
    sparsity_second_order_hessian = sparsity_hessian(data.second_order, num_states, num_actions)

    hessian_lagrangian_sparsity = [(sparsity_objective_hessian...)..., 
        (sparsity_dynamics_hessian...)..., 
        (sparsity_equality_hessian...)..., 
        (sparsity_nonnegative_hessian...)...,
        ((sparsity_second_order_hessian...)...)...,
    ]

    hessian_lagrangian_sparsity = !isempty(hessian_lagrangian_sparsity) ? hessian_lagrangian_sparsity : Tuple{Int,Int}[]
    hessian_sparsity_key = sort(unique(hessian_lagrangian_sparsity))

    TrajectoryOptimizationSparsity(
        sparsity_dynamics_jacobian,
        sparsity_equality_jacobian,
        sparsity_nonnegative_jacobian,
        sparsity_second_order_jacobian,
        hessian_sparsity_key,
        sparsity_objective_hessian,
        sparsity_dynamics_hessian,
        sparsity_equality_hessian,
        sparsity_nonnegative_hessian,
        sparsity_second_order_hessian,
    )
end