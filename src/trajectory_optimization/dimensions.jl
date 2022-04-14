struct TrajectoryOptimizationDimensions 
    states::Vector{Int} 
    actions::Vector{Int}
    parameters::Vector{Int}  
    total_variables::Int     
    total_parameters::Int   
    equality::Int 
    equality_dynamics::Int 
    equality_constraints::Int
    cone::Int
    cone_nonnegative::Int
    cone_second_order::Int
    jacobian_equality::Int 
    jacobian_cone::Int
    hessian_lagrangian::Int  
end

function TrajectoryOptimizationDimensions(data::TrajectoryOptimizationData; 
    num_hessian_lagrangian=0)
    
    # number of states 
    num_states = [length(x) for x in data.states]

    # number of actions 
    num_actions = [length(u) for u in data.actions]

    # number of parameters 
    num_parameters = [length(p) for p in data.parameters]

    # number of variables
    total_variables = sum(num_states) + sum(num_actions)

    # number of parameters 
    total_parameters = sum(num_parameters)

    # number of constraints
    num_dynamics = num_constraint(data.dynamics)
    num_equality = num_constraint(data.equality) 
    num_nonnegative = num_constraint(data.nonnegative)
    num_second_order = sum(num_constraint(data.second_order))
    total_equality = num_dynamics + num_equality 
    total_cone = num_nonnegative + num_second_order

    # number of nonzeros in constraint Jacobian
    num_dynamics_jacobian = num_jacobian(data.dynamics)
    num_equality_jacobian = num_jacobian(data.equality)  
    num_nonnegative_jacobian = num_jacobian(data.nonnegative)
    num_second_order_jacobian = sum(num_jacobian(data.second_order))

    total_equality_jacobians = num_dynamics_jacobian + num_equality_jacobian 
    total_cone_jacobians = num_nonnegative_jacobian + num_second_order_jacobian

    # number of nonzeros in Hessian of Lagrangian
    # num_hessian_lagrangian = length(hessian_lagrangian_sparsity)

    TrajectoryOptimizationDimensions(
        num_states, 
        num_actions, 
        num_parameters,
        total_variables, 
        total_parameters,
        total_equality,
        num_dynamics, 
        num_equality,
        total_cone,
        num_nonnegative, 
        num_second_order, 
        total_equality_jacobians, 
        total_cone_jacobians,
        num_hessian_lagrangian, 
    )
end