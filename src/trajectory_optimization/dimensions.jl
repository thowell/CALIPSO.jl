struct TrajectoryOptimizationDimensions 
    states::Vector{Int} 
    actions::Vector{Int}
    parameters::Vector{Int}  
    total_variables::Int     
    total_parameters::Int   
    total_equality::Int 
    equality_dynamics::Int 
    equality_constraints::Int
    total_cone::Int
    cone_nonnegative::Int
    cone_second_order::Int
    equality_jacobian_variables::Int 
    equality_jacobian_parameters::Int
    cone_jacobian_variables::Int
    cone_jacobian_parameters::Int
    jacobian_variables_variables::Int  
    jacobian_variables_parameters::Int
end

function TrajectoryOptimizationDimensions(data::TrajectoryOptimizationData; 
    num_jacobian_variables_variables=0,
    num_jacobian_variables_parameters=0)
    
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
    num_dynamics_jacobian_variables = num_jacobian_variables(data.dynamics)
    num_dynamics_jacobian_parameters = num_jacobian_parameters(data.dynamics)
    num_equality_jacobian_variables = num_jacobian_variables(data.equality)  
    num_equality_jacobian_parameters = num_jacobian_parameters(data.equality)  
    num_nonnegative_jacobian_variables = num_jacobian_variables(data.nonnegative)
    num_nonnegative_jacobian_parameters = num_jacobian_parameters(data.nonnegative)

    num_second_order_jacobian_variables = sum(num_jacobian_variables(data.second_order))
    num_second_order_jacobian_parameters = sum(num_jacobian_parameters(data.second_order))

    total_equality_jacobians_variables = num_dynamics_jacobian_variables + num_equality_jacobian_variables 
    total_equality_jacobians_parameters = num_dynamics_jacobian_parameters + num_equality_jacobian_parameters 

    total_cone_jacobians_variables = num_nonnegative_jacobian_variables + num_second_order_jacobian_variables
    total_cone_jacobians_parameters = num_nonnegative_jacobian_parameters + num_second_order_jacobian_parameters

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
        total_equality_jacobians_variables, 
        total_equality_jacobians_parameters, 
        total_cone_jacobians_variables,
        total_cone_jacobians_parameters,
        num_jacobian_variables_variables, 
        num_jacobian_variables_parameters, 
    )
end