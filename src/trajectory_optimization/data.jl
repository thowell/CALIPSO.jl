struct TrajectoryOptimizationData{T}
    state_action_trajectory::Vector{T} 
    parameter_trajectory::Vector{T}
    states::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    actions::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    parameters::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    
    objective::Vector{Cost{T}}
    dynamics::Vector{Dynamics{T}}
    equality::Vector{Constraint{T}}
    nonnegative::Vector{Constraint{T}}
    second_order::Vector{Vector{Constraint{T}}}

    duals_equality_trajectory::Vector{T} 
    duals_cone_trajectory::Vector{T}

    duals_dynamics::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}} 
    duals_equality::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    duals_nonnegative::Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}
    duals_second_order::Vector{Vector{SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}}}
end

function TrajectoryOptimizationData(
    dynamics, 
    objective, 
    equality, 
    nonnegative,
    second_order; 
    parameters=[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]])

    state_dimensions, action_dimensions, parameter_dimensions = dimensions(dynamics,
        parameters=[length(p) for p in parameters])

    dynamics_dimensions = num_constraint(dynamics)
    equality_dimensions = num_constraint(equality) 
    nonnegative_dimensions = num_constraint(nonnegative)
    second_order_dimensions = sum(num_constraint(second_order))

    state_action_trajectory = zeros(sum(state_dimensions) + sum(action_dimensions)) 
    parameter_trajectory = vcat(parameters...) 
    duals_equality_trajectory = zeros(dynamics_dimensions + equality_dimensions) 
    duals_cone_trajectory = zeros(nonnegative_dimensions + second_order_dimensions)

    x_idx = state_indices(dynamics)
    u_idx = action_indices(dynamics)
    p_idx = [sum(parameter_dimensions[1:(t-1)]) .+ collect(1:parameter_dimensions[t]) for t = 1:length(parameter_dimensions)]
    d_idx = constraint_indices(dynamics)
    e_idx = constraint_indices(equality,
        shift=num_constraint(dynamics))
    nn_idx = constraint_indices(nonnegative)
    so_idx = constraint_indices(second_order, 
        shift=num_constraint(nonnegative))

    states = [@views state_action_trajectory[idx] for idx in x_idx]
    actions = [[@views state_action_trajectory[idx] for idx in u_idx]..., @views state_action_trajectory[collect(1:0)]]
    parameters = [@views parameter_trajectory[idx] for idx in p_idx]

    duals_dynamics = [@views duals_equality_trajectory[idx] for idx in d_idx]
    duals_equality = [@views duals_equality_trajectory[idx] for idx in e_idx]
    duals_nonnegative = [@views duals_cone_trajectory[idx] for idx in nn_idx]
    duals_second_order = [[@views duals_cone_trajectory[idx] for idx in so] for so in so_idx]
   
    TrajectoryOptimizationData(
        state_action_trajectory, 
        parameter_trajectory,
        states, 
        actions, 
        parameters,
        objective, 
        dynamics, 
        equality,
        nonnegative,
        second_order, 
        duals_equality_trajectory, 
        duals_cone_trajectory,
        duals_dynamics, 
        duals_equality, 
        duals_nonnegative,
        duals_second_order,
    )
end

function TrajectoryOptimizationData(dynamics, objective)
    TrajectoryOptimizationData(
        dynamics, 
        objective, 
        [Constraint() for t = 1:length(dynamics)], 
        [Constraint() for t = 1:length(dynamics)],
        [[Constraint()] for t = 1:length(dynamics)])
end



