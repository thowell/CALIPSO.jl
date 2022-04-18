struct TrajectoryOptimizationData{T}
    states::Vector{Vector{T}}
    actions::Vector{Vector{T}}
    parameters::Vector{Vector{T}}
    objective::Objective{T}
    dynamics::Vector{Dynamics{T}}
    equality::Constraints{T}
    nonnegative::Constraints{T}
    second_order::Vector{Constraints{T}}
    duals_dynamics::Vector{Vector{T}} 
    duals_equality::Vector{Vector{T}}
    duals_nonnegative::Vector{Vector{T}}
    duals_second_order::Vector{Vector{Vector{T}}}
end

function TrajectoryOptimizationData(
    dynamics::Vector{Dynamics{T}}, 
    objective::Objective{T}, 
    equality::Constraints{T}, 
    nonnegative::Constraints{T},
    second_order::Vector{Constraints{T}}; 
    parameters=[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]) where T

    state_dimensions, action_dimensions, parameter_dimensions = dimensions(dynamics,
        parameters=[length(p) for p in parameters])

    states = [zeros(nx) for nx in state_dimensions]
    actions = [zeros(nu) for nu in action_dimensions]

    duals_dynamics = [zeros(nx) for nx in state_dimensions[2:end]]
    duals_equality = [zeros(eq.num_constraint) for eq in equality]
    duals_nonnegative = [zeros(nn.num_constraint) for nn in nonnegative]
    duals_second_order = [[zeros(so.num_constraint) for so in soc] for soc in second_order]
   
    TrajectoryOptimizationData(
        states, 
        actions, 
        parameters,
        objective, 
        dynamics, 
        equality,
        nonnegative,
        second_order, 
        duals_dynamics, 
        duals_equality, 
        duals_nonnegative,
        duals_second_order,
    )
end

function TrajectoryOptimizationData(dynamics::Vector{Dynamics}, objective::Objective)
    TrajectoryOptimizationData(
        dynamics, 
        objective, 
        [Constraint() for t = 1:length(dynamics)], 
        [Constraint() for t = 1:length(dynamics)],
        [[Constraint()] for t = 1:length(dynamics)])
end



