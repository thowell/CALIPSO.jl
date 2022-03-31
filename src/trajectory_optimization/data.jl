struct TrajectoryOptimizationData{T}
    states::Vector{Vector{T}}
    actions::Vector{Vector{T}}
    parameters::Vector{Vector{T}}
    objective::Objective{T}
    dynamics::Vector{Dynamics{T}}
    equality::Constraints{T}
    inequality::Constraints{T}
    duals_dynamics::Vector{Vector{T}} 
    duals_equality::Vector{Vector{T}}
    duals_inequality::Vector{Vector{T}}
    state_dimensions::Vector{Int}
    action_dimensions::Vector{Int}
    parameter_dimensions::Vector{Int}
end

function TrajectoryOptimizationData(dynamics::Vector{Dynamics{T}}, objective::Objective{T}, equality::Constraints{T}, inequality::Constraints{T}; 
    parameters=[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]) where T

    state_dimensions, action_dimensions, parameter_dimensions = dimensions(dynamics)

    states = [zeros(nx) for nx in state_dimensions]
    actions = [zeros(nu) for nu in action_dimensions]

    duals_dynamics = [zeros(nx) for nx in state_dimensions[2:end]]
    duals_equality = [zeros(eq.num_constraint) for eq in equality]
    duals_inequality = [zeros(ineq.num_constraint) for ineq in inequality]
   
    TrajectoryOptimizationData(
        states, 
        actions, 
        parameters,
        objective, 
        dynamics, 
        equality,
        inequality, 
        duals_dynamics, 
        duals_equality, 
        duals_inequality,
        state_dimensions, 
        action_dimensions, 
        parameter_dimensions)
end

TrajectoryOptimizationData(dynamics::Vector{Dynamics}, objective::Objective) = TrajectoryOptimizationData(dynamics, objective, [Constraint() for t = 1:length(dynamics)], [Constraint() for t = 1:length(dynamics)])



