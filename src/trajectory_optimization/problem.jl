
function TrajectoryOptimizationProblem(dynamics::Vector{Dynamics{T}}, objective::Objective{T}, constraints::Constraints{T}, bounds::Bounds{T}; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint(),
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    trajopt = TrajectoryOptimizationData(objective, dynamics, constraints, bounds, 
        parameters=parameters)
    nlp = TrajectoryOptimizationProblem(trajopt, 
        general_constraint=general_constraint, 
        evaluate_hessian=evaluate_hessian) 

    return nlp 
end

function initialize_states!(solver::TrajectoryOptimizationProblem, states) 
    for (t, xt) in enumerate(states) 
        solver.trajopt.states[t] .= xt
    end
end 

function initialize_controls!(solver::TrajectoryOptimizationProblem, actions)
    for (t, ut) in enumerate(actions) 
        solver.trajopt.actions[t] .= ut
    end
end

function get_trajectory(solver::TrajectoryOptimizationProblem) 
    return solver.nlp.trajopt.states, solver.nlp.trajopt.actions[1:end-1]
end
