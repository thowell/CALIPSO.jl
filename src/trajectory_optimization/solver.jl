struct Solver{T} <: MOI.AbstractNLPEvaluator
    nlp::NLPData{T}
    data::SolverData
end

function Solver(dynamics::Vector{Dynamics{T}}, objective::Objective{T}, constraints::Constraints{T}, bounds::Bounds{T}; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint(),
    options=Options(),
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    trajopt = TrajectoryOptimizationData(objective, dynamics, constraints, bounds, 
        parameters=parameters)
    nlp = NLPData(trajopt, 
        general_constraint=general_constraint, 
        evaluate_hessian=evaluate_hessian) 
    data = SolverData(nlp, 
        options=options)

    Solver(nlp, data) 
end

function initialize_states!(solver::Solver, states) 
    for (t, xt) in enumerate(states) 
        solver.nlp.trajopt.states[t] .= xt
    end
end 

function initialize_controls!(solver::Solver, actions)
    for (t, ut) in enumerate(actions) 
        solver.nlp.trajopt.actions[t] .= ut
    end
end

function get_trajectory(solver::Solver) 
    return solver.nlp.trajopt.states, solver.nlp.trajopt.actions[1:end-1]
end

function solve!(solver::Solver) 
    MOI.optimize!(solver.data.optimizer) 
end