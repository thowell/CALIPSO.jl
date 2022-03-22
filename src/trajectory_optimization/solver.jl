struct Solver{T} <: MOI.AbstractNLPEvaluator
    nlp::NLPData{T}
    solver_data::SolverData
end

function Solver(dynamics::Vector{Dynamics{T}}, obj::Objective{T}, constraints::Constraints{T}, bounds::Bounds{T}; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint(),
    options=Options(),
    parameters=[[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]..., zeros(0)]) where T

    trajopt = TrajectoryOptimizationData(obj, dynamics, constraints, bounds, parameters=parameters)
    nlp = NLPData(trajopt, 
        general_constraint=general_constraint, 
        evaluate_hessian=evaluate_hessian) 
    solver_data = SolverData(nlp, 
        options=options)

    Solver(nlp, solver_data) 
end

function initialize_states!(p::Solver, states) 
    for (t, xt) in enumerate(states) 
        n = length(xt)
        for i = 1:n
            MOI.set(p.solver_data.solver, MOI.VariablePrimalStart(), p.solver_data.variables[p.nlp.indices.states[t][i]], xt[i])
        end
    end
end 

function initialize_controls!(p::Solver, actions)
    for (t, ut) in enumerate(actions) 
        m = length(ut) 
        for j = 1:m
            MOI.set(p.solver_data.solver, MOI.VariablePrimalStart(), p.solver_data.variables[p.nlp.indices.actions[t][j]], ut[j])
        end
    end
end

function get_trajectory(p::Solver) 
    return p.nlp.trajopt.states, p.nlp.trajopt.actions[1:end-1]
end

function solve!(p::Solver) 
    MOI.optimize!(p.solver_data.solver) 
end