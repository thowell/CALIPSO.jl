function MOI.eval_objective(nlp::NLPData{T}, variables::Vector{T}) where T
    trajectory!(
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        variables, 
        nlp.indices.states, 
        nlp.indices.actions)
    objective(
        nlp.trajopt.objective, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters) 
end

function MOI.eval_objective_gradient(nlp::NLPData{T}, gradient, variables) where T
    fill!(gradient, 0.0)
    trajectory!(
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        variables, 
        nlp.indices.states, 
        nlp.indices.actions)
    gradient!(gradient, 
        nlp.indices.state_action, 
        nlp.trajopt.objective, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters) 
end

function MOI.eval_constraint(nlp::NLPData{T}, violations, variables) where T
    fill!(violations, 0.0)
    trajectory!(
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        variables, 
        nlp.indices.states, 
        nlp.indices.actions)
    constraints!(
        violations, 
        nlp.indices.dynamics_constraints, 
        nlp.trajopt.dynamics, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters)
    !isempty(nlp.indices.stage_constraints) && constraints!(violations, nlp.indices.stage_constraints, nlp.trajopt.constraints, nlp.trajopt.states, nlp.trajopt.actions, nlp.trajopt.parameters)
    nlp.general_constraint.num_constraint != 0 && constraints!(violations, nlp.indices.general_constraint, nlp.general_constraint, variables, nlp.parameters)
end

function MOI.eval_constraint_jacobian(nlp::NLPData{T}, jacobian, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        variables, 
        nlp.indices.states, 
        nlp.indices.actions)
    jacobian!(
        jacobian, 
        nlp.indices.dynamics_jacobians, 
        nlp.trajopt.dynamics, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters)
    !isempty(nlp.indices.stage_jacobians) && jacobian!(jacobian, nlp.indices.stage_jacobians, nlp.trajopt.constraints, nlp.trajopt.states, nlp.trajopt.actions, nlp.trajopt.parameters)
    nlp.general_constraint.num_constraint != 0 && jacobian!(jacobian, nlp.indices.general_jacobian, nlp.general_constraint, variables, nlp.parameters)
    return nothing
end

function MOI.eval_hessian_lagrangian(nlp::MOI.AbstractNLPEvaluator, hessian, variables, scaling, duals)
    fill!(hessian, 0.0)
    trajectory!(
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        variables, 
        nlp.indices.states, 
        nlp.indices.actions)
    duals!(
        nlp.trajopt.duals_dynamics, 
        nlp.trajopt.duals_constraints, 
        nlp.duals_general, 
        duals, 
        nlp.indices.dynamics_constraints, 
        nlp.indices.stage_constraints, 
        nlp.indices.general_constraint)
    hessian!(
        hessian, 
        nlp.indices.objective_hessians, 
        nlp.trajopt.objective, 
        nlp.trajopt.states, 
        nlp.trajopt.actions,
        nlp.trajopt.parameters, 
        scaling)
    hessian_lagrangian!(
        hessian, 
        nlp.indices.dynamics_hessians, 
        nlp.trajopt.dynamics, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters, 
        nlp.trajopt.duals_dynamics)
    hessian_lagrangian!(
        hessian, 
        nlp.indices.stage_hessians, 
        nlp.trajopt.constraints, 
        nlp.trajopt.states, 
        nlp.trajopt.actions, 
        nlp.trajopt.parameters, 
        nlp.trajopt.duals_constraints)
    hessian_lagrangian!(
        hessian, 
        nlp.indices.general_hessian, 
        nlp.general_constraint, 
        variables, 
        nlp.parameters, 
        nlp.duals_general)
end

MOI.features_available(nlp::MOI.AbstractNLPEvaluator) = nlp.hessian_lagrangian ? [:Grad, :jacobian, :hessian] : [:Grad, :jacobian]
MOI.initialize(nlp::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(nlp::MOI.AbstractNLPEvaluator) = nlp.jacobian_sparsity
MOI.hessian_lagrangian_structure(nlp::MOI.AbstractNLPEvaluator) = nlp.hessian_lagrangian_sparsity


