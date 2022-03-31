function objective(trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    cost(
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
end

function objective_gradient(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(gradient, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    gradient!(gradient, 
        trajopt.indices.state_action, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
    return 
end

function constraint(violations, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(violations, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    constraints!(
        violations, 
        trajopt.indices.dynamics_constraints, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    !isempty(trajopt.indices.stage_constraints) && constraints!(violations, trajopt.indices.stage_constraints, trajopt.data.constraints, trajopt.data.states, trajopt.data.actions, trajopt.data.parameters)
    trajopt.general_constraint.num_constraint != 0 && constraints!(violations, trajopt.indices.general_constraint, trajopt.general_constraint, variables, trajopt.parameters)
    return 
end

function constraint_jacobian(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian!(
        jacobian, 
        trajopt.indices.dynamics_jacobians, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    !isempty(trajopt.indices.stage_jacobians) && jacobian!(jacobian, trajopt.indices.stage_jacobians, trajopt.data.constraints, trajopt.data.states, trajopt.data.actions, trajopt.data.parameters)
    trajopt.general_constraint.num_constraint != 0 && jacobian!(jacobian, trajopt.indices.general_jacobian, trajopt.general_constraint, variables, trajopt.parameters)
    return
end

function hessian_lagrangian(hessian, trajopt::TrajectoryOptimizationProblem{T}, variables, duals; 
    scaling=1.0) where T

    fill!(hessian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    duals!(
        trajopt.data.duals_dynamics, 
        trajopt.data.duals_constraints, 
        trajopt.duals_general, 
        duals, 
        trajopt.indices.dynamics_constraints, 
        trajopt.indices.stage_constraints, 
        trajopt.indices.general_constraint)
    hessian!(
        hessian, 
        trajopt.indices.objective_hessians, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters, 
        scaling)
    hessian_lagrangian!(
        hessian, 
        trajopt.indices.dynamics_hessians, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_dynamics)
    hessian_lagrangian!(
        hessian, 
        trajopt.indices.stage_hessians, 
        trajopt.data.constraints, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_constraints)
    hessian_lagrangian!(
        hessian, 
        trajopt.indices.general_hessian, 
        trajopt.general_constraint, 
        variables, 
        trajopt.parameters, 
        trajopt.duals_general)
    return 
end
