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
    if !isempty(trajopt.indices.equality_constraints) 
        constraints!(
            violations, 
            trajopt.indices.equality_constraints, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if !isempty(trajopt.indices.inequality_constraints) 
        constraints!(
            violations, 
            trajopt.indices.inequality_constraints, 
            trajopt.data.inequality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
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
        trajopt.sparsity_dynamics, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    if !isempty(trajopt.indices.equality_constraints) 
        jacobian!(
            jacobian, 
            trajopt.sparsity_equality, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if !isempty(trajopt.indices.inequality_constraints) 
        jacobian!(
            jacobian, 
            trajopt.sparsity_inequality, 
            trajopt.data.inequality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
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
        trajopt.data.duals_equality, 
        trajopt.data.duals_inequality,
        duals, 
        trajopt.indices.dynamics_constraints, 
        trajopt.indices.equality_constraints, 
        trajopt.indices.inequality_constraints)
    hessian!(
        hessian, 
        trajopt.sparsity_objective_hessian, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters, 
        scaling=scaling)
    hessian_lagrangian!(
        hessian, 
        trajopt.sparsity_dynamics_hessian, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_dynamics)
    if !isempty(trajopt.indices.equality_jacobians) 
        hessian_lagrangian!(
            hessian, 
            trajopt.sparsity_equality_hessian, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_equality)
    end
    if !isempty(trajopt.indices.inequality_jacobians) 
        hessian_lagrangian!(
            hessian, 
            trajopt.sparsity_inequality_hessian, 
            trajopt.data.inequality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_inequality)
    end
    return 
end
