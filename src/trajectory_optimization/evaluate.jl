function objective!(trajopt::TrajectoryOptimizationProblem{T}, variables) where T
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

function objective_gradient!(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
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

function objective_hessian!(hessian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(hessian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    hessian!(
        hessian, 
        trajopt.sparsity_objective_hessian, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters)
   return
end

function equality!(violations, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
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
    if trajopt.num_equality_constraints > 0
        constraints!(
            violations, 
            trajopt.indices.equality_constraints, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return 
end

function cone!(violations, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(violations, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    if trajopt.num_cone_nonnegative > 0
        constraints!(
            violations, 
            trajopt.indices.nonnegative_constraints, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if trajopt.num_cone_second_order > 0
        constraints!(
            violations, 
            trajopt.indices.second_order_constraints, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return 
end

function equality_jacobian!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian!(
        jacobian, 
        trajopt.sparsity_dynamics_jacobian, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    if trajopt.num_equality_constraints > 0
        jacobian!(
            jacobian, 
            trajopt.sparsity_equality_jacobian, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function cone_jacobian!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    if trajopt.num_cone_nonnegative > 0
        jacobian!(
            jacobian, 
            trajopt.sparsity_nonnegative_jacobian, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if trajopt.num_cone_second_order > 0
        jacobian!(
            jacobian, 
            trajopt.sparsity_second_order_jacobian, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function equality_hessian!(hessian, trajopt::TrajectoryOptimizationProblem{T}, variables, equality_duals) where T
    fill!(hessian, 0.0)

    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    equality_duals!(
        trajopt.data.duals_dynamics, 
        trajopt.data.duals_equality, 
        equality_duals, 
        trajopt.indices.dynamics_duals, 
        trajopt.indices.equality_duals)
    hessian_lagrangian!(
        hessian, 
        trajopt.sparsity_dynamics_hessian, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_dynamics)
    if trajopt.num_equality > 0 
        hessian_lagrangian!(
            hessian, 
            trajopt.sparsity_equality_hessian, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_equality)
    end
    return 
end

function cone_hessian!(hessian, trajopt::TrajectoryOptimizationProblem{T}, variables, cone_duals) where T
    fill!(hessian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    cone_duals!(
        trajopt.data.duals_nonnegative, trajopt.data.duals_second_order,
        cone_duals, 
        trajopt.indices.nonnegative_duals, trajopt.indices.second_order_duals)
    if trajopt.num_cone_nonnegative > 0 
        hessian_lagrangian!(
            hessian, 
            trajopt.sparsity_nonnegative_hessian, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_nonnegative)
    end
    if trajopt.num_cone_second_order > 0 
        hessian_lagrangian!(
            hessian, 
            trajopt.sparsity_second_order_hessian, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_second_order)
    end
    return 
end