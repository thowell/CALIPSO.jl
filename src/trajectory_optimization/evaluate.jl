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

function objective_gradient_variables!(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(gradient, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    gradient_variables!(gradient, 
        trajopt.indices.state_action, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
    return 
end

function objective_gradient_parameters!(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(gradient, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    gradient_parameters!(gradient, 
        trajopt.indices.state_action, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
    return 
end

function objective_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian_variables_variables!(
        jacobian, 
        trajopt.sparsity.objective_jacobian_variables_variables, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters)
   return
end

function objective_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian_variables_parameters!(
        jacobian, 
        trajopt.sparsity.objective_jacobian_variables_parameters, 
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
    if trajopt.dimensions.equality_constraints > 0
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
    if trajopt.dimensions.cone_nonnegative > 0
        constraints!(
            violations, 
            trajopt.indices.nonnegative_constraints, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if trajopt.dimensions.cone_second_order > 0
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

function equality_jacobian_variables!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian_variables!(
        jacobian, 
        trajopt.sparsity.dynamics_jacobian_variables, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    if trajopt.dimensions.equality_constraints > 0
        jacobian_variables!(
            jacobian, 
            trajopt.sparsity.equality_jacobian_variables, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function equality_jacobian_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    jacobian_parameters!(
        jacobian, 
        trajopt.sparsity.dynamics_jacobian_parameters, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters)
    if trajopt.dimensions.equality_constraints > 0
        jacobian_parameters!(
            jacobian, 
            trajopt.sparsity.equality_jacobian_parameters, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function equality_dual_jacobian_variables!(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables, duals) where T
    fill!(gradient, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    equality_duals!(
        trajopt.data.duals_dynamics, 
        trajopt.data.duals_equality, 
        duals, 
        trajopt.indices.dynamics_duals, 
        trajopt.indices.equality_duals)
    constraint_dual_jacobian_variables!(
        gradient, 
        trajopt.indices.state_action_next_state, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters,
        trajopt.data.duals_dynamics)
    if trajopt.dimensions.equality_constraints > 0
        constraint_dual_jacobian_variables!(
            gradient, 
            trajopt.indices.state_action, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters,
            trajopt.data.duals_equality)
    end
    return
end

function cone_jacobian_variables!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    if trajopt.dimensions.cone_nonnegative > 0
        jacobian_variables!(
            jacobian, 
            trajopt.sparsity.nonnegative_jacobian_variables, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if trajopt.dimensions.cone_second_order > 0
        jacobian_variables!(
            jacobian, 
            trajopt.sparsity.second_order_jacobian_variables, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function cone_jacobian_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables) where T
    fill!(jacobian, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    if trajopt.dimensions.cone_nonnegative > 0
        jacobian_parameters!(
            jacobian, 
            trajopt.sparsity.nonnegative_jacobian_parameters, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    if trajopt.dimensions.cone_second_order > 0
        jacobian_parameters!(
            jacobian, 
            trajopt.sparsity.second_order_jacobian_parameters, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters)
    end
    return
end

function cone_dual_jacobian_variables!(gradient, trajopt::TrajectoryOptimizationProblem{T}, variables, duals) where T
    fill!(gradient, 0.0)
    trajectory!(
        trajopt.data.states, 
        trajopt.data.actions, 
        variables, 
        trajopt.indices.states, 
        trajopt.indices.actions)
    cone_duals!(
        trajopt.data.duals_nonnegative, 
        trajopt.data.duals_second_order,
        duals, 
        trajopt.indices.nonnegative_duals, 
        trajopt.indices.second_order_duals)
    if trajopt.dimensions.cone_nonnegative > 0
        constraint_dual_jacobian_variables!(
            gradient, 
            trajopt.indices.state_action, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters,
            trajopt.data.duals_nonnegative,
        )
    end
    if trajopt.dimensions.cone_second_order > 0
        constraint_dual_jacobian_variables!(
            gradient, 
            trajopt.indices.state_action, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters,
            trajopt.data.duals_second_order,
        )
    end
    return
end

function equality_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables, equality_duals) where T
    fill!(jacobian, 0.0)

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
    jacobian_variables_variables!(
        jacobian, 
        trajopt.sparsity.dynamics_jacobian_variables_variables, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_dynamics)
    if trajopt.dimensions.equality_constraints > 0 
        jacobian_variables_variables!(
            jacobian, 
            trajopt.sparsity.equality_jacobian_variables_variables, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_equality)
    end
    return 
end

function equality_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables, equality_duals) where T
    fill!(jacobian, 0.0)

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
    jacobian_variables_parameters!(
        jacobian, 
        trajopt.sparsity.dynamics_jacobian_variables_parameters, 
        trajopt.data.dynamics, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters, 
        trajopt.data.duals_dynamics)
    if trajopt.dimensions.equality_constraints > 0 
        jacobian_variables_parameters!(
            jacobian, 
            trajopt.sparsity.equality_jacobian_variables_parameters, 
            trajopt.data.equality, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_equality)
    end
    return 
end

function cone_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables, cone_duals) where T
    fill!(jacobian, 0.0)
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
    if trajopt.dimensions.cone_nonnegative > 0 
        jacobian_variables_variables!(
            jacobian, 
            trajopt.sparsity.nonnegative_jacobian_variables_variables, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_nonnegative)
    end
    if trajopt.dimensions.cone_second_order > 0 
        jacobian_variables_variables!(
            jacobian, 
            trajopt.sparsity.second_order_jacobian_variables_variables, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_second_order)
    end
    return 
end

function cone_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem{T}, variables, cone_duals) where T
    fill!(jacobian, 0.0)
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
    if trajopt.dimensions.cone_nonnegative > 0 
        jacobian_variables_parameters!(
            jacobian, 
            trajopt.sparsity.nonnegative_jacobian_variables_parameters, 
            trajopt.data.nonnegative, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_nonnegative)
    end
    if trajopt.dimensions.cone_second_order > 0 
        jacobian_variables_parameters!(
            jacobian, 
            trajopt.sparsity.second_order_jacobian_variables_parameters, 
            trajopt.data.second_order, 
            trajopt.data.states, 
            trajopt.data.actions, 
            trajopt.data.parameters, 
            trajopt.data.duals_second_order)
    end
    return 
end

