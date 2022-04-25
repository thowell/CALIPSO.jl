function objective!(obj, trajopt::TrajectoryOptimizationProblem, variables, parameters) 
    fill!(obj, 0.0)
    
    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

    cost(
        obj,
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
end

function objective_gradient_variables!(gradient, trajopt::TrajectoryOptimizationProblem, variables, parameters) 
    fill!(gradient, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

    gradient_variables!(gradient, 
        trajopt.indices.state_action, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
    return 
end

function objective_gradient_parameters!(gradient, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(gradient, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

    gradient_parameters!(gradient, 
        trajopt.indices.state_action, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions, 
        trajopt.data.parameters) 
    return 
end

function objective_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters) 
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

    jacobian_variables_variables!(
        jacobian, 
        trajopt.sparsity.objective_jacobian_variables_variables, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters)
   return
end

function objective_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters) 
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

    jacobian_variables_parameters!(
        jacobian, 
        trajopt.sparsity.objective_jacobian_variables_parameters, 
        trajopt.data.objective, 
        trajopt.data.states, 
        trajopt.data.actions,
        trajopt.data.parameters)
   return
end

function equality!(violations, trajopt::TrajectoryOptimizationProblem, variables, parameters) 
    fill!(violations, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

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

function cone!(violations, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(violations, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

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

function equality_jacobian_variables!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

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

function equality_jacobian_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

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

function equality_dual_jacobian_variables!(gradient, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(gradient, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_equality_trajectory .= duals

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

function cone_jacobian_variables!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters

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

function cone_jacobian_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
  
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

function cone_dual_jacobian_variables!(gradient, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(gradient, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_cone_trajectory .= duals 
    
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

function equality_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_equality_trajectory .= duals

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

function equality_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_equality_trajectory .= duals
    
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

function cone_jacobian_variables_variables!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_cone_trajectory .= duals

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

function cone_jacobian_variables_parameters!(jacobian, trajopt::TrajectoryOptimizationProblem, variables, duals, parameters)
    fill!(jacobian, 0.0)

    trajopt.data.state_action_trajectory .= variables 
    trajopt.data.parameter_trajectory .= parameters
    trajopt.data.duals_cone_trajectory .= duals

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

