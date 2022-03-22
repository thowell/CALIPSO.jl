struct TrajectoryOptimizationData{T}
    states::Vector{Vector{T}}
    actions::Vector{Vector{T}}
    parameters::Vector{Vector{T}}
    objective::Objective{T}
    dynamics::Vector{Dynamics{T}}
    constraints::Constraints{T}
    bounds::Bounds{T}
    duals_dynamics::Vector{Vector{T}} 
    duals_constraints::Vector{Vector{T}}
    state_dimensions::Vector{Int}
    action_dimensions::Vector{Int}
    parameter_dimensions::Vector{Int}
end

function TrajectoryOptimizationData(obj::Objective{T}, dynamics::Vector{Dynamics{T}}, constraints::Constraints{T}, bounds::Bounds{T}; 
    parameters=[zeros(num_parameter) for num_parameter in dimensions(dynamics)[3]]) where T

    state_dimensions, action_dimensions, parameter_dimensions = dimensions(dynamics)

    states = [zeros(num_state) for num_state in state_dimensions]
    actions = [zeros(nu) for nu in action_dimensions]

    duals_dynamics = [zeros(num_state) for num_state in state_dimensions[2:end]]
    duals_constraints = [zeros(con.num_constraint) for con in constraints]
   
    TrajectoryOptimizationData(
        states, 
        actions, 
        parameters,
        obj, 
        dynamics, 
        constraints, 
        bounds, 
        duals_dynamics, 
        duals_constraints, 
        state_dimensions, 
        action_dimensions, 
        parameter_dimensions)
end

TrajectoryOptimizationData(obj::Objective, dynamics::Vector{Dynamics}) = TrajectoryOptimizationData(obj, dynamics, [Constraint() for t = 1:length(dynamics)], [Bound() for t = 1:length(dynamics)])

struct TrajectoryOptimizationIndices 
    objective_hessians::Vector{Vector{Int}}
    dynamics_constraints::Vector{Vector{Int}} 
    dynamics_jacobians::Vector{Vector{Int}} 
    dynamics_hessians::Vector{Vector{Int}}
    stage_constraints::Vector{Vector{Int}} 
    stage_jacobians::Vector{Vector{Int}} 
    stage_hessians::Vector{Vector{Int}}
    general_constraint::Vector{Int}
    general_jacobian::Vector{Int}
    general_hessian::Vector{Int}
    states::Vector{Vector{Int}}
    actions::Vector{Vector{Int}}
    state_action::Vector{Vector{Int}}
    state_action_next_state::Vector{Vector{Int}}
end

function indices(obj::Objective{T}, dynamics::Vector{Dynamics{T}}, constraints::Constraints{T}, general::GeneralConstraint{T},
    key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_trajectory::Int) where T 
    # Jacobians
    dynamics_constraints = constraint_indices(dynamics, 
        shift=0)
    dynamics_jacobians = jacobian_indices(dynamics, 
        shift=0)
    stage_constraints = constraint_indices(constraints, 
        shift=num_constraint(dynamics))
    stage_jacobians = jacobian_indices(constraints, 
        shift=num_jacobian(dynamics)) 
    general_constraint = constraint_indices(general, 
        shift=(num_constraint(dynamics) + num_constraint(constraints)))
    general_jacobian = jacobian_indices(general, 
        shift=(num_jacobian(dynamics) + num_jacobian(constraints))) 

    # Hessian of Lagrangian 
    objective_hessians = hessian_indices(obj, key, num_state, num_action)
    dynamics_hessians = hessian_indices(dynamics, key, num_state, num_action)
    stage_hessians = hessian_indices(constraints, key, num_state, num_action)
    general_hessian = hessian_indices(general, key, num_trajectory)

    # indices
    x_idx = state_indices(dynamics)
    u_idx = action_indices(dynamics)
    xu_idx = state_action_indices(dynamics)
    xuy_idx = state_action_next_state_indices(dynamics)

    return TrajectoryOptimizationIndices(
        objective_hessians, 
        dynamics_constraints, 
        dynamics_jacobians, 
        dynamics_hessians, 
        stage_constraints, 
        stage_jacobians, 
        stage_hessians,
        general_constraint, 
        general_jacobian, 
        general_hessian,
        x_idx, 
        u_idx, 
        xu_idx, 
        xuy_idx) 
end

struct NLPData{T} <: MOI.AbstractNLPEvaluator
    trajopt::TrajectoryOptimizationData{T}
    num_variables::Int                 
    num_constraint::Int 
    num_jacobian::Int 
    num_hessian_lagrangian::Int               
    variable_bounds::Vector{Vector{T}}
    constraint_bounds::Vector{Vector{T}}
    indices::TrajectoryOptimizationIndices
    jacobian_sparsity
    hessian_lagrangian_sparsity
    hessian_lagrangian::Bool 
    general_constraint::GeneralConstraint{T}
    parameters::Vector{T}
    duals_general::Vector{T}
end

function primal_bounds(bounds::Bounds{T}, num_variables::Int, state_indices::Vector{Vector{Int}}, action_indices::Vector{Vector{Int}}) where T 
    lower, upper = -Inf * ones(num_variables), Inf * ones(num_variables) 
    for (t, bnd) in enumerate(bounds)
        length(bnd.state_lower)  > 0 && (lower[state_indices[t]]  = bnd.state_lower)
        length(bnd.state_upper)  > 0 && (upper[state_indices[t]]  = bnd.state_upper)
        length(bnd.action_lower) > 0 && (lower[action_indices[t]] = bnd.action_lower)
        length(bnd.action_upper) > 0 && (upper[action_indices[t]] = bnd.action_upper)
    end
    return lower, upper
end

function constraint_bounds(constraints::Constraints{T}, general::GeneralConstraint{T}, 
    num_dynamics::Int, num_stage::Int, indices::TrajectoryOptimizationIndices) where T
    # total constraints
    total_constraint = num_dynamics + num_stage + general.num_constraint
    # bounds
    lower, upper = zeros(total_constraint), zeros(total_constraint) 
    # stage
    for (t, con) in enumerate(constraints) 
        lower[indices.stage_constraints[t][con.indices_inequality]] .= -Inf
    end
    # general
    lower[collect(num_dynamics + num_stage .+ general.indices_inequality)] .= -Inf
    return lower, upper
end 

function NLPData(trajopt::TrajectoryOptimizationData; 
    evaluate_hessian=false, 
    general_constraint=GeneralConstraint()) 

    # number of variables
    total_variables = sum(trajopt.state_dimensions) + sum(trajopt.action_dimensions)

    # number of constraints
    num_dynamics = num_constraint(trajopt.dynamics)
    num_stage = num_constraint(trajopt.constraints)  
    num_general = num_constraint(general_constraint)
    total_constraints = num_dynamics + num_stage + num_general

    # number of nonzeros in constraint Jacobian
    num_dynamics_jacobian = num_jacobian(trajopt.dynamics)
    num_constraint_jacobian = num_jacobian(trajopt.constraints)  
    num_general_jacobian = num_jacobian(general_constraint)
    total_jacobians = num_dynamics_jacobian + num_constraint_jacobian + num_general_jacobian

    # number of nonzeros in Hessian of Lagrangian
    num_hessian_lagrangian = 0

    # constraint Jacobian sparsity
    sparsity_dynamics = sparsity_jacobian(trajopt.dynamics, trajopt.state_dimensions, trajopt.action_dimensions, 
        row_shift=0)
    sparsity_constraint = sparsity_jacobian(trajopt.constraints, trajopt.state_dimensions, trajopt.action_dimensions, 
        row_shift=num_dynamics)
    sparsity_general = sparsity_jacobian(general_constraint, total_variables, row_shift=(num_dynamics + num_stage))
    jacobian_sparsity = collect([sparsity_dynamics..., sparsity_constraint..., sparsity_general...]) 

    # Hessian of Lagrangian sparsity 
    sparsity_objective_hessians = sparsity_hessian(trajopt.objective, trajopt.state_dimensions, trajopt.action_dimensions)
    sparsity_dynamics_hessians = sparsity_hessian(trajopt.dynamics, trajopt.state_dimensions, trajopt.action_dimensions)
    sparsity_constraint_hessian = sparsity_hessian(trajopt.constraints, trajopt.state_dimensions, trajopt.action_dimensions)
    sparsity_general_hessian = sparsity_hessian(general_constraint, total_variables)
    hessian_lagrangian_sparsity = [sparsity_objective_hessians..., sparsity_dynamics_hessians..., sparsity_constraint_hessian..., sparsity_general_hessian...]
    hessian_lagrangian_sparsity = !isempty(hessian_lagrangian_sparsity) ? hessian_lagrangian_sparsity : Tuple{Int,Int}[]
    hessian_sparsity_key = sort(unique(hessian_lagrangian_sparsity))

    # indices 
    idx = indices(
        trajopt.objective, 
        trajopt.dynamics, 
        trajopt.constraints, 
        general_constraint, 
        hessian_sparsity_key, 
        trajopt.state_dimensions, 
        trajopt.action_dimensions, 
        total_variables)

    # primal variable bounds
    primal_lower, primal_upper = primal_bounds(trajopt.bounds, total_variables, idx.states, idx.actions) 

    # nonlinear constraint bounds
    constraint_lower, constraint_upper = constraint_bounds(trajopt.constraints, general_constraint, num_dynamics, num_stage, idx) 

    NLPData(trajopt, 
        total_variables, 
        total_constraints, 
        total_jacobians, 
        num_hessian_lagrangian, 
        [primal_lower, primal_upper], 
        [constraint_lower, constraint_upper], 
        idx, 
        jacobian_sparsity, 
        hessian_sparsity_key,
        evaluate_hessian, 
        general_constraint,
        vcat(trajopt.parameters...),
        zeros(general_constraint.num_constraint))
end

struct SolverData 
    # nlp_bounds::Vector{MOI.NLPBoundsPair}
    # block_data::MOI.NLPBlockData
    # solver::Ipopt.Optimizer
    # variables::Vector{MOI.VariableIndex} 
end

function SolverData(nlp::NLPData; 
    options=Options()) 

    return SolverData()

    # # solver
    # nlp_bounds = MOI.NLPBoundsPair.(nlp.constraint_bounds...)
    # block_data = MOI.NLPBlockData(nlp_bounds, nlp, true)
    
    # # instantiate NLP solver
    # solver = Ipopt.Optimizer()

    # # set NLP solver options
    # for name in fieldnames(typeof(options))
    #     solver.options[String(name)] = getfield(options, name)
    # end
    
    # z = MOI.add_variables(solver, nlp.num_variables)
    
    # for i = 1:nlp.num_variables
    #     MOI.add_constraint(solver, z[i], MOI.LessThan(nlp.variable_bounds[2][i]))
    #     MOI.add_constraint(solver, z[i], MOI.GreaterThan(nlp.variable_bounds[1][i]))
    # end
    
    # MOI.set(solver, MOI.NLPBlock(), block_data)
    # MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE) 

    # return SolverData(nlp_bounds, block_data, solver, z)
end


function trajectory!(states::Vector{Vector{T}}, actions::Vector{Vector{T}}, 
    trajectory::Vector{T}, 
    state_indices::Vector{Vector{Int}}, action_indices::Vector{Vector{Int}}) where T
    for (t, idx) in enumerate(state_indices)
        states[t] .= @views trajectory[idx]
    end 
    for (t, idx) in enumerate(action_indices)
        actions[t] .= @views trajectory[idx]
    end
end

function duals!(duals_dynamics::Vector{Vector{T}}, duals_constraints::Vector{Vector{T}}, duals_general::Vector{T}, duals, 
    dynamics_indices::Vector{Vector{Int}}, constraint_indices::Vector{Vector{Int}}, general_indices::Vector{Int}) where T
    for (t, idx) in enumerate(dynamics_indices)
        duals_dynamics[t] .= @views duals[idx]
    end 
    for (t, idx) in enumerate(constraint_indices)
        duals_constraints[t] .= @views duals[idx]
    end
    duals_general .= duals[general_indices]
end

