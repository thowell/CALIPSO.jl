mutable struct Inertia
    positive::Int  # number of positve eigenvalues
    negative::Int  # number of negative eigenvalues
    zero::Int  # number of zero eigenvalues
end

function inertia(s)
    (s.linear_solver.inertia.positive == s.dimensions.variables
  && s.linear_solver.inertia.negative == s.dimensions.equality_dual + s.dimensions.cone_dual
  && s.linear_solver.inertia.zero     == 0)
end

function factorize_regularized_residual_jacobian_variables!(s)
    residual_jacobian_variables!(s.data, s.problem, s.indices, 
        s.central_path, s.penalty, s.dual,
        s.primal_regularization, s.dual_regularization,
        constraint_hessian=s.options.constraint_hessian)
    
    residual_jacobian_variables_symmetric!(s.data.jacobian_variables_symmetric, s.data.jacobian_variables, s.indices, 
        s.problem.second_order_jacobians,
        s.problem.second_order_jacobians_inverse) 
    
    factorize!(s.linear_solver, s.data.jacobian_variables_symmetric;
        update=s.options.update_factorization)
    
    compute_inertia!(s.linear_solver)
    return nothing
end

function inertia_correction!(s)
    s.primal_regularization[1] = s.options.primal_regularization_initial
    s.dual_regularization[1] = s.options.dual_regularization_initial
    
    # IC-1
    factorize_regularized_residual_jacobian_variables!(s)
    
    if inertia(s)
        return nothing
    end

    # IC-2
    if s.linear_solver.inertia.zero != 0
        s.options.verbose ? (@warn "$(s.linear_solver.inertia.zero) zero eigen values - rank deficient constraints") : nothing
        s.dual_regularization[1] = s.options.dual_regularization * s.central_path[1]^s.options.exponent_dual_regularization
    end

    # IC-3
    if s.primal_regularization_last == 0.0
        s.primal_regularization[1] = s.options.primal_regularization_initial
    else
        s.primal_regularization[1] = max(s.options.min_regularization, s.options.scaling_regularization_last * s.primal_regularization_last[1])
    end

    while !inertia(s)
        # IC-4
        factorize_regularized_residual_jacobian_variables!(s)

        if inertia(s)
            break
        else
            # IC-5
            if s.primal_regularization_last[1] == 0.0
                s.primal_regularization[1] = s.options.scaling_regularization_initial * s.primal_regularization[1]
            else
                s.primal_regularization[1] = s.options.scaling_regularization * s.primal_regularization[1]
            end
        end

        # IC-6
        if s.primal_regularization[1] > s.options.max_regularization
            # TODO: handle inertia correction failure gracefully
            error("inertia correction failure")
        end
    end

    s.primal_regularization_last[1] = s.primal_regularization[1]

    return nothing
end

