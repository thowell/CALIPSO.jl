mutable struct Inertia
    positive::Int  # number of positve eigenvalues
    negative::Int  # number of negative eigenvalues
    zero::Int  # number of zero eigenvalues
end

function inertia(s)
    (s.linear_solver.inertia.positive == s.dimensions.variables
  && s.linear_solver.inertia.negative == s.dimensions.equality + s.dimensions.inequality
  && s.linear_solver.inertia.zero     == 0)
end

function factorize_regularized_matrix!(s::Solver)
    matrix!(s.data, s.problem, s.indices, s.variables, 
        s.central_path, s.penalty, s.dual,
        s.primal_regularization, s.dual_regularization)

    matrix_symmetric!(s.data.matrix_symmetric, s.data.matrix, s.indices) 

    factorize!(s.linear_solver, s.data.matrix_symmetric)
    compute_inertia!(s.linear_solver)

    return nothing
end

function inertia_correction!(s)

    # initialize_regularization!(s.linear_solver, s)
    s.primal_regularization = 1.0e-7
    s.dual_regularization = 1.0e-7

    # IC-1
    factorize_regularized_matrix!(s)

    if inertia(s)
        return nothing
    end

    # IC-2
    if s.linear_solver.inertia.zero != 0
        s.options.verbose ? (@warn "$(s.linear_solver.inertia.zero) zero eigen values - rank deficient constraints") : nothing
        s.dual_regularization = s.options.dual_regularization * s.central_path[1]^s.options.exponent_dual_regularization
    end

    # IC-3
    if s.primal_regularization_last == 0.0
        s.primal_regularization = s.options.primal_regularization_initial
    else
        s.primal_regularization = max(s.options.min_regularization, s.options.scaling_regularization_last * s.primal_regularization_last)
    end

    while !inertia(s)
        # IC-4
        factorize_regularized_matrix!(s)

        if inertia(s)
            break
        else
            # IC-5
            if s.primal_regularization_last == 0.0
                s.primal_regularization = s.options.scaling_regularization_initial * s.primal_regularization
            else
                s.primal_regularization = s.options.scaling_regularization * s.primal_regularization
            end
        end

        # IC-6
        if s.primal_regularization > s.options.max_regularization
            # TODO: handle inertia correction failure gracefully
            error("inertia correction failure")
        end
    end

    s.primal_regularization_last = s.primal_regularization

    return nothing
end

