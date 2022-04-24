function solve!(solver)
    # initialize
    initialize_slacks!(solver)
    initialize_duals!(solver)
    initialize_interior_point!(solver)
    initialize_augmented_lagrangian!(solver)

    # indices
    indices = solver.indices

    # variables
    solution = solver.solution
    x = solution.variables
    r = solution.equality_slack
    s = solution.cone_slack
    y = solution.equality_dual
    z = solution.cone_dual
    t = solution.cone_slack_dual

    # candidate
    candidate = solver.candidate
    x̂ = candidate.variables
    r̂ = candidate.equality_slack
    ŝ = candidate.cone_slack
    # ŷ = candidate.equality_dual
    # ẑ = candidate.cone_dual
    t̂ = candidate.cone_slack_dual

    # parameters
    parameters = solver.parameters

    # solver data
    data = solver.data

    # search direction
    step = data.step
    Δx = step.variables
    Δr = step.equality_slack
    Δs = step.cone_slack
    Δy = step.equality_dual
    Δz = step.cone_dual
    Δt = step.cone_slack_dual
    Δp = step.primals

    # constraints
    constraint_violation = solver.data.constraint_violation

    # problem
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods

    # barrier + augmented Lagrangian 
    κ = solver.central_path 
    τ = solver.fraction_to_boundary
    ρ = solver.penalty 
    λ = solver.dual 

    # options
    options = solver.options

    # counter
    total_iterations = 1

    # info
    options.verbose && solver_info(solver)

    # evaluate
    problem!(problem, methods, indices, solution, parameters,
        objective=true,
        equality_constraint=true,
        equality_jacobian_variables=true,
        cone_constraint=true,
    )

    equality_violation = norm(problem.equality_constraint, Inf) 
    cone_product_violation = norm(problem.cone_product, Inf) 

    cone!(problem, cone_methods, indices, solution,
        product=true,
        target=true
    )

    # initialize filter
    filter = solver.data.filter
    reset!(filter) 
    
    for j = 1:options.max_outer_iterations
        for i = 1:options.max_residual_iterations
            # evaluate
            problem!(problem, methods, indices, solution, parameters,
                objective_gradient_variables=true,
                equality_dual_jacobian_variables=true,
                cone_dual_jacobian_variables=true,
            )

            cone!(problem, cone_methods, indices, solution,
                barrier=true,
                barrier_gradient=true,
            )
            
            # merit
            M = merit(
                problem.objective[1],
                r, 
                problem.barrier[1],
                κ[1], λ, ρ[1],
                indices)

            merit_gradient!(
                    data.merit_gradient,
                    problem.objective_gradient_variables,
                    r, 
                    problem.barrier_gradient, 
                    κ[1], λ, ρ[1],
                    indices)
             
            # residual
            residual!(data, problem, indices, solution, κ, ρ, λ)

            # violations
            residual_violation = norm(data.residual.all, options.residual_norm) / solver.dimensions.total
            optimality_violation = optimality_error(solution, data.residual, indices)
            slack_violation = max(
                            norm(data.residual.equality_dual, Inf),
                            norm(data.residual.cone_dual, Inf),
            )
            
            # check outer convergence
            if (
                residual_violation < options.residual_tolerance &&
                slack_violation < options.slack_tolerance && 
                equality_violation <= options.equality_tolerance && 
                cone_product_violation <= options.complementarity_tolerance
                )

                # differentiate
                options.differentiate && differentiate!(solver)
                
                # status 
                options.verbose && iteration_status(
                    total_iterations, 
                    j, 
                    i, 
                    residual_violation, 
                    equality_violation, 
                    cone_product_violation, 
                    slack_violation, 
                    κ[1], 
                    ρ[1], 
                    1.0) 
                options.verbose && solver_status(solver, true)
                
                return true
            # check inner convergence
            elseif optimality_violation <= max(options.central_path_update_tolerance * κ[1], options.optimality_tolerance)
                break
            end

            # violation
            θ = constraint_violation!(constraint_violation,
                problem.equality_constraint, r, problem.cone_constraint, s, indices,
                norm_type=options.constraint_norm)
            
            # search direction
            problem!(problem, methods, indices, solution, parameters,
                objective_jacobian_variables_variables=true,
                equality_jacobian_variables=true,
                equality_dual_jacobian_variables_variables=options.constraint_hessian,
                cone_jacobian_variables=true,
                cone_dual_jacobian_variables_variables=options.constraint_hessian,
            )

            cone!(problem, cone_methods, indices, solution,
                jacobian=true,
            )
         
            search_direction!(solver)        

            # line search
            α = 1.0
            αt = 1.0

            # cone search
            for i = 1:solver.dimensions.cone_slack 
                ŝ[i] = s[i] - α * Δs[i] 
            end 

            for i = 1:solver.dimensions.cone_slack_dual 
                t̂[i] = t[i] - αt * Δt[i]
            end

            cone_iteration = 0

            while cone_violation(ŝ, s, τ, indices.cone_nonnegative, indices.cone_second_order)
                α = 0.5 * α 
                for i = 1:solver.dimensions.cone_slack 
                    ŝ[i] = s[i] - α * Δs[i] 
                end 
                cone_iteration += 1
                cone_iteration > options.max_cone_line_search && error("cone search failure")
            end

            cone_iteration = 0
            while cone_violation(t̂, t, τ, indices.cone_nonnegative, indices.cone_second_order)
                αt = 0.5 * αt
                for i = 1:solver.dimensions.cone_slack_dual 
                    t̂[i] = t[i] - αt * Δt[i]
                end
                cone_iteration += 1
                cone_iteration > options.max_cone_line_search && error("cone search failure")
            end 

            # decrease search
            for i = 1:solver.dimensions.variables
                x̂[i] = x[i] - α * Δx[i] 
            end
            for i = 1:solver.dimensions.equality_slack 
                r̂[i] = r[i] - α * Δr[i] 
            end

            problem!(problem, methods, indices, candidate, parameters,
                objective=true,
                equality_constraint=true,
                cone_constraint=true,
            )

            cone!(problem, cone_methods, indices, candidate,
                barrier=true,
                barrier_gradient=true,
            )

            M̂ = merit(
                problem.objective[1],
                r̂, 
                problem.barrier[1], 
                κ[1], λ, ρ[1],
                indices)

            θ̂  = constraint_violation!(constraint_violation,
                problem.equality_constraint, r̂, problem.cone_constraint, ŝ, indices,
                norm_type=options.constraint_norm)

            residual_iteration = 0
  
            while residual_iteration < options.max_residual_line_search
                # filter check
                if check_filter(θ̂ , M̂, filter)
                    if θ <= options.slack_tolerance && switching_condition(α, Δp, data.merit_gradient, options.merit_exponent, θ, options.violation_exponent, 1.0) && 
                        armijo(M, M̂, data.merit_gradient, Δp, α, options.armijo_tolerance, options.machine_tolerance) && break
                    elseif sufficient_progress(θ, θ̂ , M, M̂, options.violation_tolerance, options.merit_tolerance, options.machine_tolerance)
                        break
                    end                     
                end

                # decrease step size 
                α = 0.5 * α

                # update candidate
                for i = 1:solver.dimensions.variables
                    x̂[i] = x[i] - α * Δx[i] 
                end
                for i = 1:solver.dimensions.equality_slack 
                    r̂[i] = r[i] - α * Δr[i] 
                end
                for i = 1:solver.dimensions.cone_slack 
                    ŝ[i] = s[i] - α * Δs[i] 
                end

                problem!(problem, methods, indices, candidate, parameters,
                    objective=true,
                    equality_constraint=true,
                    cone_constraint=true,
                )

                cone!(problem, cone_methods, indices, solution,
                    barrier=true,
                    barrier_gradient=true,
                )
                M̂ = merit(
                    problem.objective[1],
                    r̂, 
                    problem.barrier[1], 
                    κ[1], λ, ρ[1],
                    indices)
                θ̂  = constraint_violation!(constraint_violation,
                    problem.equality_constraint, r̂, problem.cone_constraint, ŝ, indices,
                    norm_type=options.constraint_norm)

 
                residual_iteration += 1
                residual_iteration > options.max_residual_line_search && break
            end
 
             
            # update filter
            augment_filter!(solver, M, M̂, data.merit_gradient, θ, α, Δp)
             
            # update
            for i = 1:solver.dimensions.variables
                x[i] = x̂[i]
            end
            for i = 1:solver.dimensions.equality_slack
                r[i] = r̂[i]
            end
            for i = 1:solver.dimensions.cone_slack
                s[i] = ŝ[i] 
            end   
            for i = 1:solver.dimensions.equality_dual
                y[i] = y[i] - α * Δy[i] 
            end
            for i = 1:solver.dimensions.cone_dual
                z[i] = z[i] - α * Δz[i] 
            end
            for i = 1:solver.dimensions.cone_slack_dual
                t[i] = t̂[i] 
            end

            cone!(problem, cone_methods, indices, solution,
                product=true,
            )

            equality_violation = norm(problem.equality_constraint, Inf) 
            cone_product_violation = norm(problem.cone_product, Inf) 
            
            # status
            options.verbose && iteration_status(
                total_iterations, 
                j, 
                i, 
                residual_violation, 
                equality_violation, 
                cone_product_violation, 
                slack_violation, 
                κ[1], 
                ρ[1], 
                α) 
            
            total_iterations += 1
        end

        # central-path
        κ[1] = max(options.residual_tolerance / 10.0, min(options.central_path_scaling * κ[1], κ[1]^options.central_path_exponent))
        
        # fraction to the boundary
        τ[1] = max(0.99, 1.0 - κ[1])

        # augmented Lagrangian
        for i = 1:solver.dimensions.equality_slack 
            λ[i] = λ[i] + ρ[1] * r[i] 
        end
        ρ[1] = min(max(options.penalty_scaling * ρ[1], 1.0 / κ[1]), options.max_penalty)

        # reset filter
        reset!(filter)
    end

    # failure
    options.verbose && solver_status(solver, false)
    return false
end
