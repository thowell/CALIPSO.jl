function solve!(solver)
    # initialize
    initialize_slacks!(solver)
    initialize_duals!(solver)
    initialize_interior_point!(solver)
    initialize_augmented_lagrangian!(solver)

    # indices 
    indices = solver.indices 

    # variables 
    variables = solver.variables 
    x = @views variables[indices.variables] 
    r = @views variables[indices.equality_slack] 
    s = @views variables[indices.cone_slack] 
    y = @views variables[indices.equality_dual] 
    z = @views variables[indices.cone_dual] 
    t = @views variables[indices.cone_slack_dual] 

    # candidate
    candidate = solver.candidate
    x̂ = @views candidate[indices.variables] 
    r̂ = @views candidate[indices.equality_slack] 
    ŝ = @views candidate[indices.cone_slack] 
    ŷ = @views candidate[indices.equality_dual] 
    ẑ = @views candidate[indices.cone_dual] 
    t̂ = @views candidate[indices.cone_slack_dual] 

    # parameters 
    parameters = solver.parameters 

    # solver data 
    data = solver.data

    # search direction 
    step = data.step
    Δx = @views step[indices.variables] 
    Δr = @views step[indices.equality_slack]
    Δs = @views step[indices.cone_slack]
    Δy = @views step[indices.equality_dual]
    Δz = @views step[indices.cone_dual] 
    Δt = @views step[indices.cone_slack_dual] 
    Δp = @views step[indices.primals]

    # constraints 
    constraint_violation = solver.data.constraint_violation

    # problem 
    problem = solver.problem 
    methods = solver.methods

    # barrier + augmented Lagrangian 
    κ = solver.central_path 
    ρ = solver.penalty 
    λ = solver.dual 

    # options 
    options = solver.options 

    # counter
    total_iterations = 1

    # evaluate 
    problem!(problem, methods, indices, variables, parameters,
        objective=true,
        equality_constraint=true,
        equality_jacobian_variables=true,
        cone_constraint=true,
        # cone_jacobian_variables=true,
    )

    cone!(problem, methods, indices, variables,
        product=true, 
        target=true
    )

    for j = 1:options.max_outer_iterations
        for i = 1:options.max_residual_iterations
            # iterations
            options.verbose && println("iter: ($j, $i, $total_iterations)")

            # evaluate
            problem!(problem, methods, indices, variables, parameters,
                objective_gradient_variables=true,
                # equality_jacobian_variables=true,
                equality_dual_jacobian_variables=true,
                cone_dual_jacobian_variables=true,
                # cone_jacobian_variables=true,
            )

            # merit
            M = merit(
                problem.objective[1], 
                r, s, κ[1], λ, ρ[1],
                indices)

            merit_grad = vcat(merit_gradient(
                problem.objective_gradient_variables,  
                r, s, κ[1], λ, ρ[1],
                indices)...)

            # residual
            residual!(data, problem, indices, variables, κ, ρ, λ)
            res_norm = norm(data.residual, options.residual_norm) / solver.dimensions.total
            options.verbose && println("res: $(res_norm)")

            # violation
            θ = constraint_violation!(constraint_violation, 
                problem.equality_constraint, r, problem.cone_constraint, s, indices,
                norm_type=options.constraint_norm)
            
            options.verbose && println("con: $(θ)")

            # check convergence
            if res_norm < options.residual_tolerance
                break 
            end

            # search direction
            problem!(problem, methods, indices, variables, parameters,
                objective_jacobian_variables_variables=true,
                equality_jacobian_variables=true,
                equality_dual_jacobian_variables_variables=true,
                cone_jacobian_variables=true,
                cone_dual_jacobian_variables_variables=true,
            )

            cone!(problem, methods, indices, variables,
                jacobian=true,
            )

            search_direction!(solver)

            # line search
            α = 1.0  
            αt = 1.0
            
            # cone search 
            ŝ .= s - α * Δs
            t̂ .= t - αt * Δt 

            cone_iteration = 0
            while cone_violation(ŝ, indices.cone_nonnegative, indices.cone_second_order)
                α = 0.5 * α 
                ŝ .= s - α * Δs
                cone_iteration += 1 
                cone_iteration > options.max_cone_line_search && error("cone search failure")
            end

            cone_iteration = 0
            while cone_violation(t̂, indices.cone_nonnegative, indices.cone_second_order)
                αt = 0.5 * αt
                t̂ .= t - αt * Δt
                cone_iteration += 1 
                cone_iteration > options.max_cone_line_search && error("cone search failure")
            end

            # decrease search 
            x̂ .= x - α * Δx
            r̂ .= r - α * Δr

            problem!(problem, methods, indices, candidate, parameters,
                objective=true,
                equality_constraint=true,
                cone_constraint=true,
            )

            M̂ = merit(
                problem.objective[1], 
                r̂, ŝ, κ[1], λ, ρ[1], 
                indices)
            θ̂  = constraint_violation!(constraint_violation, 
                problem.equality_constraint, r̂, problem.cone_constraint, ŝ, indices,
                norm_type=options.constraint_norm)
            d = options.armijo_tolerance * dot(Δp, merit_grad)

            residual_iteration = 0

            while M̂ > M + α * d && θ̂  > θ
                # decrease step size 
                α = 0.5 * α

                # update candidate
                x̂ .= x - α * Δx
                r̂ .= r - α * Δr
                ŝ .= s - α * Δs

                problem!(problem, methods, indices, candidate, parameters,
                    objective=true,
                    equality_constraint=true,
                    cone_constraint=true,
                )

                M̂ = merit(
                    problem.objective[1], 
                    r̂, ŝ, κ[1], λ, ρ[1],
                    indices)
                θ̂  = constraint_violation!(constraint_violation, 
                    problem.equality_constraint, r̂, problem.cone_constraint, ŝ, indices,
                    norm_type=options.constraint_norm)

                residual_iteration += 1 
                residual_iteration > options.max_residual_line_search && error("residual search failure")
            end

            options.verbose && println("α = $α")

            # update
            x .= x̂
            r .= r̂
            s .= ŝ 
            y .= y - α * Δy
            z .= z - α * Δz
            t .= t̂

            cone!(problem, methods, indices, variables,
                product=true, 
            )
            
            total_iterations += 1
            options.verbose && println("con: $(norm(solver.problem.equality_constraint, Inf))")
            options.verbose && println("comp: $(norm(solver.problem.cone_product, Inf))")

            options.verbose && println("")
        end

        # convergence
        if norm(problem.equality_constraint, Inf) <= options.equality_tolerance && norm(problem.cone_product, Inf) <= options.complementarity_tolerance
            options.verbose && println("solve success!")
            return true 
        # outer update
        else
            # central-path
            κ[1] = max(options.scaling_central_path * κ[1], options.min_central_path)
            
            # augmented Lagrangian
            λ .= λ + ρ[1] * r
            ρ[1] = min(options.scaling_penalty * ρ[1], options.max_penalty) 
        end
    end

    # failure
    return false
end