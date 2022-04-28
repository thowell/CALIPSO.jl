function evaluate!(problem::ProblemData{T}, methods::ProblemMethods{O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP}, idx::Indices, solution::Point{T}, parameters::Vector{T};
    objective=false,
    objective_gradient_variables=false,
    objective_gradient_parameters=false,
    objective_jacobian_variables_variables=false,
    objective_jacobian_variables_parameters=false,
    equality_constraint=false,
    equality_jacobian_variables=false,
    equality_jacobian_parameters=false,
    equality_dual=false,
    equality_dual_jacobian_variables=false,
    equality_dual_jacobian_variables_variables=false,
    equality_dual_jacobian_variables_parameters=false,
    cone_constraint=false,
    cone_jacobian_variables=false,
    cone_jacobian_parameters=false,
    cone_dual=false,
    cone_dual_jacobian_variables=false,
    cone_dual_jacobian_variables_variables=false,
    cone_dual_jacobian_variables_parameters=false,
    ) where {T,O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP}

    x = solution.variables
    y = solution.equality_dual
    z = solution.cone_dual
    θ = parameters

    # dimensions
    nz = length(x) 
    nθ = length(θ)

    # objective
    objective && methods.objective(problem.objective, x, θ)
    (objective_gradient_variables && nz > 0) && methods.objective_gradient_variables(problem.objective_gradient_variables, x, θ)
    (objective_gradient_parameters && nθ > 0) && methods.objective_gradient_parameters(problem.objective_gradient_parameters, x, θ)
    
    if (objective_jacobian_variables_variables && nz > 0)
        methods.objective_jacobian_variables_variables(methods.objective_jacobian_variables_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.objective_jacobian_variables_variables_sparsity) 
            problem.objective_jacobian_variables_variables[idx...] = methods.objective_jacobian_variables_variables_cache[i]
        end
    end

    if (objective_jacobian_variables_parameters && nz > 0 && nθ > 0)
        methods.objective_jacobian_variables_parameters(methods.objective_jacobian_variables_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.objective_jacobian_variables_parameters_sparsity) 
            problem.objective_jacobian_variables_parameters[idx...] = methods.objective_jacobian_variables_parameters_cache[i]
        end
    end
    
    # equality
    ne = length(problem.equality_constraint)

    (equality_constraint && ne > 0) && methods.equality_constraint(problem.equality_constraint, x, θ)
    
    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity) 
            problem.equality_jacobian_variables[idx...] = methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity) 
            problem.equality_jacobian_parameters[idx...] = methods.equality_jacobian_parameters_cache[i]
        end
    end
    
    (equality_dual && ne > 0) && methods.equality_dual(problem.equality_dual, x, θ, y)
    (equality_dual_jacobian_variables && length(problem.equality_constraint) > 0) && methods.equality_dual_jacobian_variables(problem.equality_dual_jacobian_variables, x, θ, y)
    
    if (equality_dual_jacobian_variables_variables && ne > 0)
        methods.equality_dual_jacobian_variables_variables(methods.equality_dual_jacobian_variables_variables_cache, x, θ, y)
        for (i, idx) in enumerate(methods.equality_dual_jacobian_variables_variables_sparsity) 
            problem.equality_dual_jacobian_variables_variables[idx...] = methods.equality_dual_jacobian_variables_variables_cache[i]
        end
    end

    if (equality_dual_jacobian_variables_parameters && ne > 0 && nθ > 0)
        methods.equality_dual_jacobian_variables_parameters(methods.equality_dual_jacobian_variables_parameters_cache, x, θ, y)
        for (i, idx) in enumerate(methods.equality_dual_jacobian_variables_parameters_sparsity) 
            problem.equality_dual_jacobian_variables_parameters[idx...] = methods.equality_dual_jacobian_variables_parameters_cache[i]
        end
    end

    # cone
    nc = length(problem.cone_constraint)

    (cone_constraint && nc > 0) && methods.cone_constraint(problem.cone_constraint, x, θ)
    
    if (cone_jacobian_variables && nc > 0)
        methods.cone_jacobian_variables(methods.cone_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.cone_jacobian_variables_sparsity) 
            problem.cone_jacobian_variables[idx...] = methods.cone_jacobian_variables_cache[i]
        end
    end

    if (cone_jacobian_parameters && nc > 0 && nθ > 0)
        methods.cone_jacobian_parameters(methods.cone_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.cone_jacobian_parameters_sparsity) 
            problem.cone_jacobian_parameters[idx...] = methods.cone_jacobian_parameters_cache[i]
        end
    end
    
    (cone_dual && nc > 0) && methods.cone_dual(problem.cone_dual, x, θ, z)
    (cone_dual_jacobian_variables && nc > 0) && methods.cone_dual_jacobian_variables(problem.cone_dual_jacobian_variables, x, θ, z)
    
    if (cone_dual_jacobian_variables_variables && nc > 0)
        methods.cone_dual_jacobian_variables_variables(methods.cone_dual_jacobian_variables_variables_cache, x, θ, z)
        for (i, idx) in enumerate(methods.cone_dual_jacobian_variables_variables_sparsity) 
            problem.cone_dual_jacobian_variables_variables[idx...] = methods.cone_dual_jacobian_variables_variables_cache[i]
        end
    end

    if (cone_dual_jacobian_variables_parameters && nc > 0 && nθ > 0)
        methods.cone_dual_jacobian_variables_parameters(methods.cone_dual_jacobian_variables_parameters_cache, x, θ, z)
        for (i, idx) in enumerate(methods.cone_dual_jacobian_variables_parameters_sparsity) 
            problem.cone_dual_jacobian_variables_parameters[idx...] = methods.cone_dual_jacobian_variables_parameters_cache[i]
        end
    end

    return
end