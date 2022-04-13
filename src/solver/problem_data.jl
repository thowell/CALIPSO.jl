# problem data 
struct ProblemData{T}
    objective::Vector{T}
    objective_gradient::Vector{T}
    objective_hessian::Matrix{T}
    equality_constraint::Vector{T} 
    equality_jacobian::Matrix{T}
    equality_hessian::Matrix{T} 
    cone_constraint::Vector{T} 
    cone_jacobian::Matrix{T}
    cone_hessian::Matrix{T}
    cone_product::Vector{T} 
    cone_product_jacobian_primal::Matrix{T} 
    cone_product_jacobian_dual::Matrix{T} 
    cone_target::Vector{T}
end

function ProblemData(num_variables, num_equality, num_cone)
    objective = zeros(1)
    objective_gradient = zeros(num_variables)
    objective_hessian = zeros(num_variables, num_variables)
    
    equality_constraint = zeros(num_equality)
    equality_jacobian = zeros(num_equality, num_variables)
    equality_hessian = zeros(num_variables, num_variables)

    cone_constraint = zeros(num_cone)
    cone_jacobian = zeros(num_cone, num_variables)
    cone_hessian = zeros(num_variables, num_variables)

    cone_product = zeros(num_cone)
    cone_product_jacobian_primal = zeros(num_cone, num_cone) 
    cone_product_jacobian_dual = zeros(num_cone, num_cone) 
    cone_target = zeros(num_cone)

    ProblemData(
        objective,
        objective_gradient,
        objective_hessian,
        equality_constraint,
        equality_jacobian,
        equality_hessian,
        cone_constraint,
        cone_jacobian, 
        cone_hessian,
        cone_product, 
        cone_product_jacobian_primal,
        cone_product_jacobian_dual,
        cone_target,
    )
end

function problem!(problem::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T};
    objective=false,
    objective_gradient=false,
    objective_hessian=false,
    equality_constraint=false,
    equality_jacobian=false,
    equality_hessian=false,
    cone_constraint=false,
    cone_jacobian=false,
    cone_hessian=false,
    ) where T

    x = @views variables[idx.variables]
    y = @views variables[idx.equality_dual]
    z = @views variables[idx.cone_dual]

    # TODO: remove final allocations
    objective && (problem.objective[1] = methods.objective(x))
    objective_gradient && methods.objective_gradient(problem.objective_gradient, x)
    objective_hessian && methods.objective_hessian(problem.objective_hessian, x)

    equality_constraint && methods.equality_constraint(problem.equality_constraint, x)
    equality_jacobian && methods.equality_jacobian(problem.equality_jacobian, x)
    equality_hessian && methods.equality_hessian(problem.equality_hessian, x, y)

    cone_constraint && methods.cone_constraint(problem.cone_constraint, x)
    cone_jacobian && methods.cone_jacobian(problem.cone_jacobian, x)
    cone_hessian && methods.cone_hessian(problem.cone_hessian, x, z)

    return
end