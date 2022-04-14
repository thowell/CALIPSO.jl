# problem data 
struct ProblemData{T}
    objective::Vector{T}
    objective_gradient_variables::Vector{T}
    objective_gradient_data::Vector{T}
    objective_jacobian_variables_variables::Matrix{T}
    objective_jacobian_variables_data::Matrix{T}
    equality_constraint::Vector{T} 
    equality_jacobian_variables::Matrix{T}
    equality_jacobian_data::Matrix{T}
    equality_dual::Vector{T}
    equality_dual_jacobian_variables::Vector{T}
    equality_dual_jacobian_variables_variables::Matrix{T}
    equality_dual_jacobian_variables_data::Matrix{T}  
    cone_constraint::Vector{T} 
    cone_jacobian_variables::Matrix{T}
    cone_jacobian_data::Matrix{T}
    cone_dual::Vector{T}
    cone_dual_jacobian_variables::Vector{T}
    cone_dual_jacobian_variables_variables::Matrix{T}
    cone_dual_jacobian_variables_data::Matrix{T}  
    cone_product::Vector{T} 
    cone_product_jacobian_primal::Matrix{T} 
    cone_product_jacobian_dual::Matrix{T} 
    cone_target::Vector{T}
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone)
    objective = zeros(1)
    objective_gradient_variables = zeros(num_variables)
    objective_gradient_data = zeros(num_parameters)
    objective_jacobian_variables_variables = zeros(num_variables, num_variables)
    objective_jacobian_variables_data = zeros(num_variables, num_parameters)

    equality_constraint = zeros(num_equality)
    equality_jacobian_variables = zeros(num_equality, num_variables)
    equality_jacobian_data = zeros(num_equality, num_parameters)
    equality_dual = zeros(1) 
    equality_dual_jacobian_variables = zeros(num_variables)
    equality_dual_jacobian_variables_variables = zeros(num_variables, num_variables)
    equality_dual_jacobian_variables_data = zeros(num_variables, num_parameters)

    cone_constraint = zeros(num_cone)
    cone_jacobian_variables = zeros(num_cone, num_variables)
    cone_jacobian_data = zeros(num_cone, num_parameters)
    cone_dual = zeros(1)
    cone_dual_jacobian_variables = zeros(num_variables)
    cone_dual_jacobian_variables_variables = zeros(num_variables, num_variables)
    cone_dual_jacobian_variables_data = zeros(num_variables, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_primal = zeros(num_cone, num_cone) 
    cone_product_jacobian_dual = zeros(num_cone, num_cone) 
    cone_target = zeros(num_cone)

    ProblemData(
        objective,                        
        objective_gradient_variables,
        objective_gradient_data,
        objective_jacobian_variables_variables,
        objective_jacobian_variables_data,
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_data,
        equality_dual,
        equality_dual_jacobian_variables,
        equality_dual_jacobian_variables_variables,
        equality_dual_jacobian_variables_data,
        cone_constraint,
        cone_jacobian_variables,
        cone_jacobian_data,
        cone_dual,
        cone_dual_jacobian_variables,
        cone_dual_jacobian_variables_variables,
        cone_dual_jacobian_variables_data, 
        cone_product, 
        cone_product_jacobian_primal,
        cone_product_jacobian_dual,
        cone_target,
    )
end

function problem!(problem::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T}, parameters::Vector{T};
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
    θ = parameters

    # TODO: remove final allocations
    objective && (problem.objective[1] = methods.objective(x, θ))
    objective_gradient && methods.objective_gradient_variables(problem.objective_gradient_variables, x, θ)
    objective_hessian && methods.objective_jacobian_variables_variables(problem.objective_jacobian_variables_variables, x, θ)

    equality_constraint && methods.equality_constraint(problem.equality_constraint, x, θ)
    equality_jacobian && methods.equality_jacobian_variables(problem.equality_jacobian_variables, x, θ)
    equality_hessian && methods.equality_dual_jacobian_variables_variables(problem.equality_dual_jacobian_variables_variables, x, θ, y)

    cone_constraint && methods.cone_constraint(problem.cone_constraint, x, θ)
    cone_jacobian && methods.cone_jacobian_variables(problem.cone_jacobian_variables, x, θ)
    cone_hessian && methods.cone_dual_jacobian_variables_variables(problem.cone_dual_jacobian_variables_variables, x, θ, z)

    return
end