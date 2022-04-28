# problem data 
struct ProblemData{T,X}
    objective::Vector{T}
    objective_gradient_variables::Vector{T}
    objective_gradient_parameters::Vector{T}
    objective_jacobian_variables_variables::Matrix{T}
    objective_jacobian_variables_parameters::Matrix{T}
    equality_constraint::Vector{T} 
    equality_jacobian_variables::Matrix{T}
    equality_jacobian_parameters::Matrix{T}
    equality_dual::Vector{T}
    equality_dual_jacobian_variables::Vector{T}
    equality_dual_jacobian_variables_variables::Matrix{T}
    equality_dual_jacobian_variables_parameters::Matrix{T}  
    cone_constraint::Vector{T} 
    cone_jacobian_variables::Matrix{T}
    cone_jacobian_parameters::Matrix{T}
    cone_dual::Vector{T}
    cone_dual_jacobian_variables::Vector{T}
    cone_dual_jacobian_variables_variables::Matrix{T}
    cone_dual_jacobian_variables_parameters::Matrix{T}  
    cone_product::Vector{T} 
    cone_product_jacobian_primal::Matrix{T} 
    cone_product_jacobian_dual::Matrix{T} 
    cone_target::Vector{T}
    second_order_jacobians::Vector{Matrix{T}}
    second_order_jacobians_inverse::Vector{Matrix{T}}
    barrier::Vector{T} 
    barrier_gradient::Vector{T}
    custom::X
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone; 
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    custom=nothing)

    objective = zeros(1)
    objective_gradient_variables = zeros(num_variables)
    objective_gradient_parameters = zeros(num_parameters)
    objective_jacobian_variables_variables = zeros(num_variables, num_variables)
    objective_jacobian_variables_parameters = zeros(num_variables, num_parameters)

    equality_constraint = zeros(num_equality)
    equality_jacobian_variables = zeros(num_equality, num_variables)
    equality_jacobian_parameters = zeros(num_equality, num_parameters)
    equality_dual = zeros(1) 
    equality_dual_jacobian_variables = zeros(num_variables)
    equality_dual_jacobian_variables_variables = zeros(num_variables, num_variables)
    equality_dual_jacobian_variables_parameters = zeros(num_variables, num_parameters)

    cone_constraint = zeros(num_cone)
    cone_jacobian_variables = zeros(num_cone, num_variables)
    cone_jacobian_parameters = zeros(num_cone, num_parameters)
    cone_dual = zeros(1)
    cone_dual_jacobian_variables = zeros(num_variables)
    cone_dual_jacobian_variables_variables = zeros(num_variables, num_variables)
    cone_dual_jacobian_variables_parameters = zeros(num_variables, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_primal = zeros(num_cone, num_cone) 
    cone_product_jacobian_dual = zeros(num_cone, num_cone) 
    cone_target = zeros(num_cone)
    second_order_jacobians = [zeros(length(idx), length(idx)) for idx in second_order_indices]
    second_order_jacobians_inverse = [zeros(length(idx), length(idx)) for idx in second_order_indices]

    barrier = zeros(1) 
    barrier_gradient = zeros(num_variables + num_equality + num_cone)

    ProblemData(
        objective,                        
        objective_gradient_variables,
        objective_gradient_parameters,
        objective_jacobian_variables_variables,
        objective_jacobian_variables_parameters,
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        equality_dual,
        equality_dual_jacobian_variables,
        equality_dual_jacobian_variables_variables,
        equality_dual_jacobian_variables_parameters,
        cone_constraint,
        cone_jacobian_variables,
        cone_jacobian_parameters,
        cone_dual,
        cone_dual_jacobian_variables,
        cone_dual_jacobian_variables_variables,
        cone_dual_jacobian_variables_parameters, 
        cone_product, 
        cone_product_jacobian_primal,
        cone_product_jacobian_dual,
        cone_target,
        second_order_jacobians,
        second_order_jacobians_inverse,
        barrier, 
        barrier_gradient,
        custom,
    )
end

