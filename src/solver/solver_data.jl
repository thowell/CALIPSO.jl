struct SolverData{T}
    residual::Vector{T}
    residual_error::Vector{T}
    jacobian_variables::SparseMatrixCSC{T,Int}
    jacobian_parameters::Matrix{T}
    residual_symmetric::Vector{T} 
    jacobian_variables_symmetric::SparseMatrixCSC{T,Int}
    step::Vector{T}
    step_correction::Vector{T}
    step_symmetric::Vector{T}
    merit::Vector{T} 
    merit_gradient::Vector{T} 
    constraint_violation::Vector{T}
    filter::Vector{Tuple{T,T}}
    solution_sensitivity::Matrix{T}
end

function SolverData(num_variables, num_parameters, num_equality, num_cone;
    T=Float64)
    num_total = num_variables + num_equality + num_cone + num_equality + 2 * num_cone
    num_symmetric = num_variables + num_cone + num_equality

    residual = zeros(num_total)
    residual_error = zeros(num_total)

    jacobian_variables = spzeros(num_total, num_total)
    jacobian_parameters = zeros(num_total, num_parameters)

    residual_symmetric = zeros(num_symmetric)
    jacobian_variables_symmetric = spzeros(num_symmetric, num_symmetric)

    step = zeros(num_total) 
    step_correction = zeros(num_total) 
    step_symmetric = zeros(num_symmetric)

    merit = zeros(1) 
    merit_gradient = zeros(num_variables + num_equality + num_cone)

    constraint_violation = zeros(num_equality + num_cone) 

    filter = Tuple{T,T}[]
    
    solution_sensitivity = zeros(num_total, num_parameters)

    SolverData(
        residual, 
        residual_error,
        jacobian_variables,
        jacobian_parameters,
        residual_symmetric,
        jacobian_variables_symmetric,
        step,
        step_correction,
        step_symmetric,
        merit, 
        merit_gradient, 
        constraint_violation,
        filter,
        solution_sensitivity,
    )
end