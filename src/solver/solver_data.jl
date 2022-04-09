struct SolverData{T}
    residual::Vector{T}
    residual_error::Vector{T}
    matrix::SparseMatrixCSC{T,Int}
    residual_symmetric::Vector{T} 
    matrix_symmetric::SparseMatrixCSC{T,Int}
    step::Vector{T}
    step_correction::Vector{T}
    step_symmetric::Vector{T}
    merit::Vector{T} 
    merit_gradient::Vector{T} 
    constraint_violation::Vector{T}
end

function SolverData(num_variables, num_equality, num_cone)
    num_total = num_variables + num_equality + num_cone + num_equality + 2 * num_cone
    num_symmetric = num_variables + num_cone + num_equality

    residual = zeros(num_total)
    residual_error = zeros(num_total)
    matrix = spzeros(num_total, num_total)

    residual_symmetric = zeros(num_symmetric)
    matrix_symmetric = spzeros(num_symmetric, num_symmetric)

    step = zeros(num_total) 
    step_correction = zeros(num_total) 
    step_symmetric = zeros(num_symmetric)

    merit = zeros(1) 
    merit_gradient = zeros(num_variables + num_equality + num_cone)

    constraint_violation = zeros(num_equality + num_cone) 

    SolverData(
        residual, 
        residual_error,
        matrix,
        residual_symmetric,
        matrix_symmetric,
        step,
        step_correction,
        step_symmetric,
        merit, 
        merit_gradient, 
        constraint_violation,
    )
end