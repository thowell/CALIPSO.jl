struct SolverData{T}
    residual::Vector{T}
    residual_error::Vector{T}
    matrix::SparseMatrixCSC{T,Int}
    residual_symmetric::Vector{T} 
    matrix_symmetric::SparseMatrixCSC{T,Int}
    step::Vector{T}
    step_correction::Vector{T}
    step_symmetric::Vector{T}
end

function SolverData(num_variables, num_equality, num_inequality)
    num_total = num_variables + num_equality + num_inequality + num_equality + 2 * num_inequality
    num_symmetric = num_variables + num_inequality + num_equality

    residual = zeros(num_total)
    residual_error = zeros(num_total)
    matrix = spzeros(num_total, num_total)

    residual_symmetric = zeros(num_symmetric)
    matrix_symmetric = spzeros(num_symmetric, num_symmetric)

    step = zeros(num_total) 
    step_correction = zeros(num_total) 
    step_symmetric = zeros(num_symmetric)

    SolverData(
        residual, 
        residual_error,
        matrix,
        residual_symmetric,
        matrix_symmetric,
        step,
        step_correction,
        step_symmetric,
    )
end