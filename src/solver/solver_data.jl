struct SolverData{T}
    residual::Point{T}
    residual_error::Point{T}
    jacobian_variables::SparseMatrixCSC{T,Int}
    jacobian_parameters::Matrix{T}
    jacobian_parameters_vectors::Vector{Point{T}}
    residual_symmetric::PointSymmetric{T} 
    jacobian_variables_symmetric::SparseMatrixCSC{T,Int}
    step::Point{T}
    step_correction::Point{T}
    step_symmetric::PointSymmetric{T}
    merit::Vector{T} 
    merit_gradient::Vector{T} 
    constraint_violation::Vector{T}
    filter::Filter{T}
    solution_sensitivity::Matrix{T}
    solution_sensitivity_vectors::Vector{Point{T}}
    residual_second_order::SecondOrderViews{T}
    residual_error_second_order::SecondOrderViews{T} 
    step_second_order::SecondOrderViews{T} 
    step_correction_second_order::SecondOrderViews{T}
    jacobian_parameters_vectors_second_order::Vector{SecondOrderViews{T}}
    solution_sensitivity_vectors_second_order::Vector{SecondOrderViews{T}}
end

function SolverData(dims::Dimensions, idx::Indices;
    max_filter=1000,
    T=Float64)

    num_variables = dims.variables 
    num_parameters = dims.parameters 
    num_equality = dims.equality_dual
    num_cone = dims.cone_dual 

    num_total = dims.total
    num_symmetric = dims.symmetric

    residual = Point(dims, idx)
    residual_error = Point(dims, idx)

    jacobian_variables = spzeros(num_total, num_total)
    jacobian_parameters = zeros(num_total, num_parameters)
    jacobian_parameters_vectors = [Point(dims, idx) for i = 1:num_parameters]

    residual_symmetric = PointSymmetric(dims, idx)
    jacobian_variables_symmetric = spzeros(num_symmetric, num_symmetric)

    step = Point(dims, idx)
    step_correction = Point(dims, idx)
    step_symmetric = PointSymmetric(dims, idx)

    merit = zeros(1) 
    merit_gradient = zeros(num_variables + num_equality + num_cone)

    constraint_violation = zeros(num_equality + num_cone) 

    filter = Filter(; max_length=max_filter)

    solution_sensitivity = zeros(num_total, num_parameters)
    solution_sensitivity_vectors = [Point(dims, idx) for i = 1:num_parameters]

    residual_second_order = SecondOrderViews(residual, idx)
    residual_error_second_order = SecondOrderViews(residual_error, idx)
    step_second_order = SecondOrderViews(step, idx)
    step_correction_second_order = SecondOrderViews(step_correction, idx)
    jacobian_parameters_vectors_second_order = [SecondOrderViews(jp, idx) for jp in jacobian_parameters_vectors]
    solution_sensitivity_vectors_second_order = [SecondOrderViews(ss, idx) for ss in solution_sensitivity_vectors]

    SolverData(
        residual, 
        residual_error,
        jacobian_variables,
        jacobian_parameters,
        jacobian_parameters_vectors,
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
        solution_sensitivity_vectors,
        residual_second_order,
        residual_error_second_order,
        step_second_order,
        step_correction_second_order,
        jacobian_parameters_vectors_second_order,
        solution_sensitivity_vectors_second_order,
    )
end