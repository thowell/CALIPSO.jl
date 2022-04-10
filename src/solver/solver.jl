mutable struct Solver{T}
    problem::ProblemData{T} 
    methods::ProblemMethods 
    data::SolverData{T}
    variables::Vector{T} 
    candidate::Vector{T}
    indices::Indices
    dimensions::Dimensions
    linear_solver::LinearSolver
    central_path::Vector{T} 
    penalty::Vector{T}
    dual::Vector{T} 

    regularization::Vector{T}
    primal_regularization::T 
    primal_regularization_last::T
    dual_regularization::T

    options::Options{T}
end

function Solver(methods, num_variables, num_equality, num_cone; 
    options=Options())

    # problem data
    p_data = ProblemData(num_variables, num_equality, num_cone)

    # solver data
    s_data = SolverData(num_variables, num_equality, num_cone)

    # indices
    idx = Indices(num_variables, num_equality, num_cone)

    # dimensions 
    dim = Dimensions(num_variables, num_equality, num_cone)

    # variables 
    variables = zeros(dim.total) 
    candidate = zeros(dim.total)

    # interior-point 
    central_path = [1.0] 

    # augmented Lagrangian 
    penalty = [1.0] 
    dual = zeros(num_equality) 

    # linear solver TODO: constructor
    random_variables = randn(dim.total)
    problem!(p_data, methods, idx, random_variables,
        gradient=true,
        constraint=true,
        jacobian=true,
        hessian=true)
    cone!(p_data, methods, idx, random_variables,
        product=true, 
        jacobian=true,
        target=true)
    matrix!(s_data, p_data, idx, random_variables, rand(1), rand(1), randn(num_equality), 1.0e-5, 1.0e-5)
    matrix_symmetric!(s_data.matrix_symmetric, s_data.matrix, idx)

    linear_solver = ldl_solver(s_data.matrix_symmetric)

    # regularization 
    regularization = zeros(dim.total)
    primal_regularization = 0.0 
    primal_regularization_last = 0.0 
    dual_regularization = 0.0

    Solver(
        p_data, 
        methods, 
        s_data,
        variables,
        candidate, 
        idx, 
        dim,
        linear_solver,
        central_path, 
        penalty, 
        dual,
        regularization, 
        primal_regularization, 
        primal_regularization_last,
        dual_regularization,
        options,
    )
end


