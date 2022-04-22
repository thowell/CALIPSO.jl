mutable struct Solver{T}
    problem::ProblemData{T} 
    methods::ProblemMethods 
    data::SolverData{T}
    variables::Vector{T} 
    candidate::Vector{T}
    parameters::Vector{T}
    indices::Indices
    dimensions::Dimensions
    linear_solver::LinearSolver
    central_path::Vector{T} 
    fraction_to_boundary::Vector{T}
    penalty::Vector{T}
    dual::Vector{T} 

    regularization::Vector{T}
    primal_regularization::T 
    primal_regularization_last::T
    dual_regularization::T

    options::Options{T}
end

function Solver(methods, num_variables, num_parameters, num_equality, num_cone; 
    parameters=zeros(num_parameters),
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    options=Options())

    # problem data
    p_data = ProblemData(num_variables, num_parameters, num_equality, num_cone)

    # solver data
    s_data = SolverData(num_variables, num_parameters, num_equality, num_cone)

    # indices
    idx = Indices(num_variables, num_parameters, num_equality, num_cone;
        nonnegative=nonnegative_indices,
        second_order=second_order_indices)

    # dimensions 
    dim = Dimensions(num_variables, num_parameters, num_equality, num_cone;
        nonnegative=length(nonnegative_indices),
        second_order=[length(idx_soc) for idx_soc in second_order_indices])

    # variables 
    variables = zeros(dim.total) 
    candidate = zeros(dim.total)

    # interior-point 
    central_path = [0.1] 
    fraction_to_boundary = [max(0.99, 1.0 - central_path[1])]

    # augmented Lagrangian 
    penalty = [10.0] 
    dual = zeros(num_equality) 

    # linear solver TODO: constructor
    random_variables = randn(dim.total)
    problem!(p_data, methods, idx, random_variables, parameters,
        # objective=true,
        # objective_gradient_variables=true,
        objective_jacobian_variables_variables=true,
        # equality_constraint=true,
        equality_jacobian_variables=true,
        equality_dual_jacobian_variables_variables=true,
        # cone_constraint=true,
        cone_jacobian_variables=true,
        cone_dual_jacobian_variables_variables=true,
    )
    cone!(p_data, methods, idx, random_variables,
        # product=true, 
        jacobian=true,
        # target=true
    )
    residual_jacobian_variables!(s_data, p_data, idx, rand(1), rand(1), randn(num_equality), 1.0e-5, 1.0e-5,
        constraint_hessian=options.constraint_hessian)
    residual_jacobian_variables_symmetric!(s_data.jacobian_variables_symmetric, s_data.jacobian_variables, idx)

    linear_solver = ldl_solver(s_data.jacobian_variables_symmetric)

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
        parameters,
        idx, 
        dim,
        linear_solver,
        central_path,
        fraction_to_boundary,
        penalty, 
        dual,
        regularization, 
        primal_regularization, 
        primal_regularization_last,
        dual_regularization,
        options,
    )
end


