struct Solver{T,O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP,B,BX,P,PX,PXI,K}
    problem::ProblemData{T} 
    methods::ProblemMethods{T,O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP} 
    cone_methods::ConeMethods{B,BX,P,PX,PXI,K}
    data::SolverData{T}
    
    solution::Point{T} 
    candidate::Point{T}
    parameters::Vector{T}

    indices::Indices
    dimensions::Dimensions

    linear_solver::LDLSolver{T,Int}

    central_path::Vector{T} 
    fraction_to_boundary::Vector{T}
    penalty::Vector{T}
    dual::Vector{T} 

    primal_regularization::Vector{T} 
    primal_regularization_last::Vector{T}
    dual_regularization::Vector{T}

    options::Options{T}
end

function Solver(methods, num_variables, num_parameters, num_equality, num_cone; 
    parameters=zeros(num_parameters),
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    options=Options())

    # indices
    idx = Indices(num_variables, num_parameters, num_equality, num_cone;
        nonnegative=nonnegative_indices,
        second_order=second_order_indices)
    options.verbose && println("indices")

    # dimensions 
    dim = Dimensions(num_variables, num_parameters, num_equality, num_cone;
        nonnegative=length(nonnegative_indices),
        second_order=[length(idx_soc) for idx_soc in second_order_indices])

    options.verbose && println("dimensions")

    # cone methods 
    cone_methods = ConeMethods(num_cone, nonnegative_indices, second_order_indices)
    
    options.verbose && println("cone methods")

    # problem data
    p_data = ProblemData(num_variables, num_parameters, num_equality, num_cone;
        nonnegative_indices=nonnegative_indices,
        second_order_indices=second_order_indices,
    )

    options.verbose && println("problem")

    # solver data
    s_data = SolverData(dim, idx,
        max_filter=options.max_filter)

    options.verbose && println("solver data")
     
    # points 
    solution = Point(dim, idx)
    candidate = Point(dim, idx)

    # interior-point 
    central_path = [0.1] 
    fraction_to_boundary = [max(0.99, 1.0 - central_path[1])]
     
    # augmented Lagrangian 
    penalty = [10.0] 
    dual = zeros(num_equality) 

    # linear solver TODO: constructor
    random_solution = Point(dim, idx)
    random_solution.all .= randn(dim.total)
    random_solution.cone_slack .= max.(1.0, random_solution.cone_slack) 
    random_solution.cone_slack_dual .= max.(1.0, random_solution.cone_slack_dual) 
    options.verbose && println("random solution")

    # problem!(p_data, methods, idx, random_solution, parameters,
    #     objective_jacobian_variables_variables=true,
    #     equality_jacobian_variables=true,
    #     equality_dual_jacobian_variables_variables=true,
    #     cone_jacobian_variables=true,
    #     cone_dual_jacobian_variables_variables=true,
    # )

    options.verbose && println("evaluate problem")

    cone!(p_data, cone_methods, idx, random_solution,
        jacobian=true,
    )

    options.verbose && println("evalute cone")

    residual_jacobian_variables!(s_data, p_data, idx, rand(1), rand(1), randn(num_equality), 1.0e-5, 1.0e-5,
        constraint_hessian=options.constraint_hessian)

    options.verbose && println("evaluate residual")

    residual_jacobian_variables_symmetric!(s_data.jacobian_variables_symmetric, s_data.jacobian_variables, idx, 
        p_data.second_order_jacobians, p_data.second_order_jacobians)

    options.verbose && println("evaluate residual jacobian")

    linear_solver = ldl_solver(s_data.jacobian_variables_symmetric)

    options.verbose && println("solver")

    # regularization 
    primal_regularization = [0.0] 
    primal_regularization_last = [0.0] 
    dual_regularization = [0.0]

    Solver(
        p_data, 
        methods,
        cone_methods,
        s_data,
        solution,
        candidate, 
        parameters,
        idx, 
        dim,
        linear_solver,
        central_path,
        fraction_to_boundary,
        penalty, 
        dual,
        primal_regularization, 
        primal_regularization_last,
        dual_regularization,
        options,
    )
end


