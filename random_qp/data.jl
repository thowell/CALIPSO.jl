# problem data 
struct ProblemData{T}
    objective_gradient::Vector{T}
    objective_hessian::Matrix{T}
    equality::Vector{T} 
    equality_jacobian::Matrix{T}
    equality_hessian::Matrix{T} 
    inequality::Vector{T} 
    inequality_jacobian::Matrix{T}
    inequality_hessian::Matrix{T}
end

function ProblemData(num_variables, num_equality, num_inequality)
    objective_gradient = zeros(num_variables)
    objective_hessian = zeros(num_variables, num_variables)
    
    equality = zeros(num_equality)
    equality_jacobian = zeros(num_equality, num_variables)
    equality_hessian = zeros(num_variables, num_variables)

    inequality = zeros(num_inequality)
    inequality_jacobian = zeros(num_inequality, num_variables)
    inequality_hessian = zeros(num_variables, num_variables)

    ProblemData(
        objective_gradient,
        objective_hessian,
        equality,
        equality_jacobian,
        equality_hessian,
        inequality,
        inequality_jacobian, 
        inequality_hessian,
    )
end

struct ProblemMethods
    objective::Any 
    objective_gradient::Any 
    objective_hessian::Any 
    equality::Any 
    equality_jacobian::Any 
    equality_hessian::Any
    inequality::Any 
    inequality_jacobian::Any 
    inequality_hessian::Any
end

function ProblemMethods(num_variables::Int, objective::Function, equality::Function, inequality::Function )
    # generate methods
    obj, obj_grad!, obj_hess! = generate_gradients(objective, num_variables, :scalar)
    eq!, eq_jac!, eq_hess! = generate_gradients(equality, num_variables, :vector)
    ineq!, ineq_jac!, ineq_hess! = generate_gradients(inequality, num_variables, :vector)

    ProblemMethods(
        obj, 
        obj_grad!, 
        obj_hess!,
        eq!, 
        eq_jac!, 
        eq_hess!, 
        ineq!, 
        ineq_jac!, 
        ineq_hess!,
    )
end

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
    num_total = num_variables + num_inequality + num_equality + 2 * num_inequality
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