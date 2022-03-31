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

function problem!(data::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T};
    gradient=true,
    constraint=true,
    jacobian=true,
    hessian=true) where T

    x = @views variables[idx.variables]
    y = @views variables[idx.equality_dual]
    z = @views variables[idx.inequality_dual]

    # TODO: remove final allocations
    gradient && methods.objective_gradient(data.objective_gradient, x)
    hessian && methods.objective_hessian(data.objective_hessian, x)

    constraint && methods.equality(data.equality, x)
    jacobian && methods.equality_jacobian(data.equality_jacobian, x)
    hessian && methods.equality_hessian(data.equality_hessian, x, y)

    constraint && methods.inequality(data.inequality, x)
    jacobian && methods.inequality_jacobian(data.inequality_jacobian, x)
    hessian && methods.inequality_hessian(data.inequality_hessian, x, z)

    return
end