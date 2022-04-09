# problem data 
struct ProblemData{T}
    objective_gradient::Vector{T}
    objective_hessian::Matrix{T}
    equality::Vector{T} 
    equality_jacobian::Matrix{T}
    equality_hessian::Matrix{T} 
    cone::Vector{T} 
    cone_jacobian::Matrix{T}
    cone_hessian::Matrix{T}
end

function ProblemData(num_variables, num_equality, num_cone)
    objective_gradient = zeros(num_variables)
    objective_hessian = zeros(num_variables, num_variables)
    
    equality = zeros(num_equality)
    equality_jacobian = zeros(num_equality, num_variables)
    equality_hessian = zeros(num_variables, num_variables)

    cone = zeros(num_cone)
    cone_jacobian = zeros(num_cone, num_variables)
    cone_hessian = zeros(num_variables, num_variables)

    ProblemData(
        objective_gradient,
        objective_hessian,
        equality,
        equality_jacobian,
        equality_hessian,
        cone,
        cone_jacobian, 
        cone_hessian,
    )
end

function problem!(data::ProblemData{T}, methods::ProblemMethods, idx::Indices, variables::Vector{T};
    gradient=true,
    constraint=true,
    jacobian=true,
    hessian=true) where T

    x = @views variables[idx.variables]
    y = @views variables[idx.equality_dual]
    z = @views variables[idx.cone_dual]

    # TODO: remove final allocations
    gradient && methods.objective_gradient(data.objective_gradient, x)
    hessian && methods.objective_hessian(data.objective_hessian, x)

    constraint && methods.equality(data.equality, x)
    jacobian && methods.equality_jacobian(data.equality_jacobian, x)
    hessian && methods.equality_hessian(data.equality_hessian, x, y)

    constraint && methods.cone(data.cone, x)
    jacobian && methods.cone_jacobian(data.cone_jacobian, x)
    hessian && methods.cone_hessian(data.cone_hessian, x, z)

    return
end