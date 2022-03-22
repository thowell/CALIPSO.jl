struct GeneralConstraint{T}
    evaluate 
    jacobian 
    hessian
    num_variables::Int 
    num_parameter::Int
    num_constraint::Int
    num_jacobian::Int 
    num_hessian::Int
    jacobian_sparsity::Vector{Vector{Int}}
    hessian_sparsity::Vector{Vector{Int}}
    evaluate_cache::Vector{T} 
    jacobian_cache::Vector{T}
    hessian_cache::Vector{T}
    indices_inequality::Vector{Int}
end

function GeneralConstraint(f::Function, num_variables::Int, num_parameter::Int; 
    indices_inequality=collect(1:0), 
    evaluate_hessian=false)

    #TODO: option to load/save methods
    @variables z[1:num_variables], w[1:num_parameter]
    evaluate = f(z, w)
    jacobian = Symbolics.sparsejacobian(evaluate, z)
    evaluate_func = eval(Symbolics.build_function(evaluate, z, w)[2])
    jacobian_func = eval(Symbolics.build_function(jacobian.nzval, z, w)[2])
    num_constraint = length(evaluate) 
    num_jacobian = length(jacobian.nzval)
    jacobian_sparsity = [findnz(jacobian)[1:2]...]
    if evaluate_hessian
        @variables λ[1:num_constraint]
        lag_con = dot(λ, evaluate) 
        hessian = Symbolics.sparsehessian(lag_con, z)
        hessian_func = eval(Symbolics.build_function(hessian.nzval, z, w, λ)[2])
        hessian_sparsity = [findnz(hessian)[1:2]...]
        num_hessian = length(hessian.nzval)
    else 
        hessian_func = Expr(:null) 
        hessian_sparsity = [Int[]]
        num_hessian = 0
    end

    return GeneralConstraint(
        evaluate_func, 
        jacobian_func, 
        hessian_func,
        num_variables, 
        num_parameter, 
        num_constraint, 
        num_jacobian, 
        num_hessian, 
        jacobian_sparsity, 
        hessian_sparsity, 
        zeros(num_constraint), 
        zeros(num_jacobian), 
        zeros(num_hessian), 
        indices_inequality)
end

function GeneralConstraint()
    return GeneralConstraint(
        (constraint, variables, parameters) -> nothing, 
        (jacobian,   variables, parameters) -> nothing, 
        (hessian,    variables, parameters) -> nothing, 
        0, 0, 0, 0, 0, 
        [Int[], Int[]], 
        [Int[], Int[]], 
        Float64[], Float64[], Float64[],
        collect(1:0))
end

function constraints!(violations, indices, general::GeneralConstraint{T}, variables, parameters) where T
    general.evaluate(general.evaluate_cache, variables, parameters)
    @views violations[indices] .= general.evaluate_cache
    fill!(general.evaluate_cache, 0.0) # TODO: confirm this is necessary 
end

function jacobian!(jacobians, indices, general::GeneralConstraint{T}, variables, parameters) where T
    general.jacobian(general.jacobian_cache, variables, parameters)
    @views jacobians[indices] .= general.jacobian_cache
    fill!(general.jacobian_cache, 0.0) # TODO: confirm this is necessary
end

function hessian_lagrangian!(hessian, indices, general::GeneralConstraint{T}, variables, parameters, duals) where T
    if !isempty(general.hessian_cache)
        general.hessian(general.hessian_cache, variables, duals)
        @views hessian[indices] .+= general.hessian_cache
        fill!(general.hessian_cache, 0.0) # TODO: confirm this is necessary
    end
end

function sparsity_jacobian(general::GeneralConstraint{T}, num_variables::Int; 
    row_shift=0) where T

    row = Int[]
    col = Int[]
   
    push!(row, (general.jacobian_sparsity[1] .+ row_shift)...) 
    push!(col,  general.jacobian_sparsity[2]...) 
    
    return collect(zip(row, col))
end

function sparsity_hessian(general::GeneralConstraint{T}, num_variables::Int) where T
    row = Int[]
    col = Int[]

    if !isempty(general.hessian_sparsity[1])
        shift = 0
        push!(row, (general.hessian_sparsity[1] .+ shift)...) 
        push!(col, (general.hessian_sparsity[2] .+ shift)...) 
    end

    return collect(zip(row, col))
end

num_constraint(general::GeneralConstraint{T}) where T = general.num_constraint
num_jacobian(general::GeneralConstraint{T}) where T = general.num_jacobian

constraint_indices(general::GeneralConstraint{T}; shift=0) where T = collect(shift .+ (1:general.num_constraint))

jacobian_indices(general::GeneralConstraint{T}; shift=0) where T = collect(shift .+ (1:general.num_jacobian))
   
function hessian_indices(general::GeneralConstraint{T}, key::Vector{Tuple{Int,Int}}, num_variables::Int) where T
    if !isempty(general.hessian_sparsity[1])
        row = Int[]
        col = Int[]
        shift = 0
        push!(row, (general.hessian_sparsity[1] .+ shift)...) 
        push!(col, (general.hessian_sparsity[2] .+ shift)...) 
        rc = collect(zip(row, col))
        indices = [findfirst(x -> x == i, key) for i in rc]
    else 
        indices = Vector{Int}[]
    end
    return indices
end
