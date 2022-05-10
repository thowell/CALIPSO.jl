struct EqualityGeneral{T}
    constraint::Function
    jacobian_variables::Function
    jacobian_parameters::Function
    constraint_dual::Function
    constraint_dual_jacobian_variables::Function
    constraint_dual_jacobian_parameters::Function
    constraint_dual_jacobian_variables_variables::Function 
    constraint_dual_jacobian_variables_parameters::Function
    num_variables::Int 
    num_parameters::Int
    num_constraint::Int
    num_jacobian_variables::Int 
    num_jacobian_parameters::Int
    num_jacobian_variables_variables::Int
    num_jacobian_variables_parameters::Int
    jacobian_variables_sparsity::Vector{Vector{Int}}
    jacobian_parameters_sparsity::Vector{Vector{Int}}
    jacobian_variables_variables_sparsity::Vector{Vector{Int}}
    jacobian_variables_parameters_sparsity::Vector{Vector{Int}}
    constraint_cache::Vector{T} 
    jacobian_variables_cache::Vector{T}
    jacobian_parameters_cache::Vector{T}
    constraint_dual_jacobian_variables_cache::Vector{T}
    jacobian_variables_variables_cache::Vector{T}
    jacobian_variables_parameters_cache::Vector{T}
end

function EqualityGeneral(constraint::Function, num_variables::Int; 
    num_parameters::Int=0,
    checkbounds=true,
    constraint_tensor=true)

    #TODO: option to load/save methods
    @variables z[1:num_variables], w[1:num_parameters]

    c = num_parameters > 0 ? constraint(z, w) : constraint(z)
    cz = Symbolics.sparsejacobian(c, z)
    cw = Symbolics.sparsejacobian(c, w)

    num_constraint = length(c) 
    @variables y[1:num_constraint]
    cᵀy = dot(c, y) 
    cᵀyz = Symbolics.gradient(cᵀy, z) 
    cᵀyw = Symbolics.gradient(cᵀy, w) 
  
    c_func = Symbolics.build_function(c, z, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cz_func = Symbolics.build_function(cz.nzval, z, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cw_func = Symbolics.build_function(cw.nzval, z, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cᵀy_func = Symbolics.build_function([cᵀy], z, w, y,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cᵀyz_func = Symbolics.build_function(cᵀyz, z, w, y,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cᵀyw_func = Symbolics.build_function(cᵀyw, z, w, y,
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    num_jacobian_variables = length(cz.nzval)
    num_jacobian_parameters = length(cw.nzval)
    
    jacobian_variables_sparsity = [findnz(cz)[1:2]...]
    jacobian_parameters_sparsity = [findnz(cw)[1:2]...]
   
    # if constraint_tensor
    cᵀyzz = Symbolics.sparsejacobian(constraint_tensor ? cᵀyz : zeros(num_variables), z)
    cᵀyzw = Symbolics.sparsejacobian(constraint_tensor ? cᵀyz : zeros(num_variables), w)
    num_jacobian_variables_variables = length(cᵀyzz.nzval)
    num_jacobian_variables_parameters = length(cᵀyzw.nzval)
    cᵀyzz_func = Symbolics.build_function(cᵀyzz.nzval, z, w, y,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cᵀyzw_func = Symbolics.build_function(cᵀyzw.nzval, z, w, y,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    jacobian_variables_variables_sparsity = [findnz(cᵀyzz)[1:2]...]
    jacobian_variables_parameters_sparsity = [findnz(cᵀyzw)[1:2]...]
    
    return EqualityGeneral(
        c_func, 
        cz_func, 
        cw_func,
        cᵀy_func,
        cᵀyz_func,
        cᵀyw_func,
        cᵀyzz_func,
        cᵀyzw_func, 
        num_variables, 
        num_parameters, 
        num_constraint, 
        num_jacobian_variables,
        num_jacobian_parameters,
        num_jacobian_variables_variables,
        num_jacobian_variables_parameters,
        jacobian_variables_sparsity,
        jacobian_parameters_sparsity,
        jacobian_variables_variables_sparsity,
        jacobian_variables_parameters_sparsity,
        zeros(num_constraint), 
        zeros(num_jacobian_variables), 
        zeros(num_jacobian_parameters), 
        zeros(num_variables),
        zeros(num_jacobian_variables_variables),
        zeros(num_jacobian_variables_parameters),
    )
end

function EqualityGeneral()
    return EqualityGeneral(
        (constraint, variables, parameters) -> nothing, 
        (jacobian,   variables, parameters) -> nothing, 
        (jacobian,   variables, parameters) -> nothing, 
        (constraint_dual, variables, parameters, duals) -> nothing, 
        (jacobian,    variables, parameters, duals) -> nothing, 
        (jacobian,    variables, parameters, duals) -> nothing, 
        (jacobian,    variables, parameters, duals) -> nothing, 
        (jacobian,    variables, parameters, duals) -> nothing, 
        0, 0, 0, 0, 0, 0, 0,
        [Int[], Int[]], 
        [Int[], Int[]],
        [Int[], Int[]], 
        [Int[], Int[]],
        Float64[], 
        Float64[], 
        Float64[],
        Float64[], 
        Float64[], 
        Float64[], 
    )
end

function constraints!(violations, indices, constraint::EqualityGeneral{T}, variables, parameters) where T
    if constraint.num_constraint > 0
        constraint.constraint(constraint.constraint_cache, variables, parameters)
        @views violations[indices] .= constraint.constraint_cache
    end
end

function jacobian_variables!(jacobians, sparsity, constraint::EqualityGeneral{T}, variables, parameters) where T
    count = 1
    if !isempty(constraint.jacobian_variables_cache) 
        constraint.jacobian_variables(constraint.jacobian_variables_cache, variables, parameters)
        for v in constraint.jacobian_variables_cache 
            jacobians[sparsity + count] = v
            count += 1
        end
    end
end

function jacobian_parameters!(jacobians, sparsity, constraint::EqualityGeneral{T}, variables, parameters) where T
    count = 1
    if !isempty(constraint.jacobian_parameters_cache) 
        constraint.jacobian_parameters(constraint.jacobian_parameters_cache, variables, parameters)
        for v in constraint.jacobian_parameters_cache
            jacobians[sparsity + count] = v 
            count += 1
        end
    end
end

function constraint_dual_jacobian_variables!(gradient, constraint::EqualityGeneral{T}, variables, parameters, duals) where T
    if !isempty(constraint.constraint_dual_jacobian_variables_cache) 
        constraint.constraint_dual_jacobian_variables(constraint.constraint_dual_jacobian_variables_cache, variables, parameters, duals)
        n = length(gradient)
        for i = 1:n
            gradient[i] += constraint.constraint_dual_jacobian_variables_cache[i] 
        end
    end
end

function jacobian_variables_variables!(jacobians, sparsity, constraint::EqualityGeneral{T}, variables, parameters, duals) where T
    count = 1
    if !isempty(constraint.jacobian_variables_variables_cache)
        constraint.constraint_dual_jacobian_variables_variables(constraint.jacobian_variables_variables_cache, variables, parameters, duals)
        for v in constraint.jacobian_variables_variables_cache
            jacobians[sparsity + count] += v
            count += 1
        end
    end
end

function jacobian_variables_parameters!(jacobians, sparsity, constraint::EqualityGeneral{T}, variables, parameters, duals) where T
    count = 1
    if !isempty(constraint.jacobian_variables_parameters_cache)
        constraint.constraint_dual_jacobian_variables_parameters(constraint.jacobian_variables_parameters_cache, variables, parameters, duals)
        for v in constraint.jacobian_variables_parameters_cache
            jacobians[sparsity + count] += v
            count += 1
        end
    end
end

function sparsity_jacobian_variables(constraint::EqualityGeneral{T}, num_variables::Int; 
    row_shift=0) where T

    row = Int[]
    col = Int[]

    col_shift = 0
    push!(row, (constraint.jacobian_variables_sparsity[1] .+ row_shift)...) 
    push!(col, (constraint.jacobian_variables_sparsity[2] .+ col_shift)...) 

    return collect(zip(row, col))
end

function sparsity_jacobian_parameters(constraint::EqualityGeneral{T}, num_variables::Int, num_parameters::Int; 
    row_shift=0) where T

    row = Int[]
    col = Int[]

    col_shift = 0
    push!(row, (constraint.jacobian_parameters_sparsity[1] .+ row_shift)...) 
    push!(col, (constraint.jacobian_parameters_sparsity[2] .+ col_shift)...) 

    return collect(zip(row, col))
end

function sparsity_jacobian_variables_variables(constraint::EqualityGeneral{T}, num_variables::Int) where T

    row = Int[]
    col = Int[]
    if !isempty(constraint.jacobian_variables_variables_sparsity[1])
        shift = 0
        push!(row, (constraint.jacobian_variables_variables_sparsity[1] .+ shift)...) 
        push!(col, (constraint.jacobian_variables_variables_sparsity[2] .+ shift)...) 
    end

    return collect(zip(row, col))
end

function sparsity_jacobian_variables_parameters(constraint::EqualityGeneral{T}, num_variables::Int, num_parameters::Int) where T

    row = Int[]
    col = Int[]
    
    if !isempty(constraint.jacobian_variables_parameters_sparsity[1])
        row_shift = 0
        col_shift = 0
        push!(row, (constraint.jacobian_variables_parameters_sparsity[1] .+ row_shift)...) 
        push!(col, (constraint.jacobian_variables_parameters_sparsity[2] .+ col_shift)...) 
    end
    
    return collect(zip(row, col))
end

num_constraint(constraint::EqualityGeneral{T}) where T = constraint.num_constraint
num_jacobian_variables(constraint::EqualityGeneral{T}) where T = constraint.num_jacobian_variables
num_jacobian_parameters(constraint::EqualityGeneral{T}) where T = constraint.num_jacobian_parameters

function constraint_indices(constraint::EqualityGeneral{T}; 
    shift=0) where T

    indices = Int[]

    indices = [indices..., collect(shift .+ (1:constraint.num_constraint))...,]
    shift += constraint.num_constraint

    return indices
end 

function jacobian_variables_indices(constraint::EqualityGeneral{T}; 
    shift=0) where T

    indices = Int[]

    indices = [indices..., collect(shift .+ (1:constraint.num_jacobian_variables))...]
    shift += constraint.num_jacobian_variables

    return indices
end

function jacobian_parameters_indices(constraint::EqualityGeneral{T}; 
    shift=0) where T

    indices = Int[]

    indices = [indices..., collect(shift .+ (1:constraint.num_jacobian_parameters))...]
    shift += constraint.num_jacobian_parameters

    return indices
end

function jacobian_variables_variables_indices(constraint::EqualityGeneral{T}, key::Vector{Tuple{Int,Int}}, num_variables::Int) where T
    indices = Int[]
    if !isempty(constraint.jacobian_variables_variables_sparsity[1])
        row = Int[]
        col = Int[]
        shift = 0
        push!(row, (constraint.jacobian_variables_variables_sparsity[1] .+ shift)...) 
        push!(col, (constraint.jacobian_variables_variables_sparsity[2] .+ shift)...) 
        rc = collect(zip(row, col))
        indices = [indices..., [findfirst(x -> x == i, key) for i in rc]...]
    end
    return indices
end

function jacobian_variables_parameters_indices(constraint::EqualityGeneral{T}, key::Vector{Tuple{Int,Int}}, num_variables::Int, num_parameters::Int) where T
    indices = Int[]
    if !isempty(constraint.jacobian_variables_parameters_sparsity[1])
        row = Int[]
        col = Int[]
        row_shift = 0
        col_shift = 0
        push!(row, (constraint.jacobian_variables_parameters_sparsity[1] .+ row_shift)...) 
        push!(col, (constraint.jacobian_variables_parameters_sparsity[2] .+ col_shift)...) 
        rc = collect(zip(row, col))
        indices = [indices..., [findfirst(x -> x == i, key) for i in rc]...]
    end
    return indices
end

