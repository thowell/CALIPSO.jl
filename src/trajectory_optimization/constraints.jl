struct Constraint{T}
    constraint::Function
    jacobian_variables::Function
    jacobian_parameters::Function
    constraint_dual::Function
    constraint_dual_jacobian_variables::Function
    constraint_dual_jacobian_parameters::Function
    constraint_dual_jacobian_variables_variables::Function 
    constraint_dual_jacobian_variables_parameters::Function
    num_state::Int 
    num_action::Int 
    num_parameter::Int
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

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(constraint::Function, num_state::Int, num_action::Int; 
    num_parameter::Int=0,
    evaluate_hessian=true)

    #TODO: option to load/save methods
    @variables x[1:num_state], u[1:num_action], w[1:num_parameter]

    c = constraint(x, u, w)
    cz = Symbolics.sparsejacobian(c, [x; u])
    cw = Symbolics.sparsejacobian(c, w)

    num_constraint = length(c) 
    @variables y[1:num_constraint]
    cᵀy = dot(c, y) 
    cᵀyz = Symbolics.gradient(cᵀy, [x; u]) 
    cᵀyw = Symbolics.gradient(cᵀy, w) 
  
    c_func = Symbolics.build_function(c, x, u, w, expression=Val{false})[2]
    cz_func = Symbolics.build_function(cz.nzval, x, u, w, expression=Val{false})[2]
    cw_func = Symbolics.build_function(cw.nzval, x, u, w, expression=Val{false})[2]
    cᵀy_func = Symbolics.build_function([cᵀy], x, u, w, y, expression=Val{false})[2]
    cᵀyz_func = Symbolics.build_function(cᵀyz, x, u, w, y, expression=Val{false})[2]
    cᵀyw_func = Symbolics.build_function(cᵀyw, x, u, w, y, expression=Val{false})[2]

    num_jacobian_variables = length(cz.nzval)
    num_jacobian_parameters = length(cw.nzval)
    
    jacobian_variables_sparsity = [findnz(cz)[1:2]...]
    jacobian_parameters_sparsity = [findnz(cw)[1:2]...]
   
    if evaluate_hessian
        cᵀyzz = Symbolics.sparsejacobian(cᵀyz, [x; u])
        cᵀyzw = Symbolics.sparsejacobian(cᵀyz, w)
        num_jacobian_variables_variables = length(cᵀyzz.nzval)
        num_jacobian_variables_parameters = length(cᵀyzw.nzval)
        cᵀyzz_func = Symbolics.build_function(cᵀyzz.nzval, x, u, w, y, expression=Val{false})[2]
        cᵀyzw_func = Symbolics.build_function(cᵀyzw.nzval, x, u, w, y, expression=Val{false})[2]
        jacobian_variables_variables_sparsity = [findnz(cᵀyzz)[1:2]...]
        jacobian_variables_parameters_sparsity = [findnz(cᵀyzw)[1:2]...]
    else      
        num_jacobian_variables_variables = 0
        num_jacobian_variables_parameters = 0
        cᵀyzz_func = Expr(:null) 
        cᵀyzw_func = Expr(:null) 
        jacobian_variables_variables_sparsity = [Int[]]
        jacobian_variables_parameters_sparsity = [Int[]]
    end
    
    return Constraint(
        c_func, 
        cz_func, 
        cw_func,
        cᵀy_func,
        cᵀyz_func,
        cᵀyw_func,
        cᵀyzz_func,
        cᵀyzw_func, 
        num_state, 
        num_action, 
        num_parameter, 
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
        zeros(num_state + num_action),
        zeros(num_jacobian_variables_variables),
        zeros(num_jacobian_variables_parameters),
    )
end

function Constraint()
    return Constraint(
        (constraint, state, action, parameter) -> nothing, 
        (jacobian,   state, action, parameter) -> nothing, 
        (jacobian,   state, action, parameter) -> nothing, 
        (constraint_dual, state, action, parameter, dual) -> nothing, 
        (jacobian,    state, action, parameter, dual) -> nothing, 
        (jacobian,    state, action, parameter, dual) -> nothing, 
        (jacobian,    state, action, parameter, dual) -> nothing, 
        (jacobian,    state, action, parameter, dual) -> nothing, 
        0, 0, 0, 0, 0, 0, 0, 0,
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

function constraints!(violations, indices, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        if con.num_constraint > 0
            con.constraint(con.constraint_cache, states[t], actions[t], parameters[t])
            @views violations[indices[t]] .= con.constraint_cache
        end
    end
end

function jacobian_variables!(jacobians, sparsity, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.jacobian_variables_cache) 
            con.jacobian_variables(con.jacobian_variables_cache, states[t], actions[t], parameters[t])
            for (i, idx) in enumerate(sparsity[t]) 
                jacobians[idx...] = con.jacobian_variables_cache[i] 
            end
        end
    end
end

function jacobian_parameters!(jacobians, sparsity, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.jacobian_parameters_cache) 
            con.jacobian_parameters(con.jacobian_parameters_cache, states[t], actions[t], parameters[t])
            for (i, idx) in enumerate(sparsity[t]) 
                jacobians[idx...] = con.jacobian_parameters_cache[i] 
            end
        end
    end
end

function constraint_dual_jacobian_variables!(gradient, indices, constraints::Constraints{T}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.constraint_dual_jacobian_variables_cache) 
            con.constraint_dual_jacobian_variables(con.constraint_dual_jacobian_variables_cache, states[t], actions[t], parameters[t], duals[t])
            for (i, idx) in enumerate(indices[t]) 
                gradient[idx...] += con.constraint_dual_jacobian_variables_cache[i] 
            end
        end
    end
end

function jacobian_variables_variables!(jacobians, sparsity, constraints::Constraints{T}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.jacobian_variables_variables_cache)
            con.constraint_dual_jacobian_variables_variables(con.jacobian_variables_variables_cache, states[t], actions[t], parameters[t], duals[t])
            for (i, idx) in enumerate(sparsity[t]) 
                jacobians[idx...] += con.jacobian_variables_variables_cache[i]
            end
        end
    end
end

function jacobian_variables_parameters!(jacobians, sparsity, constraints::Constraints{T}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.jacobian_variables_parameters_cache)
            con.constraint_dual_jacobian_variables_parameters(con.jacobian_variables_parameters_cache, states[t], actions[t], parameters[t], duals[t])
            for (i, idx) in enumerate(sparsity[t]) 
                jacobians[idx...] += con.jacobian_variables_parameters_cache[i]
            end
        end
    end
end

function sparsity_jacobian_variables(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}; 
    row_shift=0) where T

    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints)
        row = Int[]
        col = Int[]

        col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
        push!(row, (con.jacobian_variables_sparsity[1] .+ row_shift)...) 
        push!(col, (con.jacobian_variables_sparsity[2] .+ col_shift)...) 

        s = collect(zip(row, col))
        push!(sp, s)

        row_shift += con.num_constraint
    end

    return sp
end

function sparsity_jacobian_parameters(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameter::Vector{Int}; 
    row_shift=0) where T

    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints)
        row = Int[]
        col = Int[]

        col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
        push!(row, (con.jacobian_parameters_sparsity[1] .+ row_shift)...) 
        push!(col, (con.jacobian_parameters_sparsity[2] .+ col_shift)...) 

        s = collect(zip(row, col))
        push!(sp, s)

        row_shift += con.num_constraint
    end

    return sp
end

function sparsity_jacobian_variables_variables(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints)
        row = Int[]
        col = Int[]
        if !isempty(con.jacobian_variables_variables_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_variables_sparsity[1] .+ shift)...) 
            push!(col, (con.jacobian_variables_variables_sparsity[2] .+ shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    
    return sp
end

function sparsity_jacobian_variables_parameters(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameter::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints)
        row = Int[]
        col = Int[]
        if !isempty(con.jacobian_variables_parameters_sparsity[1])
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_parameters_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_variables_parameters_sparsity[2] .+ col_shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    
    return sp
end

num_constraint(constraints::Constraints{T}) where T = sum([con.num_constraint for con in constraints])
num_jacobian_variables(constraints::Constraints{T}) where T = sum([con.num_jacobian_variables for con in constraints])
num_jacobian_parameters(constraints::Constraints{T}) where T = sum([con.num_jacobian_parameters for con in constraints])

function constraint_indices(constraints::Constraints{T}; 
    shift=0) where T

    indices = Vector{Int}[]

    for (t, con) in enumerate(constraints)
        indices = [indices..., collect(shift .+ (1:con.num_constraint)),]
        shift += con.num_constraint
    end

    return indices
end 

function jacobian_variables_indices(constraints::Constraints{T}; 
    shift=0) where T

    indices = Vector{Int}[]

    for (t, con) in enumerate(constraints) 
        push!(indices, collect(shift .+ (1:con.num_jacobian_variables)))
        shift += con.num_jacobian_variables
    end

    return indices
end

function jacobian_parameters_indices(constraints::Constraints{T}; 
    shift=0) where T

    indices = Vector{Int}[]

    for (t, con) in enumerate(constraints) 
        push!(indices, collect(shift .+ (1:con.num_jacobian_parameters)))
        shift += con.num_jacobian_parameters
    end

    return indices
end

function jacobian_variables_variables_indices(constraints::Constraints{T}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_variables_sparsity[1])
            row = Int[]
            col = Int[]
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_variables_sparsity[1] .+ shift)...) 
            push!(col, (con.jacobian_variables_variables_sparsity[2] .+ shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end

function jacobian_variables_parameters_indices(constraints::Constraints{T}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameters::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_parameters_sparsity[1])
            row = Int[]
            col = Int[]
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_parameters_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_variables_parameters_sparsity[2] .+ col_shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end

