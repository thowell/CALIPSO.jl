struct Constraint{T}
    evaluate::Any 
    jacobian::Any 
    hessian::Any
    num_state::Int 
    num_action::Int 
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

Constraints{T} = Vector{Constraint{T}} where T

function Constraint(f::Function, num_state::Int, num_action::Int; 
    num_parameter::Int=0,
    indices_inequality=collect(1:0), 
    evaluate_hessian=false)

    #TODO: option to load/save methods
    @variables x[1:num_state], u[1:num_action], w[1:num_parameter]
    evaluate = f(x, u, w)
    jac = Symbolics.sparsejacobian(evaluate, [x; u])
    evaluate_func = eval(Symbolics.build_function(evaluate, x, u, w)[2])
    jac_func = eval(Symbolics.build_function(jac.nzval, x, u, w)[2])
    num_constraint = length(evaluate) 
    num_jacobian = length(jac.nzval)
    jacobian_sparsity = [findnz(jac)[1:2]...]
    if evaluate_hessian
        @variables λ[1:num_constraint]
        lag_con = dot(λ, evaluate) 
        hessian = Symbolics.sparsehessian(lag_con, [x; u])
        hess_func = eval(Symbolics.build_function(hessian.nzval, x, u, w, λ)[2])
        hessian_sparsity = [findnz(hessian)[1:2]...]
        num_hessian = length(hessian.nzval)
    else 
        hess_func = Expr(:null) 
        hessian_sparsity = [Int[]]
        num_hessian = 0
    end
    
    return Constraint(
        evaluate_func, 
        jac_func, 
        hess_func,
        num_state, 
        num_action, 
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

function Constraint()
    return Constraint(
        (constraint, state, action, parameter) -> nothing, 
        (jacobian,   state, action, parameter) -> nothing, 
        (hessian,    state, action, parameter) -> nothing, 
        0, 0, 0, 0, 0, 0, 
        [Int[], Int[]], 
        [Int[], Int[]],
        Float64[], 
        Float64[], 
        Float64[], 
        collect(1:0))
end

function constraints!(violations, indices, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.evaluate(con.evaluate_cache, states[t], actions[t], parameters[t])
        @views violations[indices[t]] .= con.evaluate_cache
        fill!(con.evaluate_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function jacobian!(jacobians, indices, constraints::Constraints{T}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.jacobian(con.jacobian_cache, states[t], actions[t], parameters[t])
        @views jacobians[indices[t]] .= con.jacobian_cache
        fill!(con.jacobian_cache, 0.0) # TODO: confirm this is necessary
    end
end

function hessian_lagrangian!(hessians, indices, constraints::Constraints{T}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints)
        if !isempty(con.hessian_cache)
            con.hessian(con.hessian_cache, states[t], actions[t], parameters[t], duals[t])
            @views hessians[indices[t]] .+= con.hessian_cache
            fill!(con.hessian_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

function sparsity_jacobian(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}; 
    row_shift=0) where T

    row = Int[]
    col = Int[]

    for (t, con) in enumerate(constraints)
        col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
        push!(row, (con.jacobian_sparsity[1] .+ row_shift)...) 
        push!(col, (con.jacobian_sparsity[2] .+ col_shift)...) 
        row_shift += con.num_constraint
    end

    return collect(zip(row, col))
end

function sparsity_hessian(constraints::Constraints{T}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    row = Int[]
    col = Int[]

    for (t, con) in enumerate(constraints)
        if !isempty(con.hessian_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.hessian_sparsity[1] .+ shift)...) 
            push!(col, (con.hessian_sparsity[2] .+ shift)...) 
        end
    end
    
    return collect(zip(row, col))
end

num_constraint(constraints::Constraints{T}) where T = sum([con.num_constraint for con in constraints])
num_jacobian(constraints::Constraints{T}) where T = sum([con.num_jacobian for con in constraints])
# num_hessian(constraints::Constraints{T}) where T = sum([con.num_hessian for con in constraints])

function constraint_indices(constraints::Constraints{T}; 
    shift=0) where T

    indices = Vector{Int}[]

    for (t, con) in enumerate(constraints)
        indices = [indices..., collect(shift .+ (1:con.num_constraint)),]
        shift += con.num_constraint
    end

    return indices
end 

function jacobian_indices(constraints::Constraints{T}; 
    shift=0) where T

    indices = Vector{Int}[]

    for (t, con) in enumerate(constraints) 
        push!(indices, collect(shift .+ (1:con.num_jacobian)))
        shift += con.num_jacobian
    end

    return indices
end

function hessian_indices(constraints::Constraints{T}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.hessian_sparsity[1])
            row = Int[]
            col = Int[]
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.hessian_sparsity[1] .+ shift)...) 
            push!(col, (con.hessian_sparsity[2] .+ shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end
