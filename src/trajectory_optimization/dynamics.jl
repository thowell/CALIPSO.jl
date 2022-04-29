struct Dynamics{T}
    constraint::Function
    jacobian_variables::Function
    jacobian_parameters::Function
    constraint_dual::Function
    constraint_dual_jacobian_variables::Function
    constraint_dual_jacobian_parameters::Function
    constraint_dual_jacobian_variables_variables::Function
    constraint_dual_jacobian_variables_parameters::Function
    num_next_state::Int 
    num_state::Int 
    num_action::Int
    num_parameter::Int
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

function Dynamics(dynamics::Function, num_next_state::Int, num_state::Int, num_action::Int; 
    num_parameter::Int=0, 
    evaluate_hessian=true)

    #TODO: option to load/save methods
    @variables y[1:num_next_state], x[1:num_state], u[1:num_action], w[1:num_parameter] 
    d = num_parameter > 0 ? dynamics(y, x, u, w) : dynamics(y, x, u)
    dz = Symbolics.sparsejacobian(d, [x; u; y]);
    dw = Symbolics.sparsejacobian(d, w);

    d_func = Symbolics.build_function(d, y, x, u, w, expression=Val{false})[2];
    dz_func = Symbolics.build_function(dz.nzval, y, x, u, w, expression=Val{false})[2];
    dw_func = Symbolics.build_function(dw.nzval, y, x, u, w, expression=Val{false})[2];

    num_jacobian_variables = length(dz.nzval)
    num_jacobian_parameters = length(dw.nzval)

    jacobian_variables_sparsity = [findnz(dz)[1:2]...]
    jacobian_parameters_sparsity = [findnz(dw)[1:2]...]

    @variables λ[1:length(d)]
    dᵀλ = dot(d, λ) 
    dᵀλz = Symbolics.gradient(dᵀλ, [x; u; y])
    dᵀλw = Symbolics.gradient(dᵀλ, w)

    dᵀλ_func = Symbolics.build_function([dᵀλ], y, x, u, w, λ, expression=Val{false})[2]
    dᵀλz_func = Symbolics.build_function(dᵀλz, y, x, u, w, λ, expression=Val{false})[2]
    dᵀλw_func = Symbolics.build_function(dᵀλw, y, x, u, w, λ, expression=Val{false})[2]

    if evaluate_hessian
        dᵀλzz = Symbolics.sparsejacobian(dᵀλz, [x; u; y])
        dᵀλzw = Symbolics.sparsejacobian(dᵀλz, w)

        dᵀλzz_func = Symbolics.build_function(dᵀλzz.nzval, y, x, u, w, λ, expression=Val{false})[2]
        dᵀλzw_func = Symbolics.build_function(dᵀλzw.nzval, y, x, u, w, λ, expression=Val{false})[2]

        jacobian_variables_variables_sparsity = [findnz(dᵀλzz)[1:2]...]
        jacobian_variables_parameters_sparsity = [findnz(dᵀλzw)[1:2]...]

        num_jacobian_variables_variables = length(dᵀλzz.nzval)
        num_jacobian_variables_parameters = length(dᵀλzw.nzval)
    else 
        dᵀλzz_func = Expr(:null) 
        dᵀλzw_func = Expr(:null) 

        jacobian_variables_variables_sparsity = [Int[]]
        jacobian_variables_parameters_sparsity = [Int[]]

        num_jacobian_variables_variables = 0
        num_jacobian_variables_parameters = 0
    end
  
    return Dynamics(
        d_func, 
        dz_func, 
        dw_func,
        dᵀλ_func,
        dᵀλz_func,
        dᵀλw_func,
        dᵀλzz_func, 
        dᵀλzw_func,
        num_next_state, 
        num_state, 
        num_action, 
        num_parameter, 
        num_jacobian_variables,
        num_jacobian_parameters,
        num_jacobian_variables_variables,
        num_jacobian_variables_parameters,
        jacobian_variables_sparsity, 
        jacobian_parameters_sparsity, 
        jacobian_variables_variables_sparsity, 
        jacobian_variables_parameters_sparsity, 
        zeros(num_next_state), 
        zeros(num_jacobian_variables), 
        zeros(num_jacobian_parameters), 
        zeros(num_state + num_action + num_next_state),
        zeros(num_jacobian_variables_variables),
        zeros(num_jacobian_variables_parameters),
    )
end

function constraints!(violations, indices, constraints::Vector{Dynamics{T}}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.constraint(con.constraint_cache, states[t+1], states[t], actions[t], parameters[t])
        @views violations[indices[t]] .= con.constraint_cache
    end
end

function jacobian_variables!(jacobians, sparsity, constraints::Vector{Dynamics{T}}, states, actions, parameters) where T
    count = 1
    for (t, con) in enumerate(constraints) 
        con.jacobian_variables(con.jacobian_variables_cache, states[t+1], states[t], actions[t], parameters[t])
        for v in con.jacobian_variables_cache
            jacobians[sparsity + count] = v
            count += 1
        end
    end
end

function jacobian_parameters!(jacobians, sparsity, constraints::Vector{Dynamics{T}}, states, actions, parameters) where T
    count = 1
    for (t, con) in enumerate(constraints) 
        con.jacobian_parameters(con.jacobian_parameters_cache, states[t+1], states[t], actions[t], parameters[t])
        for v in con.jacobian_parameters_cache 
            jacobians[sparsity + count] = v
            count += 1
        end
    end
end

function constraint_dual_jacobian_variables!(gradient, indices, constraints::Vector{Dynamics{T}}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints) 
        con.constraint_dual_jacobian_variables(con.constraint_dual_jacobian_variables_cache, states[t+1], states[t], actions[t], parameters[t], duals[t])
        for (i, idx) in enumerate(indices[t]) 
            gradient[idx...] += con.constraint_dual_jacobian_variables_cache[i]
        end
    end
end

function jacobian_variables_variables!(jacobians, sparsity, constraints::Vector{Dynamics{T}}, states, actions, parameters, duals) where T
    count = 1
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_variables_cache)
            con.constraint_dual_jacobian_variables_variables(con.jacobian_variables_variables_cache, states[t+1], states[t], actions[t], parameters[t], duals[t])
            for v in con.jacobian_variables_variables_cache
                jacobians[sparsity + count] += v
                count += 1
            end
        end
    end
end

function jacobian_variables_parameters!(jacobians, sparsity, constraints::Vector{Dynamics{T}}, states, actions, parameters, duals) where T
    count = 1
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_parameters_cache)
            con.constraint_dual_jacobian_variables_parameters(con.jacobian_variables_parameters_cache, states[t+1], states[t], actions[t], parameters[t], duals[t])
            for v in con.jacobian_variables_parameters_cache
                jacobians[sparsity + count] += v
                count += 1
            end
        end
    end
end

function sparsity_jacobian_variables(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}; 
    row_shift=0) where T

    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints) 
        row = Int[]
        col = Int[]
        col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_actions[1:t-1])) : 0)
        push!(row, (con.jacobian_variables_sparsity[1] .+ row_shift)...) 
        push!(col, (con.jacobian_variables_sparsity[2] .+ col_shift)...) 
        s = collect(zip(row, col))
        push!(sp, s)
        row_shift += con.num_next_state
    end

    return sp
end

function sparsity_jacobian_parameters(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}, num_parameter::Vector{Int}; 
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
        row_shift += con.num_next_state
    end

    return sp
end

function sparsity_jacobian_variables_variables(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints) 
        row = Int[]
        col = Int[]
        if !isempty(con.jacobian_variables_variables_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_actions[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_variables_sparsity[1] .+ shift)...) 
            push!(col, (con.jacobian_variables_variables_sparsity[2] .+ shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    return sp
end

function sparsity_jacobian_variables_parameters(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}, num_parameter::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]

    for (t, con) in enumerate(constraints) 
        row = Int[]
        col = Int[]
        if !isempty(con.jacobian_variables_parameters_sparsity[1])
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_actions[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_parameters_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_variables_parameters_sparsity[2] .+ col_shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    return sp
end


num_state_action_next_state(constraints::Vector{Dynamics{T}}) where T = sum([con.num_state + con.num_action for con in constraints]) + constraints[end].num_next_state
num_constraint(constraints::Vector{Dynamics{T}}) where T = sum([con.num_next_state for con in constraints])
num_jacobian_variables(constraints::Vector{Dynamics{T}}) where T = sum([con.num_jacobian_variables for con in constraints])
num_jacobian_parameters(constraints::Vector{Dynamics{T}}) where T = sum([con.num_jacobian_parameters for con in constraints])

# num_hessian(constraints::Vector{Dynamics{T}}) where T = sum([con.num_hessian for con in constraints])

function constraint_indices(constraints::Vector{Dynamics{T}}; 
    shift=0) where T
    [collect(shift + (t > 1 ? sum([constraints[s].num_next_state for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_next_state)) for t = 1:length(constraints)]
end 

function jacobian_variables_indices(constraints::Vector{Dynamics{T}}; 
    shift=0) where T
    [collect(shift + (t > 1 ? sum([constraints[s].num_jacobian_variables for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_jacobian_variables)) for t = 1:length(constraints)]
end

function jacobian_parameters_indices(constraints::Vector{Dynamics{T}}; 
    shift=0) where T
    [collect(shift + (t > 1 ? sum([constraints[s].num_jacobian_parameters for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_jacobian_parameters)) for t = 1:length(constraints)]
end

function jacobian_variables_variables_indices(constraints::Vector{Dynamics{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_variables_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            row = collect(con.jacobian_variables_variables_sparsity[1] .+ shift)
            col = collect(con.jacobian_variables_variables_sparsity[2] .+ shift)
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end

function jacobian_variables_parameters_indices(constraints::Vector{Dynamics{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameter::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.jacobian_variables_parameters_sparsity[1])
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            row = collect(con.jacobian_variables_parameters_sparsity[1] .+ row_shift)
            col = collect(con.jacobian_variables_parameters_sparsity[2] .+ col_shift)
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end


function state_indices(constraints::Vector{Dynamics{T}}) where T 
    [[collect((t > 1 ? sum([constraints[s].num_state + constraints[s].num_action for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_state)) for t = 1:length(constraints)]..., 
        collect(sum([constraints[s].num_state + constraints[s].num_action for s = 1:length(constraints)]) .+ (1:constraints[end].num_next_state))]
end

function action_indices(constraints::Vector{Dynamics{T}}) where T 
    [collect((t > 1 ? sum([constraints[s].num_state + constraints[s].num_action for s = 1:(t-1)]) : 0) + constraints[t].num_state .+ (1:constraints[t].num_action)) for t = 1:length(constraints)]
end

function state_action_indices(constraints::Vector{Dynamics{T}}) where T 
    [[collect((t > 1 ? sum([constraints[s].num_state + constraints[s].num_action for s = 1:(t-1)]) : 0) .+ (1:(+ constraints[t].num_state + constraints[t].num_action))) for t = 1:length(constraints)]..., 
        collect(sum([constraints[s].num_state + constraints[s].num_action for s = 1:length(constraints)]) .+ (1:constraints[end].num_next_state))]
end

function state_action_next_state_indices(constraints::Vector{Dynamics{T}}) where T 
    [collect((t > 1 ? sum([constraints[s].num_state + constraints[s].num_action for s = 1:(t-1)]) : 0) .+ (1:(+ constraints[t].num_state + constraints[t].num_action + constraints[t].num_next_state))) for t = 1:length(constraints)]
end

function dimensions(dynamics::Vector{Dynamics{T}}; 
    parameters=[0 for t = 1:(length(dynamics) + 1)]) where T 
    states = [[d.num_state for d in dynamics]..., dynamics[end].num_next_state]
    actions = [[d.num_action for d in dynamics]..., 0]
    return states, actions, parameters
end