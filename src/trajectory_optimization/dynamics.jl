struct Dynamics{T}
    evaluate::Any
    jacobian::Any
    hessian::Any
    num_next_state::Int 
    num_state::Int 
    num_action::Int
    num_parameter::Int
    num_jacobian::Int
    num_hessian::Int
    jacobian_sparsity::Vector{Vector{Int}}
    hessian_sparsity::Vector{Vector{Int}}
    evaluate_cache::Vector{T} 
    jacobian_cache::Vector{T}
    hessian_cache::Vector{T}
end

function Dynamics(f::Function, num_next_state::Int, num_state::Int, num_action::Int; 
    num_parameter::Int=0, 
    evaluate_hessian=false)

    #TODO: option to load/save methods
    @variables y[1:num_next_state], x[1:num_state], u[1:num_action], w[1:num_parameter] 
    evaluate = f(y, x, u, w) 
    jac = Symbolics.sparsejacobian(evaluate, [x; u; y]);
    evaluate_func = eval(Symbolics.build_function(evaluate, y, x, u, w)[2]);
    jacobian_func = eval(Symbolics.build_function(jac.nzval, y, x, u, w)[2]);
    num_jacobian = length(jac.nzval)
    jacobian_sparsity = [findnz(jac)[1:2]...]
    if evaluate_hessian
        @variables λ[1:num_next_state] 
        lag_con = dot(λ, evaluate)
        hessian = Symbolics.sparsehessian(lag_con, [x; u; y])
        hessian_func = eval(Symbolics.build_function(hessian.nzval, y, x, u, w, λ)[2])
        hessian_sparsity = [findnz(hessian)[1:2]...]
        num_hessian = length(hessian.nzval)
    else 
        hessian_func = Expr(:null) 
        hessian_sparsity = [Int[]]
        num_hessian = 0
    end
  
    return Dynamics(evaluate_func, 
        jacobian_func, 
        hessian_func, 
        num_next_state, 
        num_state, 
        num_action, 
        num_parameter, 
        num_jacobian,
        num_hessian,
        jacobian_sparsity, 
        hessian_sparsity, 
        zeros(num_next_state), 
        zeros(num_jacobian), 
        zeros(num_hessian))
end

function Dynamics(constraint::Function, constraint_jacobian::Function, num_next_state::Int, num_state::Int, num_action::Int; 
    num_parameter::Int=0)  

    # jacobian function 
    num_variables = num_state + num_action + num_next_state
    jacobian_func = (J, y, x, u, w) -> constraint_jacobian(reshape(view(J, :), num_next_state, num_variables), y, x, u, w)

    # number of Jacobian elements
    num_jacobian = num_next_state * num_variables

    # Jacobian sparsity
    row = Int[]
    col = Int[]
    for j = 1:num_variables
        for i = 1:num_next_state 
            push!(row, i) 
            push!(col, j)
        end
    end

    jacobian_sparsity = [row, col]
  
    # Hessian
    hessian_func = Expr(:null) 
    hessian_sparsity = [Int[]]
    num_hessian = 0
  
    return Dynamics(
        constraint, 
        jacobian_func, 
        hessian_func, 
        num_next_state, 
        num_state, 
        num_action, 
        num_parameter, 
        num_jacobian, 
        num_hessian,
        jacobian_sparsity, 
        hessian_sparsity, 
        zeros(num_next_state), 
        zeros(num_jacobian), 
        zeros(num_hessian))
end

function constraints!(violations, indices, constraints::Vector{Dynamics{T}}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints)
        con.evaluate(con.evaluate_cache, states[t+1], states[t], actions[t], parameters[t])
        @views violations[indices[t]] .= con.evaluate_cache
        fill!(con.evaluate_cache, 0.0) # TODO: confirm this is necessary 
    end
end

function jacobian!(jacobians, indices, constraints::Vector{Dynamics{T}}, states, actions, parameters) where T
    for (t, con) in enumerate(constraints) 
        con.jacobian(con.jacobian_cache, states[t+1], states[t], actions[t], parameters[t])
        @views jacobians[indices[t]] .= con.jacobian_cache
        fill!(con.jacobian_cache, 0.0) # TODO: confirm this is necessary
    end
end

function hessian_lagrangian!(hessians, indices, constraints::Vector{Dynamics{T}}, states, actions, parameters, duals) where T
    for (t, con) in enumerate(constraints) 
        if !isempty(con.hessian_cache)
            con.hessian(con.hessian_cache, states[t+1], states[t], actions[t], parameters[t], duals[t])
            @views hessians[indices[t]] .+= con.hessian_cache
            fill!(con.hessian_cache, 0.0) # TODO: confirm this is necessary
        end
    end
end

function sparsity_jacobian(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}; 
    row_shift=0) where T

    row = Int[]
    col = Int[]
    for (t, con) in enumerate(constraints) 
        col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_actions[1:t-1])) : 0)
        push!(row, (con.jacobian_sparsity[1] .+ row_shift)...) 
        push!(col, (con.jacobian_sparsity[2] .+ col_shift)...) 
        row_shift += con.num_next_state
    end

    return collect(zip(row, col))
end

function sparsity_hessian(constraints::Vector{Dynamics{T}}, num_state::Vector{Int}, num_actions::Vector{Int}) where T
    row = Int[]
    col = Int[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.hessian_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_actions[1:t-1])) : 0)
            push!(row, (con.hessian_sparsity[1] .+ shift)...) 
            push!(col, (con.hessian_sparsity[2] .+ shift)...) 
        end
    end
    return collect(zip(row, col))
end

num_state_action_next_state(constraints::Vector{Dynamics{T}}) where T = sum([con.num_state + con.num_action for con in constraints]) + constraints[end].num_next_state
num_constraint(constraints::Vector{Dynamics{T}}) where T = sum([con.num_next_state for con in constraints])
num_jacobian(constraints::Vector{Dynamics{T}}) where T = sum([con.num_jacobian for con in constraints])
# num_hessian(constraints::Vector{Dynamics{T}}) where T = sum([con.num_hessian for con in constraints])

function constraint_indices(constraints::Vector{Dynamics{T}}; 
    shift=0) where T
    [collect(shift + (t > 1 ? sum([constraints[s].num_next_state for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_next_state)) for t = 1:length(constraints)]
end 

function jacobian_indices(constraints::Vector{Dynamics{T}}; 
    shift=0) where T
    [collect(shift + (t > 1 ? sum([constraints[s].num_jacobian for s = 1:(t-1)]) : 0) .+ (1:constraints[t].num_jacobian)) for t = 1:length(constraints)]
end

function hessian_indices(constraints::Vector{Dynamics{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, con) in enumerate(constraints) 
        if !isempty(con.hessian_sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            row = collect(con.hessian_sparsity[1] .+ shift)
            col = collect(con.hessian_sparsity[2] .+ shift)
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