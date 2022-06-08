struct Cost{T}
    cost::Function
    gradient_variables::Function
    gradient_parameters::Function
    jacobian_variables_variables::Function
    jacobian_variables_parameters::Function
    num_gradient_variables::Int 
    num_gradient_parameters::Int 
    num_jacobian_variables_variables::Int
    num_jacobian_variables_parameters::Int
    sparsity_variables_variables::Vector{Vector{Int}}
    sparsity_variables_parameters::Vector{Vector{Int}}
    cost_cache::Vector{T}
    gradient_variables_cache::Vector{T}
    gradient_parameters_cache::Vector{T}
    jacobian_variables_variables_cache::Vector{T}
    jacobian_variables_parameters_cache::Vector{T}
end

""" 
    Cost(cost, num_state, num_action;
        num_parameter, checkbounds, constraint_tensor)

    stage-cost type 

    cost: Function 
    num_state: Int - dimension of state 
    num_action: Int - dimension of action 
    num_parameter: Int - dimension of problem data 
    checkbounds: Bool - flag for checking @inbounds for codegen methods 
    constraint_tensor: Bool - flag for generating second-derivative methods 
"""
function Cost(cost::Function, num_state::Int, num_action::Int; 
    num_parameter::Int=0,
    checkbounds=true,
    constraint_tensor=true)

    #TODO: option to load/save methods
    @variables x[1:num_state], u[1:num_action], w[1:num_parameter]
    
    c = num_parameter > 0 ? cost(x, u, w) : cost(x, u)
    gz = Symbolics.gradient(c, [x; u])
    gw = Symbolics.gradient(c, w)

    num_gradient_variables = num_state + num_action
    num_gradient_parameters = num_parameter

    cost_func = Symbolics.build_function([c], x, u, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    gradient_variables_func = Symbolics.build_function(gz, x, u, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    gradient_parameters_func = Symbolics.build_function(gw, x, u, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    # if constraint_tensor 
    jacobian_variables_variables = Symbolics.sparsejacobian(constraint_tensor ? gz : zeros(num_state + num_action), [x; u])
    jacobian_variables_parameters = Symbolics.sparsejacobian(constraint_tensor ? gz : zeros(num_state + num_action), w)

    jacobian_variables_variables_func = Symbolics.build_function(jacobian_variables_variables.nzval, x, u, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    jacobian_variables_parameters_func = Symbolics.build_function(jacobian_variables_parameters.nzval, x, u, w,
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    sparsity_variables_variables = [findnz(jacobian_variables_variables)[1:2]...]
    num_jacobian_variables_variables = length(jacobian_variables_variables.nzval)

    sparsity_variables_parameters = [findnz(jacobian_variables_parameters)[1:2]...]
    num_jacobian_variables_parameters = length(jacobian_variables_parameters.nzval)
    # else 
    #     jacobian_variables_variables_func = Expr(:null)
    #     jacobian_variables_parameters_func = Expr(:null)

    #     sparsity_variables_variables = [Int[]]
    #     num_jacobian_variables_variables = 0

    #     sparsity_variables_parameters = [Int[]]
    #     num_jacobian_variables_parameters = 0
    # end

    return Cost(
        cost_func, 
        gradient_variables_func, 
        gradient_parameters_func, 
        jacobian_variables_variables_func, 
        jacobian_variables_parameters_func, 
        num_gradient_variables, 
        num_gradient_parameters, 
        num_jacobian_variables_variables,
        num_jacobian_variables_parameters,
        sparsity_variables_variables,
        sparsity_variables_parameters,
        zeros(1), 
        zeros(num_gradient_variables), 
        zeros(num_gradient_parameters), 
        zeros(num_jacobian_variables_variables),
        zeros(num_jacobian_variables_parameters),
    )
end

Objective{T} = Vector{Cost{T}} where T

function cost(c, objective::Vector{Cost{T}}, states, actions, parameters) where T 
    for (t, obj) in enumerate(objective)
        obj.cost(obj.cost_cache, states[t], actions[t], parameters[t])
        c[1] += obj.cost_cache[1]
    end
    return
end

function gradient_variables!(gradient, indices, objective::Vector{Cost{T}}, states, actions, parameters) where T
    for (t, obj) in enumerate(objective)
        obj.gradient_variables(obj.gradient_variables_cache, states[t], actions[t], parameters[t])
        @views gradient[indices[t]] .+= obj.gradient_variables_cache
    end
end

function gradient_parameters!(gradient, indices, objective::Vector{Cost{T}}, states, actions, parameters) where T
    for (t, obj) in enumerate(objective)
        if !isempty(obj.gradient_parameters_cache)
            obj.gradient_parameters(obj.gradient_parameters_cache, states[t], actions[t], parameters[t])
            @views gradient[indices[t]] .+= obj.gradient_parameters_cache
        end
    end
end

function jacobian_variables_variables!(jacobians, sparsity, objective::Vector{Cost{T}}, states, actions, parameters) where T
    count = 1
    for (t, obj) in enumerate(objective)
        obj.jacobian_variables_variables(obj.jacobian_variables_variables_cache, states[t], actions[t], parameters[t])
        for v in obj.jacobian_variables_variables_cache
            jacobians[count] += v
            count += 1
        end
    end
end

function jacobian_variables_parameters!(jacobians, sparsity, objective::Vector{Cost{T}}, states, actions, parameters) where T
    count = 1
    for (t, obj) in enumerate(objective)
        if !isempty(obj.jacobian_variables_parameters_cache)
            obj.jacobian_variables_parameters(obj.jacobian_variables_parameters_cache, states[t], actions[t], parameters[t])
            for v in obj.jacobian_variables_parameters_cache
                jacobians[count] += v
                count += 1
            end
        end 
    end
end

function sparsity_jacobian_variables_variables(objective::Vector{Cost{T}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]
    for (t, obj) in enumerate(objective)
        row = Int[]
        col = Int[]
        if !isempty(obj.sparsity_variables_variables[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (obj.sparsity_variables_variables[1] .+ shift)...) 
            push!(col, (obj.sparsity_variables_variables[2] .+ shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    return sp
end

function sparsity_jacobian_variables_parameters(objective::Vector{Cost{T}}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameter::Vector{Int}) where T
    sp = Vector{Tuple{Int,Int}}[]
    for (t, obj) in enumerate(objective)
        row = Int[]
        col = Int[]
        if !isempty(obj.sparsity_variables_parameters[1])
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            push!(row, (obj.sparsity_variables_parameters[1] .+ row_shift)...) 
            push!(col, (obj.sparsity_variables_parameters[2] .+ col_shift)...) 
        end
        s = collect(zip(row, col))
        push!(sp, s)
    end
    return sp
end

function jacobian_variables_variables_indices(objective::Vector{Cost{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, obj) in enumerate(objective)
        if !isempty(obj.sparsity_variables_variables[1])
            row = Int[]
            col = Int[]
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (obj.sparsity_variables_variables[1] .+ shift)...) 
            push!(col, (obj.sparsity_variables_variables[2] .+ shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end

function jacobian_variables_parameters_indices(objective::Vector{Cost{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}, num_parameter::Vector{Int}) where T
    indices = Vector{Int}[]
    for (t, obj) in enumerate(objective)
        if !isempty(obj.sparsity_variables_parameters[1])
            row = Int[]
            col = Int[]
            row_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            col_shift = (t > 1 ? (sum(num_parameter[1:t-1])) : 0)
            push!(row, (obj.sparsity_variables_parameters[1] .+ row_shift)...) 
            push!(col, (obj.sparsity_variables_parameters[2] .+ col_shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end

# num_gradient(objective::Vector{Cost{T}}{T}) where T = sum([obj.num_gradient  for obj in objective])
# num_hessian(objective::Vector{Cost{T}}{T}) where T  = sum([obj.num_hessian   for obj in objective])
