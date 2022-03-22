struct Cost{T}
    evaluate::Any
    gradient::Any
    hessian::Any
    sparsity::Vector{Vector{Int}}
    evaluate_cache::Vector{T}
    gradient_cache::Vector{T}
    hessian_cache::Vector{T}
end

function Cost(f::Function, num_state::Int, num_action::Int; 
    num_parameter::Int=0,
    evaluate_hessian=false)

    #TODO: option to load/save methods
    @variables x[1:num_state], u[1:num_action], w[1:num_parameter]
    evaluate = f(x, u, w)
    grad = Symbolics.gradient(evaluate, [x; u])
    evaluate_func = eval(Symbolics.build_function([evaluate], x, u, w)[2])
    gradient_func = eval(Symbolics.build_function(grad, x, u, w)[2])
    if evaluate_hessian 
        hess = Symbolics.sparsehessian(evaluate, [x; u])
        hessian_func = eval(Symbolics.build_function(hess.nzval, x, u, w)[2])
        sparsity = [findnz(hess)[1:2]...]
        num_hessian = length(hess.nzval)
    else 
        hessian_func = Expr(:null) 
        sparsity = [Int[]]
        num_hessian = 0
    end

    return Cost(
        evaluate_func, 
        gradient_func, 
        hessian_func, 
        sparsity,
        zeros(1), 
        zeros(num_state + num_action), 
        zeros(num_hessian))
end

Objective{T} = Vector{Cost{T}} where T

function objective(obj::Objective, states, actions, parameters) 
    J = 0.0
    for (t, cost) in enumerate(obj)
        cost.evaluate(cost.evaluate_cache, states[t], actions[t], parameters[t])
        J += cost.evaluate_cache[1]
    end
    return J 
end

function gradient!(gradient, indices, obj::Objective, states, actions, parameters)
    for (t, cost) in enumerate(obj)
        cost.gradient(cost.gradient_cache, states[t], actions[t], parameters[t])
        @views gradient[indices[t]] .+= cost.gradient_cache
        # fill!(cost.gradient_cache, 0.0) # TODO: confirm this is necessary
    end
end

#TODO: test
function hessian!(hessian, indices, obj::Objective, states, actions, parameters, scaling)
    for (t, cost) in enumerate(obj)
        cost.hessian(cost.hessian_cache, states[t], actions[t], parameters[t])
        cost.hessian_cache .*= scaling
        @views hessian[indices[t]] .+= cost.hessian_cache
        # fill!(cost.hessian_cache, 0.0) # TODO: confirm this is necessary
    end
end

function sparsity_hessian(obj::Objective, num_state::Vector{Int}, num_action::Vector{Int})
    row = Int[]
    col = Int[]
    for (t, cost) in enumerate(obj)
        if !isempty(cost.sparsity[1])
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (cost.sparsity[1] .+ shift)...) 
            push!(col, (cost.sparsity[2] .+ shift)...) 
        end
    end
    return collect(zip(row, col))
end

function hessian_indices(obj::Objective, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int})
    indices = Vector{Int}[]
    for (t, cost) in enumerate(obj)
        if !isempty(cost.sparsity[1])
            row = Int[]
            col = Int[]
            shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (cost.sparsity[1] .+ shift)...) 
            push!(col, (cost.sparsity[2] .+ shift)...) 
            rc = collect(zip(row, col))
            push!(indices, [findfirst(x -> x == i, key) for i in rc])
        end
    end
    return indices
end