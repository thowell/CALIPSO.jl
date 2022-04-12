function constraints!(violations, indices, constraints::Vector{Constraints{T}}, states, actions, parameters) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if con.num_constraint > 0
                con.evaluate(con.evaluate_cache, states[t], actions[t], parameters[t])
                @views violations[indices[t][i]] .= con.evaluate_cache
                # fill!(con.evaluate_cache, 0.0) # TODO: confirm this is necessary 
            end
        end
    end
end

function jacobian!(jacobians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_cache) 
                con.jacobian(con.jacobian_cache, states[t], actions[t], parameters[t])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    jacobians[idx...] = con.jacobian_cache[j] 
                end
                # @views jacobians[indices[t]] .= con.jacobian_cache
                # fill!(con.jacobian_cache, 0.0) # TODO: confirm this is necessary
            end
        end
    end
end

function hessian_lagrangian!(hessians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters, duals) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.hessian_cache)
                con.hessian(con.hessian_cache, states[t], actions[t], parameters[t], duals[t][i])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    hessians[idx...] += con.hessian_cache[j]
                end
                # @views hessians[indices[t]] .+= con.hessian_cache
                # fill!(con.hessian_cache, 0.0) # TODO: confirm this is necessary
            end
        end
    end
end

function sparsity_jacobian(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}; 
    row_shift=0) where T

    SP = Vector{Vector{Tuple{Int,Int}}}[]

    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]

        for (i, con) in enumerate(cons)
            row = Int[]
            col = Int[]

            col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.jacobian_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_sparsity[2] .+ col_shift)...) 

            s = collect(zip(row, col))
            push!(sp, s)

            row_shift += con.num_constraint
        end
        push!(SP, sp)
    end

    return SP
end

function sparsity_hessian(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    SP = Vector{Vector{Tuple{Int,Int}}}[]
    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]
        for (i, con) in enumerate(cons)
            row = Int[]
            col = Int[]
            if !isempty(con.hessian_sparsity[1])
                shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
                push!(row, (con.hessian_sparsity[1] .+ shift)...) 
                push!(col, (con.hessian_sparsity[2] .+ shift)...) 
            end
            s = collect(zip(row, col))
            push!(sp, s)
        end
        push!(SP, sp)
    end
    return SP
end


num_constraint(constraints::Vector{Constraints{T}}) where T = [sum([con.num_constraint for con in cons]) for cons in constraints]
num_jacobian(constraints::Vector{Constraints{T}}) where T = [sum([con.num_jacobian for con in cons]) for cons in constraints]
# num_hessian(constraints::Constraints{T}) where T = sum([con.num_hessian for con in constraints])

function constraint_indices(constraints::Vector{Constraints{T}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]
    for (t, cons) in enumerate(constraints)
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            indices = [indices..., collect(shift .+ (1:con.num_constraint)),]
            shift += con.num_constraint
        end
        push!(INDICES, indices)
    end

    return INDICES
end 

function jacobian_indices(constraints::Vector{Constraints{T}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]

    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            push!(indices, collect(shift .+ (1:con.num_jacobian)))
            shift += con.num_jacobian
        end
        push!(INDICES, indices)
    end

    return INDICES
end

function hessian_indices(constraints::Vector{Constraints{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    INDICES = Vector{Vector{Int}}[]
    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            if !isempty(con.hessian_sparsity[1])
                row = Int[]
                col = Int[]
                shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
                push!(row, (con.hessian_sparsity[1] .+ shift)...) 
                push!(col, (con.hessian_sparsity[2] .+ shift)...) 
                rc = collect(zip(row, col))
                push!(indices, [findfirst(x -> x == j, key) for j in rc])
            end
        end
        push!(INDICES, indices)
    end
    return INDICES
end

