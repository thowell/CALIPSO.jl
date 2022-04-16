function constraints!(violations, indices, constraints::Vector{Constraints{T}}, states, actions, parameters) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if con.num_constraint > 0
                con.constraint(con.constraint_cache, states[t], actions[t], parameters[t])
                @views violations[indices[t][i]] .= con.constraint_cache
            end
        end
    end
end

function jacobian_variables!(jacobians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_variables_cache) 
                con.jacobian_variables(con.jacobian_variables_cache, states[t], actions[t], parameters[t])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    jacobians[idx...] = con.jacobian_variables_cache[j] 
                end
            end
        end
    end
end

function jacobian_parameters!(jacobians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_parameters_cache) 
                con.jacobian_parameters(con.jacobian_parameters_cache, states[t], actions[t], parameters[t])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    jacobians[idx...] = con.jacobian_parameters_cache[j] 
                end
            end
        end
    end
end

function jacobian_variables_variables!(jacobians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters, duals) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_variables_variables_cache)
                con.constraint_dual_jacobian_variables_variables(con.jacobian_variables_variables_cache, states[t], actions[t], parameters[t], duals[t][i])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    jacobians[idx...] += con.jacobian_variables_variables_cache[j]
                end
            end
        end
    end
end

function jacobian_variables_parameters!(jacobians, sparsity, constraints::Vector{Constraints{T}}, states, actions, parameters, duals) where T
    for (t, cons) in enumerate(constraints)
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_variables_parameters_cache)
                con.constraint_dual_jacobian_variables_parameters(con.jacobian_variables_parameters_cache, states[t], actions[t], parameters[t], duals[t][i])
                for (j, idx) in enumerate(sparsity[t][i]) 
                    jacobians[idx...] += con.jacobian_variables_parameters_cache[j]
                end
            end
        end
    end
end

function sparsity_jacobian_variables(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}; 
    row_shift=0) where T

    SP = Vector{Vector{Tuple{Int,Int}}}[]

    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]

        for (i, con) in enumerate(cons)
            row = Int[]
            col = Int[]

            col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.jacobian_variables_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_variables_sparsity[2] .+ col_shift)...) 

            s = collect(zip(row, col))
            push!(sp, s)

            row_shift += con.num_constraint
        end
        push!(SP, sp)
    end

    return SP
end

function sparsity_jacobian_parameters(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}; 
    row_shift=0) where T

    SP = Vector{Vector{Tuple{Int,Int}}}[]

    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]

        for (i, con) in enumerate(cons)
            row = Int[]
            col = Int[]

            col_shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
            push!(row, (con.jacobian_parameters_sparsity[1] .+ row_shift)...) 
            push!(col, (con.jacobian_parameters_sparsity[2] .+ col_shift)...) 

            s = collect(zip(row, col))
            push!(sp, s)

            row_shift += con.num_constraint
        end
        push!(SP, sp)
    end

    return SP
end

function sparsity_jacobian_variables_variables(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    SP = Vector{Vector{Tuple{Int,Int}}}[]
    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]
        for (i, con) in enumerate(cons)
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
        push!(SP, sp)
    end
    return SP
end

function sparsity_jacobian_variables_parameters(constraints::Vector{Constraints{T}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    SP = Vector{Vector{Tuple{Int,Int}}}[]
    for (t, cons) in enumerate(constraints)
        sp = Vector{Tuple{Int,Int}}[]
        for (i, con) in enumerate(cons)
            row = Int[]
            col = Int[]
            if !isempty(con.jacobian_variables_parameters_sparsity[1])
                shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
                push!(row, (con.jacobian_variables_parameters_sparsity[1] .+ shift)...) 
                push!(col, (con.jacobian_variables_parameters_sparsity[2] .+ shift)...) 
            end
            s = collect(zip(row, col))
            push!(sp, s)
        end
        push!(SP, sp)
    end
    return SP
end


num_constraint(constraints::Vector{Constraints{T}}) where T = [sum([con.num_constraint for con in cons]) for cons in constraints]
num_jacobian_variables(constraints::Vector{Constraints{T}}) where T = [sum([con.num_jacobian_variables for con in cons]) for cons in constraints]
num_jacobian_parameters(constraints::Vector{Constraints{T}}) where T = [sum([con.num_jacobian_parameters for con in cons]) for cons in constraints]

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

function jacobian_variables_indices(constraints::Vector{Constraints{T}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]

    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            push!(indices, collect(shift .+ (1:con.num_jacobian_variables)))
            shift += con.num_jacobian_variables
        end
        push!(INDICES, indices)
    end

    return INDICES
end

function jacobian_parameters_indices(constraints::Vector{Constraints{T}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]

    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            push!(indices, collect(shift .+ (1:con.num_jacobian_parameters)))
            shift += con.num_jacobian_parameters
        end
        push!(INDICES, indices)
    end

    return INDICES
end

function jacobian_variables_variables_indices(constraints::Vector{Constraints{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    INDICES = Vector{Vector{Int}}[]
    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_variables_variables_sparsity[1])
                row = Int[]
                col = Int[]
                shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
                push!(row, (con.jacobian_variables_variables_sparsity[1] .+ shift)...) 
                push!(col, (con.jacobian_variables_variables_sparsity[2] .+ shift)...) 
                rc = collect(zip(row, col))
                push!(indices, [findfirst(x -> x == j, key) for j in rc])
            end
        end
        push!(INDICES, indices)
    end
    return INDICES
end

function jacobian_variables_parameters_indices(constraints::Vector{Constraints{T}}, key::Vector{Tuple{Int,Int}}, num_state::Vector{Int}, num_action::Vector{Int}) where T
    INDICES = Vector{Vector{Int}}[]
    for (t, cons) in enumerate(constraints) 
        indices = Vector{Int}[]
        for (i, con) in enumerate(cons)
            if !isempty(con.jacobian_variables_parameters_sparsity[1])
                row = Int[]
                col = Int[]
                shift = (t > 1 ? (sum(num_state[1:t-1]) + sum(num_action[1:t-1])) : 0)
                push!(row, (con.jacobian_variables_parameters_sparsity[1] .+ shift)...) 
                push!(col, (con.jacobian_variables_parameters_sparsity[2] .+ shift)...) 
                rc = collect(zip(row, col))
                push!(indices, [findfirst(x -> x == j, key) for j in rc])
            end
        end
        push!(INDICES, indices)
    end
    return INDICES
end


