# ## indices
function state_action_indices(nx::Vector{Int}, nu::Vector{Int}) 
    T = length(nx) 
    @assert length(nu) == T - 1

    x_idx = [sum(nx[1:(t-1)]) + sum(nu[1:(t-1)]) .+ collect(1:nx[t]) for t = 1:T]
    u_idx = [sum(nx[1:t]) + sum(nu[1:(t-1)]) .+ collect(1:nu[t]) for t = 1:T-1]

    return x_idx, u_idx 
end

function parameter_indices(np::Vector{Int}) 
    T = length(np)
    p_idx = [sum(np[1:(t-1)]) .+ collect(1:np[t]) for t = 1:T]
    
    return p_idx 
end


function constraint_indices(dims::Vector{Int}; 
    shift=0) where T

    indices = Vector{Int}[]

    for d in dims
        indices = [indices..., collect(shift .+ (1:d)),]
        shift += d
    end

    return vcat(indices...)
end 

function constraint_indices(dims::Vector{Vector{Int}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]
    for dim in dims
        indices = Vector{Int}[]
        for d in dim
            indices = [indices..., collect(shift .+ (1:d)),]
            shift += d
        end
        push!(INDICES, indices)
    end

    return [(INDICES...)...]
end 