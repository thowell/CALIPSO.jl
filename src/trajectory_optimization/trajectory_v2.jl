function trajectory(states::Vector{Vector{T}}, actions::Vector{Vector{T}}) where T
    H = length(states) 
    @assert length(actions) == H - 1

    # dimensions
    nx = [length(s) for s in states] 
    nu = [length(u) for u in actions]

    # indices 
    x_idx, u_idx = state_action_indices(nx, nu) 

    # trajectory 
    trajectory = zeros(sum(nx) + sum(nu)) 

    for t = 1:H 
        trajectory[x_idx[t]] = states[t] 
        t == H && continue 
        trajectory[u_idx[t]] = actions[t] 
    end

    return trajectory 
end