"""
    linear_interpolation(initial_state, final_state, horizon)

    method for generating a linear interpolating trajectory 

    initial_state: Vector{Real} - first state 
    final_state: Vector{Real} - last state 
    horizon: Int - length of trajectory
"""
function linear_interpolation(initial_state, final_state, horizon)
    n = length(initial_state)
    X = [copy(Array(initial_state)) for t = 1:horizon]
    for t = 1:horizon
        for i = 1:n
            X[t][i] = (final_state[i] - initial_state[i]) / (horizon - 1) * (t - 1) + initial_state[i]
        end
    end
    return X
end