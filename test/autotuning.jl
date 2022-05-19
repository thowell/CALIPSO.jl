using FiniteDiff 

# ## learning
function loss(state, action, state_reference, action_reference, state_cost, action_cost)
    J = 0.0 
    J += 0.5 * (state - state_reference)' * state_cost * (state - state_reference)
    @show action 
    @show action_reference 
    @show action_cost
    J += 0.5 * (action - action_reference)' * action_cost * (action - action_reference)
    return J 
end

function loss_gradient_state(state, action, state_reference, action_reference, state_cost, action_cost)
    Jx = state_cost * (state - state_reference)
    return Jx
end

function loss_gradient_action(state, action, state_reference, action_reference, state_cost, action_cost)
    Ju = action_cost * (action - action_reference)
    return Ju
end

function total_loss(states, actions, state_references, action_references, state_costs, action_costs)
    horizon = length(states)

    J = 0.0

    for t = 1:horizon-1
        J += loss(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t])
    end

    J += loss(states[horizon], zeros(0), state_references[horizon], zeros(0), state_costs[horizon], zeros(0, 0))

    return J ./ horizon
end

function total_loss_gradient_parameters(states, actions, parameters, state_references, action_references, state_costs, action_costs) 
    # dimensions
    state_dim = length(states[1])
    parameters_dim = length(vcat(parameters...))
    horizon = length(states) 

    # initialize gradient
    Jθ = zeros(parameters_dim)
    
    # initialize Jacobians
    ∂x∂θ = [zeros(state_dim, parameters_dim)]
    ∂u∂θ = []
    ∂π∂x = [] 
    ∂π∂θ = []

    for t = 1:horizon-1
        # ∂u∂θ
        push!(∂π∂x, policy_jacobian_state(parameters, states[t], t))
        push!(∂π∂θ, policy_jacobian_parameters(parameters, states[t], t)) 
        
        # ∂f∂x, ∂f∂u
        ∂f∂x = dynamics_jacobian_state(states[t], actions[t], t)
        ∂f∂u = dynamics_jacobian_action(states[t], actions[t], t)

        push!(∂u∂θ, ∂π∂x[end] * ∂x∂θ[end] + ∂π∂θ[end])
        push!(∂x∂θ, ∂f∂x * ∂x∂θ[end] + ∂f∂u * ∂u∂θ[end])
    end

    for t = 1:horizon-1
        Jθ += ∂x∂θ[t]' * loss_gradient_state(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t]) 
        Jθ += ∂u∂θ[t]' * loss_gradient_action(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t])
    end 

    Jθ += ∂x∂θ[horizon]' * loss_gradient_state(states[horizon], zeros(0), state_reference[horizon], zeros(0), state_costs[horizon], zeros(0, 0))

    return Jθ ./ horizon
end

function loss_gradient_parameters(states, actions, parameters, state_references, action_references, state_costs, action_costs) 
    # dimensions
    state_dim = length(states[1])
    parameters_dim = length(parameters)
    horizon = length(states) 

    # initialize gradient
    Jθ = zeros(parameters_dim)
    
    # initialize Jacobians
    ∂x∂θ = [zeros(state_dim, parameters_dim)]
    ∂u∂θ = []
    ∂π∂x = [] 
    ∂π∂θ = []

    for t = 1:horizon-1
        # ∂u∂
        push!(∂π∂θ, policy_jacobian_parameters(parameters, states[t], t)) 
        push!(∂π∂x, policy_jacobian_state(parameters, states[t], t))
        
        # ∂f∂x, ∂f∂u
        ∂f∂x = dynamics_jacobian_state(states[t], actions[t], t)
        ∂f∂u = dynamics_jacobian_action(states[t], actions[t], t)

        push!(∂u∂θ, ∂π∂x[end] * ∂x∂θ[end] + ∂π∂θ[end])
        push!(∂x∂θ, ∂f∂x * ∂x∂θ[end] + ∂f∂u * ∂u∂θ[end])
    end

    for t = 1:horizon-1
        Jθ += ∂x∂θ[t]' * loss_gradient_state(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t]) 
        Jθ += ∂u∂θ[t]' * loss_gradient_action(states[t], actions[t], state_references[t], action_references[t], state_costs[t], action_costs[t])
    end 

    Jθ += ∂x∂θ[horizon]' * loss_gradient_state(states[horizon], zeros(0), state_reference[horizon], zeros(0), state_costs[horizon], zeros(0, 0))

    return Jθ ./ horizon
end

function rollout(state_initial, parameters, horizon) 
    states = [state_initial] 
    actions = [] 
    for t = 1:horizon-1 
        push!(actions, policy(parameters, states[end], t))
        push!(states, dynamics(states[end], actions[end], t))
    end 
    return states, actions
end

function autotune!(parameters, state_reference, action_reference, state_cost, action_cost, horizon;
    max_iterations=10,
    state_initial=zeros(length(state_reference[1])),
    gradient_tolerance=1.0e-3,
    max_linesearch=25,
    ) 

    # initialize
    cost_gradient = zeros(length(parameters))
    parameters_candidate = zero(length(parameters))

    # evaluate 
    states, actions = rollout(state_initial, parameters, horizon)
    cost = total_loss(states, actions, state_reference, action_reference, state_cost, action_cost)

    # "learning"
    for i = 1:max_iterations
        if i == 1 
            println("iteration: $(i)") 
            println("cost: $(cost)") 
        end    
    
        # gradient
        cost_gradient .= total_loss_gradient_parameters(states, actions, parameters, state_reference, action_reference, state_cost, action_cost)
        
        # line search
        cost_candidate = Inf
        step_size = 1.0
        linesearch_iteration = 0
    
        while cost_candidate >= cost 
            parameters_candidate = parameters - step_size * cost_gradient
            states, actions = rollout(state_initial, parameters_candidate, horizon)
            cost_candidate = total_loss(states, actions, state_reference, action_reference, state_cost, action_cost)
            step_size = 0.5 * step_size
            linesearch_iteration += 1 
            linesearch_iteration > max_linesearch && error("linesearch failure") 
        end
    
        # update 
        cost = cost_candidate 
        parameters .= parameters_candidate
    
        # check convergence
        norm(cost_gradient, Inf) < gradient_tolerance && break
    end
end

# test 
num_state = 2 
num_action = 1
dynamics(x, u, t) = [1.0 1.0; 0.0 1.0] * x + [0.0; 1.0] * u[1]
dynamics_jacobian_state(x, u, t) = [1.0 1.0; 0.0 1.0]
dynamics_jacobian_action(x, u, t) = [0.0; 1.0]

function policy(parameters, state, t)
    num_state = 2 
    num_action = 1
    parameter_indices = collect((t - 1) * (num_action * num_state) .+ (1:(num_state * num_action)))

    K = reshape(parameters[parameter_indices], num_action, num_state) 

    return -reshape(K * state, num_action)
end

policy_jacobian_parameters(parameters, state, t) = FiniteDiff.finite_difference_jacobian(a -> policy(a, state, t), parameters)
policy_jacobian_state(parameters, state, t) = FiniteDiff.finite_difference_jacobian(a -> policy(parameters, a, t), state)

state_history = [rand(num_state)] 
action_history = Vector{Float64}[]
rollout_horizon = 5
parameters = vcat(([randn(num_action, num_state) for t = 1:rollout_horizon-1]...)...)

state_reference = [zeros(num_state) for t = 1:rollout_horizon] 
action_reference = [zeros(num_action) for t = 1:rollout_horizon-1] 
state_cost = [Diagonal(ones(num_state)) for t = 1:rollout_horizon]
action_cost = [Diagonal(ones(num_action)) for t = 1:rollout_horizon-1]

policy(parameters, state_history[1], 3)
policy_jacobian_parameters(parameters, state_history[1], 3)
policy_jacobian_state(parameters, state_history[1], 3)

state_initial = 1.0 * ones(num_state)
state_history, action_history = rollout(state_initial, parameters, rollout_horizon)

loss(state_history[1], action_history[1], state_reference[1], action_reference[1], state_cost[1], action_cost[1])
loss(state_history[end], zeros(0), state_reference[end], zeros(0), state_cost[end], zeros(0, 0))
loss_gradient_state(state_history[1], action_history[1], state_reference[1], action_reference[1], state_cost[1], action_cost[1])
loss_gradient_action(state_history[1], action_history[1], state_reference[1], action_reference[1], state_cost[1], action_cost[1])
total_loss(state_history, action_history, state_reference, action_reference, state_cost, action_cost)

loss_gradient_parameters(state_history, action_history, parameters, state_reference, action_reference, state_cost, action_cost)

autotune!(parameters, state_reference, action_reference, state_cost, action_cost, rollout_horizon;
    max_iterations=10,
    state_initial=state_initial,
    gradient_tolerance=1.0e-2,
    max_linesearch=25,
    ) 

state_history, action_history = rollout(state_initial, parameters, rollout_horizon)
total_loss(state_history, action_history, state_reference, action_reference, state_cost, action_cost)


