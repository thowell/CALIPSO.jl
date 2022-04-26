using CALIPSO 
using LinearAlgebra 

# ## horizon 
horizon = 11 

# ## dimensions 
num_states = [2 for t = 1:horizon]
num_actions = [1 for t = 1:horizon-1] 
num_parameters = [0 for t = 1:horizon] 

# ## initialization
state_initial = [0.0; 0.0] 
state_goal = [π; 0.0] 

# ## objective 
objective = [
        [(x, u, w) -> dot(x[1:2], x[1:2]) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
        (x, u, w) -> 0.1 * dot(x[1:2], x[1:2]),
]

# ## dynamics
function pendulum(x, u, w)
    mass = 1.0
    length_com = 0.5
    gravity = 9.81
    damping = 0.1

    [
        x[2],
        (u[1] / ((mass * length_com * length_com))
            - gravity * sin(x[1]) / length_com
            - damping * x[2] / (mass * length_com * length_com))
    ]
end

function pendulum_midpoint(y, x, u, w)
    timestep = 0.05 
    y - (x + timestep * pendulum(0.5 * (x + y), u, w))
end

dynamics = [pendulum_midpoint for t = 1:horizon-1]

# ## constraints 
equality = [
        (x, u, w) -> x - state_initial, 
        [(x, u, w) -> zeros(0) for t = 2:horizon-1]..., 
        (x, u, w) -> x - state_goal,
]
nonnegative = [(x, u, w) -> zeros(0) for t = 1:horizon]
second_order = [[(x, u, w) -> zeros(0)] for t = 1:horizon]

# ## solver
solver = Solver(
    objective, 
    dynamics, 
    equality, 
    nonnegative, 
    second_order, 
    num_states, 
    num_actions, 
    num_parameters;
    options=Options(
        verbose=true))

# ## initialize
states_guess = linear_interpolation(state_initial, state_goal, horizon)
actions_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
x = trajectory(states_guess, actions_guess) # assemble into a single vector
initialize!(solver, x)

# ## solve 
solve!(solver)

# @benchmark solve!($solver)

