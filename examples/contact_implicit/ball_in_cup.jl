# ## dependencies 
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO
include("models/ball_in_cup.jl")

# ## horizon
horizon = 21
timestep = 0.075

# ## dimensions
nx = 2 * ballincup.num_configuration
nu = ballincup.num_action + 1

num_states = [8, [9 for t = 2:horizon]...] 
num_actions = [3 for t = 1:horizon-1] 

# ## dynamics
dynamics = [(y, x, u) -> ballincup_discrete(ballincup, [timestep], y, x, u) for t = 1:horizon-1]

# ## states
x1 = [0.0; 0.0; 0.0; -0.99; 0.0; 0.0; 0.0; -0.99]
xT = [0.0; 0.0; 0.0; 0.125; 0.0; 0.0; 0.0; 0.125]

# ## intermediate states
xM1 = [0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0]
dr = sqrt(0.5 * ballincup.string_length^2)
xM2 = [0.0; 0.0; dr; dr; 0.0; 0.0; dr; dr]
tM1 = 11
tM2 = 16

# ## objective
function obj1(x, u)
    J = 0.0
    v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
    J += 0.5 * 1.0e-1 * dot(v, v)
    Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
    J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
    J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
    return J
end

function objt(x, u)
    J = 0.0
    v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
    J += 0.5 * 1.0e-1 * dot(v, v)
    Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
    J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
    J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
    return J
end

function objT(x, u)
    J = 0.0
    v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
    J += 0.5 * 1.0e-1 * dot(v, v)
    Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
    J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
    Δcup_ball = x[4 .+ (1:2)] - x[6 .+ (1:2)]
    J += 0.5 * 1.0 * dot(Δcup_ball, Δcup_ball)
    return J
end

objective = [
        obj1, 
        [objt for t = 2:horizon-1]..., 
        objT,
]

# ## constraints
function equality_1(x, u)
    [
        x - x1;
    ]
end

function equality_t(x, u)
    [
        contact_constraints_equality_t(ballincup, timestep, x, u);
    ]
end

function equality_tM1(x, u)
    [
        contact_constraints_equality_t(ballincup, timestep, x, u);
        x[6 .+ (1:2)] - xM1[6 .+ (1:2)];
    ]
end

function equality_tM2(x, u)
    [
        contact_constraints_equality_t(ballincup, timestep, x, u);
        x[6 .+ (1:2)] - xM2[6 .+ (1:2)];
    ]
end

function equality_T(x, u)
    [
        contact_constraints_equality_t(ballincup, timestep, x, u);
        x[1:2] - xT[1:2];
        x[4 .+ (1:2)] - xT[4 .+ (1:2)];
        x[6 .+ (1:2)] - xT[6 .+ (1:2)];
    ]
end

equality = [
        equality_1, 
        [t == tM1 ? equality_tM1 : (t == tM2 ? equality_tM2 : equality_t) for t = 2:horizon-1]..., 
        equality_T,
]

function inequality_1(x, u)
    [
        contact_constraints_inequality_t(ballincup, timestep, x, u);
    ]
end

function inequality_t(x, u)
    [
        contact_constraints_inequality_t(ballincup, timestep, x, u);
    ]
end

function inequality_T(x, u)
    [
        contact_constraints_inequality_T(ballincup, timestep, x, u);
    ]
end

nonnegative = [
        inequality_1, 
        [inequality_t for t = 2:horizon-1]..., 
        inequality_T,
]

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions,
    equality=equality,
    nonnegative=nonnegative,
    options=Options()
    );

# ## initialize
x_interpolation = [linear_interpolation(x1, xM1, 11)..., linear_interpolation(xM1, xM2, 6)[2:end]..., linear_interpolation(xM2, xT, 6)[2:end]...]
state_guess = [x_interpolation[1], [[x_interpolation[t]; zeros(1)] for t = 2:horizon]...]
action_guess = [[1.0e-3 * randn(2); 1.0e-3 * ones(1)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_controls!(solver, action_guess)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visualizer
vis = Visualizer()
render(vis)

# ## visualize
visualize!(vis, ballincup, 
    [[x[1:4] for x in x_sol]..., x_sol[end][4 .+ (1:4)]], 
    Δt=timestep,
    r_cup=0.1,
    r_ball=0.05
)