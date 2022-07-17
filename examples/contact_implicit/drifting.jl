# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RoboDojo
include("models/cybertruck.jl") 

# ## horizon
horizon = 15
timestep = 0.1

# ## Optimization Dimensions 
nx = 2 * nq
nu = 2 + nc * 6
num_states = [nx for t = 1:horizon]
num_actions = [nu for t = 1:horizon-1]

# ## dynamics
dynamics = [(y, x, u) -> cybertruck_dynamics(cybertruck, [timestep], y, x, u) for t = 1:horizon-1]

# ## states
state_initial = [0.0; 1.5; -0.5 * π; 0.0; 1.5; -0.5 * π] 
state_goal = [3.0; 0.0; 0.5 * π; 3.0; 0.0; 0.5 * π]

# ## objective
function obj1(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    # vc = contact_jacobian(model, x[4:6]) * v 
    # J += 0.5 * 1.0e-5 * v[3]^2.0
    # J += 0.5 * 1.0e-5 * dot(vc, vc)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6 * nc)]) * u
    return J
end

function objt(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6 * nc)]) * u
    return J
end

function objT(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 5.0 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    return J
end

objective = [
    obj1, 
    [objt for t = 2:horizon-1]..., 
    objT,
];

# ## constraints
function equality_1(x, u)
    [
        contact_equality(cybertruck, timestep, x, u);
        x - state_initial;
    ]
end

function equality_t(x, u)
    [
        contact_equality(cybertruck, timestep, x, u);
    ]
end

function equality_T(x, u)
    [
        (x - state_goal)[1:3];
    ]
end

equality = [
    equality_1, 
    [equality_t for t = 2:horizon-1]..., 
    equality_T,
];

u_min = [0.0; -0.5]
u_max = [25.0;  0.5]
p_car1 = [3.0, 1.0 * 0.65]
p_car2 = [3.0, 1.0 * -0.65]
circle_obstacle(x, p; r=0.5) = (x[1] - p[1])^2.0 + (x[2] - p[2])^2.0 - r^2.0
nonnegative = [
        [(x, u) -> [
                u_max - u[1:2]; 
                u[1:2] - u_min;
                circle_obstacle(x, p_car1, r=0.1); 
                circle_obstacle(x, p_car2, r=0.1);
            ] for t = 1:horizon-1]..., 
        empty_constraint,
]

second_order = [
    [
        [
            (x, u) -> u[2 .+ (1:3)], 
            (x, u) -> u[2 + 3 .+ (1:3)],
            (x, u) -> u[2 + 6 .+ (1:3)],
            (x, u) -> u[2 + 9 .+ (1:3)],
        ] for t = 1:horizon-1]..., 
    [empty_constraint],
]

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    nonnegative=nonnegative,
    second_order=second_order,
    );

# ## initialize
state_guess = linear_interpolation(state_initial, state_goal, horizon)
action_guess = [[1.0e-3 * randn(2); vcat([[1.0; 0.1; 0.1] for i = 1:(2 * nc)]...)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_actions!(solver, action_guess)

# ## options
solver.options.linear_solver = :LU
solver.options.residual_tolerance=1.0e-3
solver.options.optimality_tolerance=1.0e-3
solver.options.equality_tolerance=1.0e-3
solver.options.complementarity_tolerance=1.0e-3
solver.options.slack_tolerance=1.0e-3

# ## solve
solve!(solver)

# ## solution trajectories
x_sol, u_sol = CALIPSO.get_trajectory(solver)

# ## visualize (NOTE: mesh not included)
vis = Visualizer()
render(vis)
set_background!(vis)
visualize!(vis, cybertruck, [[x_sol[1] for t = 1:10]..., x_sol..., [x_sol[end] for t = 1:10]...], 
    Δt=timestep)