# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using LinearAlgebra

# ## horizon
T = 51

# ## rocket
num_state = 6
num_action = 7 + 4 # [tx; ty; tz; g+; g-; c+; c-]
num_parameter = 0

function rocket(x, u)
    mass = 1.0
    gravity = -9.81

    p = x[1:3]
    v = x[4:6]

    f = u[1:3]

    [
        v;
        [0.0; 0.0; gravity] + 1.0 / mass * f;
    ]
end

function midpoint_implicit(y, x, u)
    h = 0.05 # timestep
    y - (x + h * rocket(0.5 * (x + y), u))
end

# ## dimensions
num_states = [num_state for t = 1:T]
num_actions = [num_action for t = 1:T-1]

dynamics = [midpoint_implicit for t = 1:T-1]

# ## initialization
x1 = [-5.0; 0.0; 5.0; 0.0; 0.0; 0.0]
xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# ## objective
objective = [
    [(x, u) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]) + 0.1 * dot(u[1:3], u[1:3]) for t = 1:T-1]...,
    (x, u) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)])
]

# ## constraints
Fx_min = -10.0 # -8.0
Fx_max = 10.0 # 8.0
Fy_min = -10.0 # -8.0
Fy_max = 10.0 # 8.0
Fz_min = 0.0 # 6.0
Fz_max = 20.0 # 12.5

a = -0.5
b = 3.0
c = 0.3
d = 3.0

function stc_con(x,u)
    tx, ty, tz, g1p, g1m, c1p, c1m, g2p, g2m, c2p, c2m = u

    g1 = -x[1] + a # ≧ 0
    c1 = x[3] - b  # ≧ 0

    g2 = x[1] - c
    c2 = x[3] - d

    [g1p - g1m - g1;
     c1p - c1m - c1;
     g1p*c1m;
     g2p - g2m - g2;
     c2p - c2m - c2;
     g2p*c2m]
end

equality = [(x, u) -> x - x1,[stc_con for t = 2:T-1]...,(x, u) -> x - xT]

nonnegative = [[(x, u) ->
    [
        u[1] - Fx_min; Fx_max - u[1];
        u[2] - Fy_min; Fy_max - u[2];
        u[3] - Fz_min; Fz_max - u[3];
        u[4:7];
        u[8:11]                         # g+, g-, c+, c- all ≥ 0
    ] for t = 1:T-1]..., empty_constraint]
x_interpolation = linear_interpolation(x1, xT, T)

# ## solver 
## NOTE: we codegen the entire problem as a single NLP--this make take some time
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality, 
    nonnegative=nonnegative, 
    options=Options(
        verbose=true,
        penalty_initial=1.0e3)
);

# ## initialization 
for i = 1:T
    h = 0.05/2
    x_interpolation[i][4:6] = (xT[1:3]-x1[1:3])/(h*T)
end

u_guess = [zeros(num_action) for t = 1:T-1]

for i = 1:T-1
    u_guess[i][1:3] = [0;0;9.8]
    g1 = -x_interpolation[i][1] + a
    if g1 >= 0
        u_guess[i][4] = g1
        u_guess[i][5] = 0
    else
        u_guess[i][4] = 0
        u_guess[i][5] = -g1
    end
    c1 =  x_interpolation[i][3] - b
    if c1 >= 0
        u_guess[i][6] = c1
        u_guess[i][7] = 0
    else
        u_guess[i][6] = 0
        u_guess[i][7] = -c1
    end

    g2 = x_interpolation[i][1] - c
    if g2 >= 0
        u_guess[i][4+4] = g2
        u_guess[i][5+4] = 0
    else
        u_guess[i][4+4] = 0
        u_guess[i][5+4] = -g2
    end
    c2 =  x_interpolation[i][3] - d
    if c2 >= 0
        u_guess[i][6+4] = c2
        u_guess[i][7+4] = 0
    else
        u_guess[i][6+4] = 0
        u_guess[i][7+4] = -c2
    end
end

initialize_states!(solver, x_interpolation)
initialize_actions!(solver, u_guess)

# ## solver
solve!(solver)

# ## solution
x_sol, u_sol  = get_trajectory(solver)

## TODO: add animation
