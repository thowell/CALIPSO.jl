# PREAMBLE

# PKG_SETUP

# ## Setup

using LinearAlgebra
using Plots
using DirectTrajectoryOptimization 

# ## 
eval_hess_lag = true 

# ## horizon 
T = 51

# ## rocket 
num_state = 6
num_action = 3 
num_parameter = 0 

function rocket(x, u, w)
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

function midpoint_implicit(y, x, u, w)
    h = 0.05 # timestep 
    y - (x + h * rocket(0.5 * (x + y), u, w))
end

# ## model
dt = DirectTrajectoryOptimization.Dynamics(
    midpoint_implicit, 
    num_state, 
    num_state, 
    num_action, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
dynamics = [dt for t = 1:T-1] 

# ## initialization

x1 = [3.0; 2.0; 1.0; 0.0; 0.0; 0.0] 
xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)])
ct = Cost(ot, num_state, num_action, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
cT = Cost(oT, num_state, 0, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
objective = [[ct for t = 1:T-1]..., cT]

# ## constraints
bnd1 = Bound(num_state, num_action)
bndt = Bound(num_state, num_action)
bndT = Bound(num_state, 0)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

# ## initial 
Fx_min = -10.0#-8.0 
Fx_max = 10.0#8.0
Fy_min = -10.0#-8.0 
Fy_max = 10.0#8.0
Fz_min = 0.0#6.0
Fz_max = 100.0#12.5

# Fx_min = -8.0 
# Fx_max = 8.0
# Fy_min = -8.0 
# Fy_max = 8.0
# Fz_min = 6.0
# Fz_max = 12.5

con1 = Constraint(
    (x, u, w) -> [
        x - x1; 
        Fx_min - u[1]; u[1] - Fx_max;
        Fy_min - u[2]; u[2] - Fy_max; 
        Fz_min - u[3]; u[3] - Fz_max;
        ], 
    num_state, num_action,
    indices_inequality=collect(num_state .+ (1:6)),
    evaluate_hessian=eval_hess_lag)
cont = Constraint(
    (x, u, w) -> [
        Fx_min - u[1]; u[1] - Fx_max;
        Fy_min - u[2]; u[2] - Fy_max; 
        Fz_min - u[3]; u[3] - Fz_max;
        ], 
    num_state, num_action,
    indices_inequality=collect(0 .+ (1:6)),
    evaluate_hessian=eval_hess_lag)
conT = Constraint((x, u, w) -> x - xT, 
    num_state, num_action,
    evaluate_hessian=eval_hess_lag)
constraints = [con1, [cont for t = 2:T-1]..., conT]

# ## problem 
solver = Solver(dynamics, objective, constraints, bounds, 
    evaluate_hessian=eval_hess_lag,
    options=Options{Float64}())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

initialize_states!(solver, x_interpolation)
initialize_controls!(solver, u_guess)

# ## solve
@time solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

@show x_sol[1]
@show x_sol[T]

# ## state
plot(hcat(x_sol...)', 
    label=["px" "py" "pz" "vx" "vy" "vz"])

# ## control
plot(hcat(u_sol[1:end-1]..., u_sol[end-1])', 
    linetype=:steppost,
    legend=:topleft,
    label=["Fx" "Fy" "Fz"])

soc_check = [norm(u[1:2]) < u[3] for u in u_sol]
all(soc_check)