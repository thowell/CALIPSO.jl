# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra

# ## horizon 
T = 101

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
dt = Dynamics(
    midpoint_implicit, 
    num_state, 
    num_state, 
    num_action, 
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1] 

# ## initialization
x1 = [3.0; 2.0; 1.0; 0.0; 0.0; 0.0] 
xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)])
ct = Cost(ot, num_state, num_action, 
    num_parameter=num_parameter)
cT = Cost(oT, num_state, 0, 
    num_parameter=num_parameter)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints 
# Fx_min = -10.0#-8.0 
# Fx_max = 10.0#8.0
# Fy_min = -10.0#-8.0 
# Fy_max = 10.0#8.0
# Fz_min = 0.0#6.0
# Fz_max = 100.0#12.5

eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
eq = [eq1, [Constraint() for t = 2:T-1]..., eqT]

# ineq = [[Constraint((x, u, w) -> 
#     [
#         u[1] - Fx_min; Fx_max - u[1];
#         u[2] - Fy_min; Fy_max - u[2]; 
#         u[3] - Fz_min; Fz_max - u[3];
#     ], num_state, num_action
# ) for t = 1:T-1]..., Constraint()]
ineq = [Constraint() for t = 1:T]

function thrust_cone(x, u, w) 
    [
        u[3]; 
        u[1]; 
        u[2];
    ]
end

# so = [[Constraint()] for t = 1:T]
so = [[[Constraint(thrust_cone, num_state, num_action)] for t = 1:T-1]..., [Constraint()]]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [[1.0; 0.1; 0.1] for t = 1:T-1]

methods = ProblemMethods(trajopt)
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# ## solver
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    options=Options(verbose=true, penalty_initial=1.0e6))
initialize_states!(solver, trajopt, x_interpolation) 
initialize_controls!(solver, trajopt, u_guess)

# ## solve 
solve!(solver)
norm(solver.data.residual, Inf) < 1.0e-4

# ## solution
# x_sol, u_sol = get_trajectory(solver, trajopt)

# @show x_sol[1]
# @show x_sol[T]

# # ## state
# using Plots
# plot(hcat(x_sol...)', 
#     label=["px" "py" "pz" "vx" "vy" "vz"])

# # ## control
# plot(hcat(u_sol[1:end-1]..., u_sol[end-1])', 
#     linetype=:steppost,
#     legend=:topleft,
#     label=["Fx" "Fy" "Fz"])

# soc_check = [norm(u[1:2]) < u[3] for u in u_sol]
# all(soc_check)