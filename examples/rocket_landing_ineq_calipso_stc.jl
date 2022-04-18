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
T = 51

# ## rocket
num_state = 6
num_action = 7 # [tx;ty;tz;g+;g-;c+;c-]
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
x1 = [-5.0;0;5;0;0;0]
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
Fx_min = -10.0#-8.0
Fx_max = 10.0#8.0
Fy_min = -10.0#-8.0
Fy_max = 10.0#8.0
Fz_min = 0.0#6.0
Fz_max = 20.0#12.5


# state triggered constraints
# if g(x) > 0, then c(x) ≧ 0
# g(x) = -x[1] + a > 0
# c(x) = x[3] - b ≧ 0
#
# this is handled by introducing 4 new variables at each time step as controls
# g+,g-,c+,c- all ≥ 0
# g+ - g- = g(x)
# c+ - c- = c(x)
# g+ ⋅ c- = 0

a = -1.0
b = 3.0

eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
eqSTC = Constraint((x,u,w) -> [u[5] - u[4] - (-x[1] + a);  # g+ - g- = -x[1] + a
                               u[7] - u[6] - (x[3] - b);   # c+ - c- = x[3] - b
                               u[5]*u[6]],                 # g+ ⋅ c- = 0
                               num_state,num_action)
eq = [eq1, [eqSTC for t = 2:T-1]...,eqT]


ineq = [[Constraint((x, u, w) ->
    [
        u[1] - Fx_min; Fx_max - u[1];
        u[2] - Fy_min; Fy_max - u[2];
        u[3] - Fz_min; Fz_max - u[3];
        u[4:7]                         # g+,g-,c+,c- all ≥ 0
    ], num_state, num_action
) for t = 1:T-1]..., Constraint()]

so = [[Constraint()] for t = 1:T]

# ## problem
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so)

# ## initialize
using Random
Random.seed!(2)
x_interpolation = [x1 + 0.1*randn(6) for t = 1:T ]
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

methods = ProblemMethods(trajopt)

# ## solver
solver = Solver(methods, trajopt.num_variables, trajopt.num_equality, trajopt.num_cone,
    options=Options(verbose=true))
initialize_states!(solver, trajopt, x_interpolation)
initialize_controls!(solver, trajopt, u_guess)

# ## solve
solve!(solver)
norm(solver.data.residual, Inf) < 1.0e-5

# ## solution
x_sol, u_sol = get_trajectory(solver, trajopt)

@show x_sol[1]
@show x_sol[T]

Xm = hcat(x_sol...)
Um = hcat(u_sol...)

using MATLAB
mat"
figure
hold on
plot($Xm(1,:),$Xm(3,:))
s = 10;
for i = 1:$T-1
    quiver($Xm(1,i),$Xm(3,i),$Um(1,i)/s,$Um(3,i)/s,'r')
end
patch([-5 $a $a -5],[0 0 $b $b],'b')
hold off
"

mat"
figure
hold on
plot($Um')
hold off
"
