# CALIPSO
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO

# Examples
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra

function ⊙(q1,q2)
    s1 = q1[1]
    v1 = q1[2:4]
    s2 = q2[1]
    v2 = q2[2:4]
    [s1*s2 - dot(v1,v2);
     s1*v2 + s2*v1 + cross(v1,v2)]
end
# ## horizon
T = 101

# ## rocket
num_state = 6
num_action = 11 # [tx;ty;tz;g+;g-;c+;c-]
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
    h = 0.025*2 # timestep
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

a = -0.5
b = 3.0
c = 0.5
d = 3.0

function stc_con(x,u)
    tx,ty,tz,g1p,g1m,c1p,c1m,g2p,g2m,c2p,c2m = u

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

eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
eqSTC = Constraint((x,u,w) -> stc_con(x,u),                 # g+ ⋅ c- = 0
                               num_state,num_action)
eq = [eq1, [eqSTC for t = 2:T-1]...,eqT]


ineq = [[Constraint((x, u, w) ->
    [
        u[1] - Fx_min; Fx_max - u[1];
        u[2] - Fy_min; Fy_max - u[2];
        u[3] - Fz_min; Fz_max - u[3];
        u[4:7];
        u[8:11]                         # g+,g-,c+,c- all ≥ 0
    ], num_state, num_action
) for t = 1:T-1]..., Constraint()]

so = [[Constraint()] for t = 1:T]

# ## problem
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
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

methods = ProblemMethods(trajopt)

solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    options=Options(verbose=true,penalty_initial=1.0e1,max_residual_iterations = 500))
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
plot($Xm(1,:),$Xm(3,:),'bo')
s = 10;
for i = 1:$T-1
    quiver($Xm(1,i),$Xm(3,i),$Um(1,i)/s,$Um(3,i)/s,'r')
end
patch([-5 $a $a -5],[0 0 $b $b],'b')
patch([$c 5 5 $c],[0 0 $d $d],'b')
hold off
"

mat"
figure
title('Thrust')
hold on
plot($Um(1:3,:)')
hold off
"

# # mat"
# # figure
# # title('Thrust')
# # hold on
# # plot($Um[1:3]')
# # hold off
# # "
#
# using MeshCat
# using Meshing
# using MeshIO
# vis = Visualizer()
# #
# # include("/Users/kevintracy/devel/robot_meshes/rocket/rocket.jl")
# # setbarge!(vis["barge"])
# # setrocket!(vis["rocket"])
# #
# #
# # setrocket!(vis; scale=1.0)
# # setbarge!(vis; scale=1.0)
# # setocean!(vis; dim=40)
#
# obj_starship = joinpath(pwd(), "/Users/kevintracy/devel/robot_meshes/starship/Starship.obj")
# # mtl_starship = joinpath(pwd(), "/Users/kevintracy/devel/robot_meshes/starship/Starship.mtl")
#
# # meshfile = joinpath(@__DIR__, "space_x_booster.obj")
# meshfile = joinpath(pwd(), "/Users/kevintracy/devel/robot_meshes/starship/Starship.obj")
# obj = MeshFileObject(meshfile)
# setobject!(vis["starship"], obj)
# # scale = 1.0
# # settransform!(vis["starship"], compose(Translation(0,0,0), LinearMap(scale*RotX(pi/2))))
#
#
# # # ctm_platform = ModifiedMeshFileObject(obj_starship,mtl_starship,scale=1.0)
# # ctm_starship = MeshFileObject(obj_starship)
# # setobject!(vis["starship"],ctm_starship)
#
# anim = MeshCat.Animation(floor(Int,1/0.05))
# q0 = UnitQuaternion([cos(pi/4);sin(pi/4);0;0])
# θv = [atan(u_sol[k][3],u_sol[k][1]) for k = 1:(T-1)]
# for k = 1:T-1
#     atframe(anim,k) do
#         # θ = atan(u_sol[k][3],u_sol[k][1]) - π/2
#         θ = -θv[k] + π/2
#         # θ = deg2rad(86)
#         r = 10*x_sol[k][1:3]
#         q = [cos(θ/2);0;sin(θ/2);0]
#         settransform!(vis["starship"], compose(Translation(r), LinearMap(UnitQuaternion(q)*q0)))
#     end
# end
# setanimation!(vis, anim)
#
