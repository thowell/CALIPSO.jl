# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra
using RoboDojo

"""
    CYBERTRUCK
"""
struct CYBERTRUCK{T} <: RoboDojo.Model{T}
	# dimensions
	nq::Int # generalized coordinates
	nu::Int # controls
	nw::Int # parameters
	nc::Int # contact points

    mass::T
    inertia::T

    friction_body_world::Vector{T}
    friction_joint::Vector{T} 
end

function mass_matrix(model::CYBERTRUCK, q) 
    Diagonal([model.mass, model.mass, model.inertia])
end

function dynamics_bias(model::CYBERTRUCK, q, q̇) 
    [0.0; 0.0; 0.0]
end

function input_jacobian(model::CYBERTRUCK, q)
	[
        cos(q[3]) sin(q[3]) 0.0; 
        0.0       0.0       1.0;
    ]
end

function contact_jacobian(model::CYBERTRUCK, q)
	[
        1.0 0.0 0.0; 
        0.0 1.0 0.0;
    ]
end

# nominal configuration 
function nominal_configuration(model::CYBERTRUCK)
	[0.0; 0.0; 0.0]
end

# friction coefficients 
friction_coefficients(model::CYBERTRUCK) = model.friction_body_world

function dynamics(model, mass_matrix, dynamics_bias, h, q0, q1, u1, w1, λ1, q2)
    # evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = RoboDojo.lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
    D1L2, D2L2 = RoboDojo.lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    d = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2 # variational integrator (midpoint)
    d .+= transpose(input_jacobian(model, qm2)) * u1        # control inputs
    d .+= λ1                                                # contact impulses

    return d
end

# Dimensions
nq = 3 # configuration dimension
nu = 2 # control dimension
nw = 0 # parameters
nc = 1 # number of contact points

# Parameters
body_mass = 1.0
body_inertia = 0.1
friction_body_world = [0.5]  # coefficient of friction

# Model
cybertruck = CYBERTRUCK(nq, nu, nw, nc,
		  body_mass, body_inertia,
		  friction_body_world, zeros(0))

# Dimensions 
nx = 2 * nq
nu = 2 + 6

function dynamics(model::CYBERTRUCK, h, y, x, u, w)
    
    # configurations
    q1⁻ = x[1:3]
    q2⁻ = x[3 .+ (1:3)]
    q2⁺ = y[1:3]
    q3⁺ = y[3 .+ (1:3)]

    # control
    u_control = u[1:2]

    # friction
    β = u[2 .+ (1:3)] 
    b = β[2:3]

    # contact impulses
    J = contact_jacobian(model, q2⁺)
    λ = transpose(J) * b 

    [
        q2⁺ - q2⁻;
        dynamics(model, q -> mass_matrix(model, q), (q, q̇) -> dynamics_bias(model, q, q̇),
            h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺);
    ]
end

function contact_equality(model, h, x, u, w)
    # configurations
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]

    # friction primals and duals
    β = u[2 .+ (1:3)] 
    η = u[2 + 3 .+ (1:3)] 

    # friction coefficient
    μ = model.friction_body_world[1]
   
    v = (q3 - q2) ./ h[1]

    [
        β[1] - μ * model.mass * 9.81 * h[1];
        v[1:2] - η[2:3];
        second_order_product(β, η);
    ]
end

# ## horizon
T = 26
h = 0.1

## model
dt = CALIPSO.Dynamics((y, x, u, w) -> dynamics(cybertruck, [h], y, x, u, w), nx, nx, nu)
dyn = [dt for t = 1:T-1]

# ## initial conditions

# Initial 
x1 = [0.0; 2.0; -0.5 * π; 0.0; 1.0; -0.5 * π] 

# Terminal 
xT = [3.0; 0.0; 0.5 * π; 3.0; 0.0; 0.5 * π]

# x1 = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 

# # Terminal 
# xT = [1.0; 1.0; 0.0; 1.0; 1.0; 0.0]


# ## objective
function obj1(x, u, w)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ h[1]
    J += 0.5 * 1.0e-5 * dot(v, v)
    # J += 0.5 * transpose(x - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - xT)
    # J += 0.5 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6)]) * u
    return J
end

function objt(x, u, w)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ h[1]
    J += 0.5 * 1.0e-5 * dot(v, v)
    # J += 0.5 * transpose(x - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - xT)
    # J += 0.5 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6)]) * u
    return J
end

function objT(x, u, w)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ h[1]
    J += 0.5 * 1.0e-5 * dot(v, v)
    # J += 0.5 * 1.0 * transpose(x - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - xT)
    return J
end
c1 = CALIPSO.Cost(obj1, nx, nu)
ct = CALIPSO.Cost(objt, nx, nu)
cT = CALIPSO.Cost(objT, nx, 0)
obj = [c1, [ct for t = 2:T-1]..., cT];

# ## constraints
function equality_1(x, u, w)
    [
        contact_equality(cybertruck, h, x, u, w);
        x - x1;
    ]
end
function equality_t(x, u, w)
    [
        contact_equality(cybertruck, h, x, u, w);
    ]
end
function equality_T(x, u, w)
    # zeros(0)
    [
        x - xT;
    ]
end

eq1 = CALIPSO.Constraint(equality_1, nx, nu)
eqt = CALIPSO.Constraint(equality_t, nx, nu)
eqT = CALIPSO.Constraint(equality_T, nx, 0)
eq = [eq1, [eqt for t = 2:T-1]..., eqT];

u_min = [0.0; -0.5]
u_max = [1.0;  0.5]
p_car1 = [3.0, 2.0 * 0.65]
p_car2 = [3.0, 2.0 * -0.65]
circle_obstacle(x, p; r=0.5) = (x[1] - p[1])^2.0 + (x[2] - p[2])^2.0 - r^2.0
ineq = [[Constraint(
            (x, u, w) -> [
                u_max - u[1:2]; 
                u[1:2] - u_min;
                # circle_obstacle(x, p_car1, r=0.1); 
                # circle_obstacle(x, p_car2, r=0.1);
            ], nx, nu) for t = 1:T-1]..., Constraint()]

soc = [[[Constraint((x, u, w) -> u[2 .+ (1:3)], nx, nu), Constraint((x, u, w) -> u[2 + 3 .+ (1:3)], nx, nu)] for t = 1:T-1]..., [Constraint()]]
# soc = [[Constraint()] for t = 1:T]

# ## initialize
x_guess = linear_interpolation(x1, xT, T)
u_guess = [[1.0e-3 * randn(2); 1.0e-5 * ones(6)] for t = 1:T-1] # may need to run more than once to get good trajectory

# ## problem
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc);
methods = ProblemMethods(trajopt);
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# ## solver
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    options=Options(verbose=true, penalty_initial=1.0, residual_tolerance=1.0e-4));
initialize_states!(solver, trajopt, x_guess);
initialize_controls!(solver, trajopt, u_guess);

# solve
solve!(solver)

x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)

# plot
using Plots
plot(hcat(x_sol...)[4:6, :]', label="")
plot(hcat(u_sol...)[1:2, :]', label=["v" "ω"])

# meshes
vis = Visualizer()
open(vis)

function set_background!(vis::Visualizer; top_color=RGBA(1,1,1.0), bottom_color=RGBA(1,1,1.0))
    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize!(vis, model::CYBERTRUCK, q;
    scale=0.1,
    Δt = 0.1)

    # default_background!(vis)
    path_meshes = joinpath(@__DIR__, "..", "..", "robot_meshes")
    meshfile = joinpath(path_meshes, "cybertruck", "cybertruck.obj")
    obj = RoboDojo.MeshCat.MeshFileObject(meshfile);
    
    RoboDojo.MeshCat.setobject!(vis["cybertruck"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    anim = RoboDojo.MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

    for t = 1:length(q)
        RoboDojo.MeshCat.atframe(anim, t) do
            RoboDojo.MeshCat.settransform!(vis["cybertruck"], RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(q[t][1:2]..., 0.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(q[t][3]))))
        end
    end

    RoboDojo.MeshCat.settransform!(vis["/Cameras/default"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(2.0, 0.0, 1.0),RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.0))))
    
    # # add parked cars
    
    # RoboDojo.MeshCat.setobject!(vis["cybertruck_park1"], obj)
    # RoboDojo.MeshCat.settransform!(vis["cybertruck_park1"]["mesh"],
    #     RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car1...; 0.0]),
    #     RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(pi + pi / 2) * RoboDojo.Rotations.RotX(pi / 2.0))))

    # RoboDojo.MeshCat.setobject!(vis["cybertruck_park2"], obj)
    # RoboDojo.MeshCat.settransform!(vis["cybertruck_park2"]["mesh"],
    #     RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car2...; 0.0]),
    #     RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(pi + pi / 2) * RoboDojo.Rotations.RotX(pi / 2.0))))

    RoboDojo.MeshCat.setanimation!(vis, anim)
end

visualize!(vis, cybertruck, x_sol, Δt=h)
set_background!(vis)
