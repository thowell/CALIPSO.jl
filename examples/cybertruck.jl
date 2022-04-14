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
    Cybertruck
"""
struct Cybertruck{T} <: RoboDojo.Model{T}
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

function mass_matrix(model::Cybertruck, q) 
    Diagonal([model.mass, model.mass, model.inertia])
end

function dynamics_bias(model::Cybertruck, q, q̇) 
    [0.0; 0.0; 0.0]
end

function input_jacobian(model::Cybertruck, q)
	[
        cos(q[3]) sin(q[3]) 0.0; 
        0.0 0.0 1.0;
    ]
end

function contact_jacobian(model::Cybertruck, q)
	[
        1.0 0.0 0.0; 
        0.0 1.0 0.0;
    ]
end

# nominal configuration 
function nominal_configuration(model::Cybertruck)
	[0.0; 0.0; 0.0]
end

# friction coefficients 
friction_coefficients(model::Cybertruck) = model.friction_body_world

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
cybertruck = Cybertruck(nq, nu, nw, nc,
		  body_mass, body_inertia,
		  friction_body_world, zeros(0))

# cybertruck_contact_kinematics = [
#     q -> q[1:2],
# ]
    
# cybertruck_contact_kinematics_jacobians = [
#     q -> [1.0 0.0 0.0; 0.0 1.0 0.0],
# ]

# name(::Cybertruck) = :cybertruck
# floating_base_dim(::Cybertruck) = 3

# Dimensions 
nx = 2 * nq
nu = 2 + 6

function dynamics(model::Cybertruck, h, y, x, u, w)
    
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
        β[1] - μ * model.mass * 9.81;
        v[1:2] - η[2:3];
        second_order_product(β, η);
    ]
end

## model
dt = CALIPSO.Dynamics((y, x, u, w) -> dynamics(cybertruck, [h], y, x, u, w), nx, nx, nu)

dyn = [dt for t = 1:T-1]


# ## horizon
T = 11
h = 0.1

# ## initial conditions

# Initial 
x1 = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0] 

# Terminal 
xT = [1.0; 0.0; 0.0; 1.0; 0.0; 0.0]

# ## objective
function obj1(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - xT)
    J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(2); 1.0e-5 * ones(6)]) * u
    return J
end

function objt(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x[1:nx] - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - xT)
    J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(2); 1.0e-5 * ones(6)]) * u
    return J
end
function objT(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x[1:nx] - xT) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - xT)
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
    return zeros(0)
    [
        x - xT;
    ]
end

eq1 = CALIPSO.Constraint(equality_1, nx, nu)
eqt = CALIPSO.Constraint(equality_t, nx, nu)
eqT = CALIPSO.Constraint(equality_T, nx, 0)
eq = [eq1, [eqt for t = 2:T-1]..., eqT];

ineq = [Constraint() for t = 1:T];

soc = [[[Constraint((x, u, w) -> u[2 .+ (1:3)], nx, nu), Constraint((x, u, w) -> u[2 + 3 .+ (1:3)], nx, nu)] for t = 1:T-1]..., [Constraint()]]

# ## initialize
x_guess = linear_interpolation(x1, xT, T)
u_guess = [[1.0 * randn(2); 1.0e-3 * ones(6)] for t = 1:T-1] # may need to run more than once to get good trajectory

# ## problem
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc);
methods = ProblemMethods(trajopt);
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# ## solver
solver = Solver(methods, trajopt.num_variables, trajopt.num_equality, trajopt.num_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    options=Options(verbose=true));
initialize_states!(solver, trajopt, x_guess);
initialize_controls!(solver, trajopt, u_guess);

# solve
solve!(solver)

x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)

using Plots 

plot(hcat(x_sol...)[4:6, :]', label="")