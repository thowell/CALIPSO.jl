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

function box_dyn(mass_matrix, dynamics_bias, h, y, x, u, w)
    model = RoboDojo.box
    
    # configurations
    q1⁻ = x[1:3]
    q2⁻ = x[3 .+ (1:3)]
    q2⁺ = y[1:3]
    q3⁺ = y[3 .+ (1:3)]

    # control
    u_control = u[1:3]
    γ = u[3 .+ (1:4)]
    β = u[3 + 4 .+ (1:8)]
    E = [1.0 -1.0] # friction mapping
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J) * [[E * β[1:2]; γ[1]];
                        [E * β[3:4]; γ[2]];
                        [E * β[5:6]; γ[3]];
                        [E * β[7:8]; γ[4]]]
    [
    q2⁺ - q2⁻;
    RoboDojo.dynamics(model, mass_matrix, dynamics_bias,
        h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end
function box_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
    box_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
    y[6 .+ (1:4)] - u[3 .+ (1:4)];
    y[6 + 4 .+ (1:6)] - x;
    ]
end
function box_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
    box_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
    y[6 .+ (1:4)] - u[3 .+ (1:4)];
    y[6 + 4 .+ (1:6)] - x[6 + 4 .+ (1:6)];
    ]
end
function contact_constraints_inequality_1(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    γ = u[3 .+ (1:4)]
    β = u[3 + 4 .+ (1:8)]
    ϕ = RoboDojo.signed_distance(model, q3)
    μ = model.friction_body_world
    fc = μ .* γ[1:4] - [sum(β[1:2]); sum(β[3:4]); sum(β[5:6]); sum(β[7:8])]
    [
    ϕ;
    fc;
    ]
end
function contact_constraints_inequality_t(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    γ = u[3 .+ (1:4)]
    β = u[3 + 4 .+ (1:8)]
    ϕ = RoboDojo.signed_distance(model, q3)
    μ = model.friction_body_world
    fc = μ .* γ[1:4] - [sum(β[1:2]); sum(β[3:4]); sum(β[5:6]); sum(β[7:8])]
    [
    ϕ;
    fc;
    ]
end

function contact_constraints_inequality_T(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    ϕ = RoboDojo.signed_distance(model, q3)
    [
    ϕ;
    ]
end

function contact_constraints_equality_1(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    γ = u[3 .+ (1:4)]
    β = u[3 + 4 .+ (1:8)]
    ψ = u[3 + 4 + 8 .+ (1:4)]
    η = u[3 + 4 + 8 + 4 .+ (1:8)]
    μ = model.friction_body_world
    fc = μ .* γ[1:4] - [sum(β[1:2]); sum(β[3:4]); sum(β[5:6]); sum(β[7:8])]
    v = (q3 - q2) ./ h[1]
    E = [1.0; -1.0]
    vT = vcat([E * (RoboDojo.box_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    [
    η - vT - ψ_stack;
    β .* η;
    ψ .* fc;
    ]
end

function contact_constraints_equality_t(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    γ = u[3 .+ (1:4)]
    β = u[3 + 4 .+ (1:8)]
    ψ = u[3 + 4 + 8 .+ (1:4)]
    η = u[3 + 4 + 8 + 4 .+ (1:8)]
    ϕ = RoboDojo.signed_distance(model, q3)
    γ⁻ = x[nx .+ (1:4)]
    μ = model.friction_body_world
    fc = μ .* γ[1:4] - [sum(β[1:2]); sum(β[3:4]); sum(β[5:6]); sum(β[7:8])]
    v = (q3 - q2) ./ h[1]
    E = [1.0; -1.0]
    vT = vcat([E * (RoboDojo.box_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    [
    η - vT - ψ_stack;
    γ⁻ .* ϕ; 
    β .* η; 
    ψ .* fc; 
    ]
end
function contact_constraints_equality_T(h, x, u, w)
    model = RoboDojo.box
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]
    ϕ = RoboDojo.signed_distance(model, q3)
    γ⁻ = x[6 .+ (1:4)]
    [
    γ⁻ .* ϕ;
    ]
end

# ## horizon
T = 11
h = 0.1

# ## box
nx = 2 * RoboDojo.box.nq
nu = RoboDojo.box.nu + 4 + 8 + 4 + 8

## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.box)
d1 = CALIPSO.Dynamics((y, x, u, w) -> box_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 4, nx, nu)
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.box)
dt = CALIPSO.Dynamics((y, x, u, w) -> box_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 4, 2 * nx + 4, nu)

dyn = [d1, [dt for t = 2:T-1]...];

# ## initial conditions
q1 = [0.0; 0.5; 0.0]
qM = [0.0; 0.5; 0.0]
# qT = [0.0; 0.5 + 1.0; 0.0]
qT = [1.0; 0.5; 0.0]

q_ref = qT
x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## objective
function obj1(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - x_ref)
    J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(RoboDojo.box.nu); 1.0e-5 * ones(nu - RoboDojo.box.nu)]) * u
    return J
end
function objt(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(RoboDojo.box.nu); 1.0e-5 * ones(nu - RoboDojo.box.nu)]) * u
    return J
end
function objT(x, u, w)
    J = 0.0
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    return J
end
c1 = CALIPSO.Cost(obj1, nx, nu)
ct = CALIPSO.Cost(objt, 2 * nx + 4, nu)
cT = CALIPSO.Cost(objT, 2 * nx + 4, 0)
obj = [c1, [ct for t = 2:T-1]..., cT];

# ## constraints
function equality_1(x, u, w)
    [
    contact_constraints_equality_1(h, x, u, w);
    x - x1;
    ]
end
function equality_t(x, u, w)
    [
    contact_constraints_equality_t(h, x, u, w);
    ]
end
function equality_T(x, u, w)
    [
    contact_constraints_equality_T(h, x, u, w);
    x[1:6] - xT;
    ]
end
eq1 = CALIPSO.Constraint(equality_1, nx, nu)
eqt = CALIPSO.Constraint(equality_t, 2nx + 4, nu)
eqT = CALIPSO.Constraint(equality_T, 2nx + 4, 0)
eq = [eq1, [eqt for t = 2:T-1]..., eqT];
function inequality_1(x, u, w)
    [
    contact_constraints_inequality_1(h, x, u, w);
    u - [-10.0; -10.0; zeros(nu - 2)];
    [10.0; 10.0] - u[1:2];
    ]
end
function inequality_t(x, u, w)
    [
    contact_constraints_inequality_t(h, x, u, w);
    u - [-10.0; -10.0; zeros(nu - 2)];
    [10.0; 10.0] - u[1:2];
    ]
end
function inequality_T(x, u, w)
    [
    contact_constraints_inequality_T(h, x, u, w);
    ]
end
ineq1 = CALIPSO.Constraint(inequality_1, nx, nu)
ineqt = CALIPSO.Constraint(inequality_t, 2nx + 4, nu)
ineqT = CALIPSO.Constraint(inequality_T, 2nx + 4, 0)
ineq = [ineq1, [ineqt for t = 2:T-1]..., ineqT];

soc = [[Constraint()] for t = 1:T] 

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
x_guess = [x_interpolation[1], [[x_interpolation[t]; zeros(4); x_interpolation[t]] for t = 2:T]...]
u_guess = [[0.0 * randn(RoboDojo.box.nu); 1.0e-1 * ones(nu - RoboDojo.box.nu)] for t = 1:T-1] # may need to run more than once to get good trajectory

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

@test norm(solver.data.residual, Inf) < 1.0e-3
@test norm(solver.problem.equality_constraint, Inf) < 1.0e-3
@test norm(solver.variables[solver.indices.inequality_slack] .* solver.variables[solver.indices.inequality_slack_dual], Inf) < 1.0e-3

# ## visualize
vis = Visualizer()
render(vis)
RoboDojo.visualize!(vis, RoboDojo.box, x_sol, Δt=h, r=0.5);

