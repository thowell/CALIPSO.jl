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

function hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.hopper

    # configurations
    
    q1⁻ = x[1:4] 
    q2⁻ = x[4 .+ (1:4)]
    q2⁺ = y[1:4]
    q3⁺ = y[4 .+ (1:4)]

    # control 
    u_control = u[1:2] 
    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    
    E = [1.0 -1.0] # friction mapping 
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J) * [[E * β[1:2]; γ[1]];
                        [E * β[3:4]; γ[2]];
                            γ[3:4]]
    λ[3] += (model.body_radius * E * β[1:2])[1] # friction on body creates a moment

    [
        q2⁺ - q2⁻;
        RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
        h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end

function hopper_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
        hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[8 .+ (1:4)] - u[2 .+ (1:4)];
        y[8 + 4 .+ (1:8)] - x
    ]
end

function hopper_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
    [
        hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[8 .+ (1:4)] - u[2 .+ (1:4)];
        y[8 + 4 .+ (1:8)] - x[8 + 4 .+ (1:8)]
    ]
end

function contact_constraints_inequality_1(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    u_control = u[1:2] 
    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    ψ = u[2 + 4 + 4 .+ (1:2)] 
    η = u[2 + 4 + 4 + 2 .+ (1:4)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
        ϕ; 
        fc;
    ]
end

function contact_constraints_inequality_t(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    ψ = u[2 + 4 + 4 .+ (1:2)] 
    η = u[2 + 4 + 4 + 2 .+ (1:4)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[8 .+ (1:4)] 
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
        ϕ; 
        fc;
    ]
end

function contact_constraints_inequality_T(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[8 .+ (1:4)] 
    
    [
        ϕ; 
    ]
end


function contact_constraints_equality_1(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    u_control = u[1:2] 
    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    ψ = u[2 + 4 + 4 .+ (1:2)] 
    η = u[2 + 4 + 4 + 2 .+ (1:4)] 
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    [
    η - vT - ψ_stack;
        β .* η;
        ψ .* fc;
    ]
end

function contact_constraints_equality_t(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    ψ = u[2 + 4 + 4 .+ (1:2)] 
    η = u[2 + 4 + 4 + 2 .+ (1:4)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[8 .+ (1:4)] 
    γ = u[2 .+ (1:4)] 
    β = u[2 + 4 .+ (1:4)] 
    ψ = u[2 + 4 + 4 .+ (1:2)] 
    η = u[2 + 4 + 4 + 2 .+ (1:4)] 

    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    
    [
        η - vT - ψ_stack;
        γ⁻ .* ϕ;
        β .* η; 
        ψ .* fc; 
    ]
end

function contact_constraints_equality_T(h, x, u, w) 
    model = RoboDojo.hopper

    q2 = x[1:4] 
    q3 = x[4 .+ (1:4)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[8 .+ (1:4)] 

    [
        γ⁻ .* ϕ;
    ]
end

# ## horizon 
T = 21 
h = 0.05

# ## hopper 
nx = 2 * RoboDojo.hopper.nq
nu = RoboDojo.hopper.nu + 4 + 4 + 2 + 4

# ## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.hopper)
d1 = CALIPSO.Dynamics((y, x, u, w) -> hopper_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 4, nx, nu)
dt = CALIPSO.Dynamics((y, x, u, w) -> hopper_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 4, 2 * nx + 4, nu)

dyn = [d1, [dt for t = 2:T-1]...];

# ## initial conditions
q1 = [0.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + RoboDojo.hopper.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# ## gate 
GAIT = 1 
GAIT = 2 
GAIT = 3

if GAIT == 1 
    r_cost = 1.0e-1 
    q_cost = 1.0e-1
elseif GAIT == 2 
    r_cost = 1.0
    q_cost = 1.0
elseif GAIT == 3 
    r_cost = 1.0e-3
    q_cost = 1.0e-1
end

# ## objective
function obj1(x, u, w)
    J = 0.0 
    J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x - x_ref) 
    J += 0.5 * transpose(u) * Diagonal([r_cost * ones(RoboDojo.hopper.nu); zeros(nu - RoboDojo.hopper.nu)]) * u
    return J
end

function objt(x, u, w)
    J = 0.0 
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal(q_cost * [1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x[1:nx] - x_ref)
    J += 0.5 * transpose(u) * Diagonal([r_cost * ones(RoboDojo.hopper.nu); zeros(nu - RoboDojo.hopper.nu)]) * u
    return J
end

function objT(x, u, w)
    J = 0.0 
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    return J
end

c1 = CALIPSO.Cost(obj1, nx, nu)
ct = CALIPSO.Cost(objt, 2 * nx + 4, nu)
cT = CALIPSO.Cost(objT, 2 * nx + 4, 0)
obj = [c1, [ct for t = 2:T-1]..., cT];

# ## constraints
function equality_1(x, u, w) 
    [
        # equality (8)
        RoboDojo.kinematics_foot(RoboDojo.hopper, x[1:RoboDojo.hopper.nq]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[1:RoboDojo.hopper.nq]);
        RoboDojo.kinematics_foot(RoboDojo.hopper, x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]);
        contact_constraints_equality_1(h, x, u, w); 
        # initial condition 4 
        x[1:4] - q1;
    ]
end

function equality_t(x, u, w) 
    [
        # equality (4)
        contact_constraints_equality_t(h, x, u, w); 
    ]
end

function equality_T(x, u, w) 
    θ = x[8 + 4 .+ (1:8)]
    [
        contact_constraints_equality_T(h, x, u, w); 
        # equality (6)
        x[1:RoboDojo.hopper.nq][collect([2, 3, 4])] - θ[1:RoboDojo.hopper.nq][collect([2, 3, 4])];
        x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])] - θ[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])];
    ]
end

eq1 = CALIPSO.Constraint(equality_1, nx, nu)
eqt = CALIPSO.Constraint(equality_t, 2nx + 4, nu)
eqT = CALIPSO.Constraint(equality_T, 2nx + 4, 0)
eq = [eq1, [eqt for t = 2:T-1]..., eqT];

function inequality_1(x, u, w) 
    [
        # inequality (12)
        contact_constraints_inequality_1(h, x, u, w);
        # + 17 + 2 inequality 
        u - [-10.0; -10.0; zeros(nu - 2)]; 
        [10.0; 10.0] - u[1:2] ;
        # + 6 state bounds 
        x[2];
        x[4];
        x[6];
        x[8];
        1.0 - x[4]; 
        1.0 - x[8];
    ]
end

function inequality_t(x, u, w) 
    [
    # equality (4)
        contact_constraints_inequality_t(h, x, u, w);
        # + 17 + 2 inequality 
        u - [-10.0; -10.0; zeros(nu - 2)]; 
        [10.0; 10.0] - u[1:2];
        # + 6 state bounds 
        x[2];
        x[4];
        x[6];
        x[8];
        1.0 - x[4]; 
        1.0 - x[8];
    ]
end

function inequality_T(x, u, w) 
    x_travel = 0.5
    θ = x[8 + 4 .+ (1:8)]
    [
        (x[1] - θ[1]) - x_travel;
        (x[RoboDojo.hopper.nq + 1] - θ[RoboDojo.hopper.nq + 1]) - x_travel; 
        contact_constraints_inequality_T(h, x, u, w);
        # + 6 state bounds 
        x[2];
        x[4];
        x[6];
        x[8];
        1.0 - x[4]; 
        1.0 - x[8];
    ]
end

ineq1 = CALIPSO.Constraint(inequality_1, nx, nu)
ineqt = CALIPSO.Constraint(inequality_t, 2nx + 4, nu) 
ineqT = CALIPSO.Constraint(inequality_T, 2nx + 4, 0)
ineq = [ineq1, [ineqt for t = 2:T-1]..., ineqT];

soc = [[Constraint()] for t = 1:T]

# ## initialize
x_interpolation = [x1, [[x1; zeros(4); x1] for t = 2:T]...]
u_guess = [[0.0; RoboDojo.hopper.gravity * RoboDojo.hopper.mass_body * 0.5 * h[1]; 1.0e-1 * ones(nu - 2)] for t = 1:T-1] # may need to run more than once to get good trajectory

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc)
methods = ProblemMethods(trajopt)
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# solver
solver = Solver(methods, trajopt.num_variables, trajopt.num_equality, trajopt.num_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    options=Options(verbose=true))
initialize_states!(solver, trajopt, x_interpolation)
initialize_controls!(solver, trajopt, u_guess)

# solve 
solve!(solver)

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)
# ## visualize 
vis = Visualizer() 
render(vis)
RoboDojo.visualize!(vis, RoboDojo.hopper, x_sol, Δt=h);