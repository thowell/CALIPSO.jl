# @testset "Examples: Hopper gait" begin
function hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.hopper

    # dimensions
    nq = model.nq
    nu = model.nu 

    # configurations
    
    q1⁻ = x[1:nq] 
    q2⁻ = x[nq .+ (1:nq)]
    q2⁺ = y[1:nq]
    q3⁺ = y[nq .+ (1:nq)]

    # control 
    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    
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
    nx = 8 
    [
    hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
    y[nx .+ (1:5)] - [u[2 .+ (1:4)]; u[end]];
    y[nx + 5 .+ (1:nx)] - x
    ]
end

function hopper_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
    nx = 8
    [
    hopper_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
    y[nx .+ (1:5)] - [u[2 .+ (1:4)]; u[end]];
    y[nx + 5 .+ (1:nx)] - x[nx + 5 .+ (1:nx)]
    ]
end

function contact_constraints_inequality_1(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    # sα = u[nu + 4 + 4 + 2 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 

    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
    ϕ; 
    fc;
    #  β .* η .- sα;
    #  ψ .* fc  .- sα;
    ]
end

function contact_constraints_inequality_t(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    # sα = u[nu + 4 + 4 + 2 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[nx .+ (1:4)] 
    # sα⁻ = x[nx + 4 .+ (1:1)]
    
    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    [
    ϕ; 
    fc;
    #  γ⁻ .* ϕ .- sα⁻;
    #  β .* η .- sα;
    #  ψ .* fc  .- sα;
    ]
end

function contact_constraints_inequality_T(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[nx .+ (1:4)] 
    # sα⁻ = x[nx + 4 .+ (1:1)]

    [
    ϕ; 
    #  γ⁻ .* ϕ .- sα⁻;
    ]
end


function contact_constraints_equality_1(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    # sα = u[nu + 4 + 4 + 2 + 4 .+ (1:1)]

    # ϕ = RoboDojo.signed_distance(model, q3) 

    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    [
    #  -ϕ; 
    #  -fc;
    η - vT - ψ_stack;
    β .* η# .- sα;
    ψ .* fc#  .- sα;
    ]
end

function contact_constraints_equality_t(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nu = model.nu 
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 
    # sα = u[nu + 4 + 4 + 2 + 4 .+ (1:1)]

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[nx .+ (1:4)] 
    # sα⁻ = x[nx + 4 .+ (1:1)]
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:4)] 
    ψ = u[nu + 4 + 4 .+ (1:2)] 
    η = u[nu + 4 + 4 + 2 .+ (1:4)] 

    μ = [model.friction_body_world; model.friction_foot_world]
    fc = μ .* γ[1:2] - [sum(β[1:2]); sum(β[3:4])]

    v = (q3 - q2) ./ h[1]
    vT_body = v[1] + model.body_radius * v[3]
    vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]
    
    [
    #  -ϕ; 
    #  -fc;
    η - vT - ψ_stack;
    γ⁻ .* ϕ #.- sα⁻;
    β .* η #.- sα;
    ψ .* fc # .- sα;
    ]
end

function contact_constraints_equality_T(h, x, u, w) 
    model = RoboDojo.hopper

    nq = model.nq
    nx = 2nq

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    ϕ = RoboDojo.signed_distance(model, q3) 
    γ⁻ = x[nx .+ (1:4)] 
    # sα⁻ = x[nx + 4 .+ (1:1)]
    # γ = u[nu .+ (1:4)] 
    # β = u[nu + 4 .+ (1:4)] 
    # ψ = u[nu + 4 + 4 .+ (1:2)] 
    # η = u[nu + 4 + 4 + 2 .+ (1:4)] 

    # v = (q3 - q2) ./ h[1]
    # vT_body = v[1] + model.body_radius * v[3]
    # vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
    # vT = [vT_body; -vT_body; vT_foot; -vT_foot]
    
    # ψ_stack = [ψ[1] * ones(2); ψ[2] * ones(2)]

    [
    # η - vT - ψ_stack;
    #  -ϕ; 
    γ⁻ .* ϕ;# .- sα⁻;
    ]
end

# ## horizon 
T = 21 
h = 0.05

# ## hopper 
nx = 2 * RoboDojo.hopper.nq
nu = RoboDojo.hopper.nu + 4 + 4 + 2 + 4 #+ 1

# ## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.hopper)
d1 = CALIPSO.Dynamics((y, x, u, w) -> hopper_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 5, nx, nu)
dt = CALIPSO.Dynamics((y, x, u, w) -> hopper_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), 2 * nx + 5, 2 * nx + 5, nu)

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
    # J += 1000.0 * u[nu]
    return J
end

function objt(x, u, w)
    J = 0.0 
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal(q_cost * [1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0]) * (x[1:nx] - x_ref)
    J += 0.5 * transpose(u) * Diagonal([r_cost * ones(RoboDojo.hopper.nu); zeros(nu - RoboDojo.hopper.nu)]) * u
    # J += 1000.0 * u[nu]
    return J
end

function objT(x, u, w)
    J = 0.0 
    J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
    return J
end

c1 = CALIPSO.Cost(obj1, nx, nu)
ct = CALIPSO.Cost(objt, 2 * nx + 5, nu)
cT = CALIPSO.Cost(objT, 2 * nx + 5, 0)
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
    #  # inequality (12)
    #  contact_constraints_inequality_1(h, x, u, w);
    #  # + 17 + 2 inequality 
    #  [-10.0; -10.0; zeros(nu - 2)] - u; 
    #  u[1:2] - [10.0; 10.0];
    #  # + 6 state bounds 
    #  -x[2];
    #  -x[4];
    #  -x[6];
    #  -x[8];
    #  x[4] - 1.0; 
    #  x[8] - 1.0;
    ]
end

function equality_t(x, u, w) 
    [
    # equality (4)
    contact_constraints_equality_t(h, x, u, w); 
    #  # inequality (16)
    #  contact_constraints_inequality_t(h, x, u, w);
    # # + 17 + 2 inequality 
    # ul - u; 
    # u[1:2] - [10.0; 10.0];
    #  # + 6 state bounds 
    #  -x[2];
    #  -x[4];
    #  -x[6];
    #  -x[8];
    #  x[4] - 1.0; 
    #  x[8] - 1.0;
    ]
end

function equality_T(x, u, w) 
    x_travel = 0.5
    θ = x[nx + 5 .+ (1:nx)]
    [
    contact_constraints_equality_T(h, x, u, w); 
    # equality (6)
    x[1:RoboDojo.hopper.nq][collect([2, 3, 4])] - θ[1:RoboDojo.hopper.nq][collect([2, 3, 4])];
    x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])] - θ[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])];
    #  # inequality (10)
    #  x_travel - (x[1] - θ[1])
    #  x_travel - (x[RoboDojo.hopper.nq + 1] - θ[RoboDojo.hopper.nq + 1])
    #  contact_constraints_inequality_T(h, x, u, w);
    #   # + 6 state bounds 
    #   -x[2];
    #   -x[4];
    #   -x[6];
    #   -x[8];
    #   x[4] - 1.0; 
    #   x[8] - 1.0;
    ]
end

eq1 = CALIPSO.Constraint(equality_1, nx, nu)#, idx_ineq=collect(8 + 4 .+ (1:(12 + 19 + 6)))) 
eqt = CALIPSO.Constraint(equality_t, 2nx + 5, nu)#, idx_ineq=collect(4 .+ (1:(16 + 19 + 6)))) 
eqT = CALIPSO.Constraint(equality_T, 2nx + 5, nu)#, idx_ineq=collect(6 .+ (1:(10 + 6)))) 
eq = [eq1, [eqt for t = 2:T-1]..., eqT];

function inequality_1(x, u, w) 
    [
    #  # equality (8)
    #  RoboDojo.kinematics_foot(RoboDojo.hopper, x[1:RoboDojo.hopper.nq]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[1:RoboDojo.hopper.nq]);
    #  RoboDojo.kinematics_foot(RoboDojo.hopper, x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]) - RoboDojo.kinematics_foot(RoboDojo.hopper, x1[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]);
    #  contact_constraints_equality(h, x, u, w); 
    #  # initial condition 4 
    #  x[1:4] - q1;
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
    #  # equality (4)
    #  contact_constraints_equality(h, x, u, w); 
    # inequality (16)
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
    θ = x[nx + 5 .+ (1:nx)]
    [
    #  # equality (6)
    #  x[1:RoboDojo.hopper.nq][collect([2, 3, 4])] - θ[1:RoboDojo.hopper.nq][collect([2, 3, 4])];
    #  x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])] - θ[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])];
    # inequality (10)
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

ineq1 = CALIPSO.Constraint(inequality_1, nx, nu)#, idx_ineq=collect(8 + 4 .+ (1:(12 + 19 + 6)))) 
ineqt = CALIPSO.Constraint(inequality_t, 2nx + 5, nu)#, idx_ineq=collect(4 .+ (1:(16 + 19 + 6)))) 
ineqT = CALIPSO.Constraint(inequality_T, 2nx + 5, nu)#, idx_ineq=collect(6 .+ (1:(10 + 6)))) 
ineq = [ineq1, [ineqt for t = 2:T-1]..., ineqT];


# ## problem 
# p = DTO.solver(dyn, obj, cons, bnds, 
#     options=DTO.Options(
#         tol=1.0e-2,
#         constr_viol_tol=1.0e-2,
#         # print_level=0
#         ))

# ## initialize
x_interpolation = [x1, [[x1; zeros(5); x1] for t = 2:T]...]
u_guess = [[0.0; RoboDojo.hopper.gravity * RoboDojo.hopper.mass_body * 0.5 * h[1]; 1.0e-1 * ones(nu - 2)] for t = 1:T-1] # may need to run more than once to get good trajectory
# DTO.initialize_states!(p, x_interpolation)
# DTO.initialize_controls!(p, u_guess);

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq)
methods = CALIPSO.ProblemMethods(trajopt)

# solver
solver = Solver(methods, trajopt.num_variables, trajopt.num_equality, trajopt.num_inequality,
    options=Options(verbose=true))
initialize_states!(solver, trajopt, x_interpolation)
initialize_controls!(solver, trajopt, u_guess)

# solve 
solve!(solver)

# # ## solve
# @time DTO.solve!(p);

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)

@test norm(solver.data.residual, Inf) < 1.0e-3
@test norm((x_sol[1] - x_sol[T][1:nx])[[2; 3; 4; 6; 7; 8]], Inf) < 1.0e-3
@test norm(solver.problem.equality, Inf) < 1.0e-3 
@test norm(solver.variables[solver.indices.inequality_slack] .* solver.variables[solver.indices.inequality_slack_dual], Inf) < 1.0e-3 

# # # ## visualize 
# # vis = Visualizer() 
# # render(vis)
# # # q_sol = state_to_configuration([x[1:nx] for x in x_sol])
# # RoboDojo.visualize!(vis, RoboDojo.hopper, x_sol, Δt=h);
# # end