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
const RD = RoboDojo

function quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
    model = RoboDojo.quadruped

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
    β = u[nu + 4 .+ (1:8)] 
   
    E = [1.0 -1.0] # friction mapping 
    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J[1:8, :]) * [
                                [E * β[1:2]; γ[1]];
                                [E * β[3:4]; γ[2]];
                                [E * β[5:6]; γ[3]];
                                [E * β[7:8]; γ[4]];
                               ]

    [
        q2⁺ - q2⁻;
        RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
            h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end

function quadruped_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    nx = 2 * model.nq
    nu = model.nu
    nc = 4
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[nx .+ (1:nc)] - u[nu .+ (1:nc)];
        y[nx + nc .+ (1:nx)] - x[1:nx];
    ]
end

function quadruped_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
	model = RoboDojo.quadruped
    nx = 2 * model.nq
    nu = model.nu
    nc = 4
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[nx .+ (1:nc)] - u[nu .+ (1:nc)];
        y[nx + nc .+ (1:nx)] - x[nx + nc .+ (1:nx)];
    ]
end

function contact_constraints_inequality_1(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     ϕ;
     fc;
    ]
end

function contact_constraints_inequality_t(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
   
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
     ϕ;
     fc;
    ]
end

function contact_constraints_inequality_T(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 


    ϕ = RoboDojo.signed_distance(model, q3)[1:4]

    [
     ϕ;
    ]
end

function contact_constraints_equality_1(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
   
    v = (q3 - q2) ./ h[1]
    E = [1.0; -1.0]
    vT = vcat([E * (RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)
    
    return [
            η - vT - ψ_stack;
            β .* η;
            ψ .* fc;
    ]
end

function contact_constraints_equality_t(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    u_control = u[1:nu] 
    γ = u[nu .+ (1:4)] 
    β = u[nu + 4 .+ (1:8)] 
    ψ = u[nu + 4 + 8 .+ (1:4)] 
    η = u[nu + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[nx .+ (1:4)] 

    v = (q3 - q2) ./ h[1]
    E = [1.0; -1.0]
    vT = vcat([E * (RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
    ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)
    
    return [
        η - vT - ψ_stack;
        γ⁻ .* ϕ;
        β .* η; 
        ψ .* fc;
    ]
end

function contact_constraints_equality_T(h, x, u, w) 
    model = RoboDojo.quadruped

    nq = model.nq
    nu = model.nu 

    q2 = x[1:nq] 
    q3 = x[nq .+ (1:nq)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[nx .+ (1:4)] 

    return [
        γ⁻ .* ϕ;
    ]
end

function initial_configuration(model::RoboDojo.Quadruped, θ1, θ2, θ3)
    q1 = zeros(model.nq)
    q1[3] = 0.0
    q1[4] = -θ1
    q1[5] = θ2

    q1[8] = -θ1
    q1[9] = θ2

    q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

    q1[10] = -θ3
    q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

    q1[6] = -θ3
    q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

    return q1
end

# ## quadruped 
nc = 4
nq = RD.quadruped.nq
nx = 2 * nq
nu = RD.quadruped.nu + nc + 8 + nc + 8
nw = RD.quadruped.nw

# ## time 
T = 26
h = 0.1

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(RD.quadruped, θ1, θ2, θ3)
q1[2] += 1.0#
qT = initial_configuration(RD.quadruped, θ1, θ2, θ3)

# ## model
mass_matrix, dynamics_bias = RD.codegen_dynamics(RD.quadruped)
d1 = CALIPSO.Dynamics((y, x, u, w) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + nx, nx, nu)
dt = CALIPSO.Dynamics((y, x, u, w) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc + nx, nx + nc + nx, nu)
dyn = [d1, [dt for t = 2:T-1]...]

# ## objective
obj = CALIPSO.Cost{Float64}[]

function obj1(x, u, w)
    u_ctrl = u[1:RD.quadruped.nu]
    q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

	J = 0.0 
    J += 1.0e-2 * dot(u_ctrl, u_ctrl)
    J += 1.0 * dot(q - qT, q - qT)
    return J
end
push!(obj, CALIPSO.Cost(obj1, nx, nu))

for t = 2:T-1
    function objt(x, u, w)
        u_ctrl = u[1:RD.quadruped.nu]
        q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

        J = 0.0 
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end
    push!(obj, CALIPSO.Cost(objt, nx + nc + nx, nu))
end

function objT(x, u, w)
    q = x[RD.quadruped.nq .+ (1:RD.quadruped.nq)]

	J = 0.0 
    J += 1.0e-3 * dot(q - qT, q - qT)

    return J
end
push!(obj, CALIPSO.Cost(objT, nx + nc + nx, 0))

# control limits
eq = CALIPSO.Constraint{Float64}[]
function equality_1(x, u, w) 
    nq = RD.quadruped.nq
    [
     x[1:nq] - q1;
     x[nq .+ (1:nq)] - q1;
     contact_constraints_equality_1(h, x, u, w); 
    ]
end
push!(eq, CALIPSO.Constraint(equality_1, nx, nu))

for t = 2:T-1
    function equality_t(x, u, w) 
        [
        contact_constraints_equality_t(h, x, u, w); 
        ]
    end
    push!(eq, CALIPSO.Constraint(equality_t, nx + nc + nx, nu))
end

function equality_T(x, u, w) 
    [
    contact_constraints_equality_T(h, x, u, w); 
    ]
end
push!(eq, CALIPSO.Constraint(equality_T, nx + nc + nx, nu))

ineq = CALIPSO.Constraint{Float64}[]
function inequality_1(x, u, w) 
    [
     contact_constraints_inequality_1(h, x, u, w);
     u[RD.quadruped.nu .+ (1:(nu - RD.quadruped.nu))];
    ]
end
push!(ineq, CALIPSO.Constraint(inequality_1, nx, nu))

for t = 2:T-1
    function inequality_t(x, u, w) 
        [
        contact_constraints_inequality_t(h, x, u, w);
        u[RD.quadruped.nu .+ (1:(nu - RD.quadruped.nu))];
        ]
    end
    push!(ineq, CALIPSO.Constraint(inequality_t, nx + nc + nx, nu))
end

function inequality_T(x, u, w) 
    [
     contact_constraints_inequality_T(h, x, u, w);
    ]
end
push!(ineq, CALIPSO.Constraint(inequality_T, nx + nc + nx, nu))

# ## initialize
q_interp = CALIPSO.linear_interpolation(q1, qT, T+1)
x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:T]
u_guess = [max.(0.0, 1.0e-2 * randn(nu)) for t = 1:T-1] # may need to run more than once to get good trajectory
x_guess = [t == 1 ? x_interp[1] : [x_interp[t]; max.(0.0, 1.0e-2 * randn(nc)); x_interp[1]] for t = 1:T]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq);
methods = ProblemMethods(trajopt);

# solver
solver = Solver(methods, trajopt.num_variables, trajopt.num_equality, trajopt.num_inequality,
    options=Options(verbose=true));
initialize_states!(solver, trajopt, x_guess);
initialize_controls!(solver, trajopt, u_guess);

# solve 
solve!(solver)

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)

norm(solver.data.residual, Inf) < 1.0e-3
norm(solver.problem.equality_constraint, Inf) < 1.0e-3 
norm(solver.variables[solver.indices.inequality_slack] .* solver.variables[solver.indices.inequality_slack_dual], Inf) < 1.0e-3 

# ## visualize 
vis = Visualizer() 
render(vis)
RoboDojo.visualize!(vis, RoboDojo.quadruped, x_sol, Δt=h);