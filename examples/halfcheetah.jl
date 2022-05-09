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
using JLD2

# ## half cheetah 
model = RoboDojo.halfcheetah4

# ## Initial conditions
q1 = RoboDojo.nominal_configuration(halfcheetah4)
v1 = zeros(halfcheetah4.nq)
# q1[2] -= 0.025
# q1[3] += 0.0 * π

# ## Time
h = 0.1
T = 100

# ## Simulator
s = Simulator(halfcheetah4, T, h=h)

# ## Simulate
# q1[1] += 1
simulate!(s, q1, v1)

# ## Visualizer
vis = Visualizer()
render(vis)

# ## Visualize
visualize!(vis, s)

# ## MPC policy

# ## horizon
horizon = 3
timestep = 0.1

# ## RoboDojo dynamics 
include(joinpath(@__DIR__, "../test/examples/robodojo.jl"))

model = RoboDojo.halfcheetah4
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)
for (i, n) in enumerate(num_actions) 
    num_actions[i] = n - 3 # no slack controls
end

# ## linearized dynamics
function robodojo_linearized_dynamics(sim::RoboDojo.Simulator, z̄, θ̄, y, x, u; 
    indices_linearized=collect(1:(sim.model.nq + sim.model.nc + sim.model.nc)))

    # dimensions
    nz = length(sim.ip.z) 
    nθ = length(sim.ip.θ)

    # indices 
    indices_nonlinear = [i for i = 1:nz if !(i in indices_linearized)]

    # configurations
    q1 = x[1:sim.model.nq]
    q2⁻ = x[sim.model.nq .+ (1:sim.model.nq)] 
    q2⁺ = y[1:sim.model.nq] 

    # residual
    r_nonlinear = zeros(eltype(y), nz)
    z = y[sim.model.nq .+ (1:nz)] 
    θ = zeros(eltype(y), nθ)
    RoboDojo.initialize_θ!(θ, sim.model, sim.idx_θ, q1, q2⁻, [zeros(3); u], sim.dist.w, sim.f, sim.h)
    sim.ip.methods.r!(r_nonlinear, z, θ, [0.0])

    # linearization 
    r̄ = zeros(eltype(y), nz)  
    r̄z = zeros(eltype(y), nz, nz)
    r̄θ = zeros(eltype(y), nz, nθ)
    sim.ip.methods.r!(r̄, z̄, θ̄, [0.0])
    sim.ip.methods.rz!(r̄z, z̄, θ̄)
    sim.ip.methods.rθ!(r̄θ, z̄, θ̄)

    r_linearized = r̄ + r̄z * (z - z̄) + r̄θ * (θ - θ̄)

    [
        # r_nonlinear;
        r_linearized[indices_linearized];
        r_nonlinear[indices_nonlinear];
        q2⁺ - q2⁻;
    ]
end

z̄ = copy(s.ip.z)
θ̄ = copy(s.ip.θ)
dynamics = [(y, x, u, w) -> robodojo_linearized_dynamics(sim, z̄, θ̄, y, x, u) for t = 1:horizon-1]

# ## states
q1 = RoboDojo.nominal_configuration(halfcheetah4)
v1 = zeros(halfcheetah4.nq)
state_initial = [q1; q1 - v1 * timestep]

# ## objective
function obj1(x, u, w)
    J = 0.0

    # velocity
    v = (x[sim.model.nq .+ (1:sim.model.nq)] - x[1:sim.model.nq]) ./ sim.h
    J += 0.5 * w[2 * sim.model.nq + 1]^2 * (v[1] - 1.0)^2

    # control
    J += 0.5 * transpose(u) * Diagonal(w[2 * sim.model.nq + 1 .+ (1:6)].^2) * u

    return J
end

function objt(x, u, w)
    J = 0.0 

    # velocity
    v = (x[sim.model.nq .+ (1:sim.model.nq)] - x[1:sim.model.nq]) ./ sim.h
    J += 0.5 * w[1]^2 * (v[1] - 1.0)^2

    # control
    J += 0.5 * transpose(u) * Diagonal(w[1 .+ (1:6)].^2) * u

    return J
end

function objT(x, u, w)
    J = 0.0 

    # velocity
    v = (x[sim.model.nq .+ (1:sim.model.nq)] - x[1:sim.model.nq]) ./ sim.h
    J += 0.5 * w[1]^2 * (v[1] - 1.0)^2

    return J
end

objective = [
        obj1, 
        [objt for t = 2:horizon-1]..., 
        objT,
]

# ## constraints
equality = [
        (x, u, w) -> 1.0 * (x - w[1:(2 * sim.model.nq)]), 
        [empty_constraint for t = 2:horizon-1]..., 
        empty_constraint,
];

control_lower = -5.0 .* sim.h * ones(sim.model.nu - 3)
control_upper = 5.0 .* sim.h * ones(sim.model.nu - 3)
nonnegative_contact = robodojo_nonnegative(sim, horizon; parameters=true);
nonnegative = [t == horizon ? nonnegative_contact[t] : (x, u, w) -> [nonnegative_contact[t](x, u, w); control_upper - u; u - control_lower] for t = 1:horizon]
second_order = robodojo_second_order(sim, horizon; parameters=true);

# ## options 
options = Options(
    constraint_tensor=true,
    residual_tolerance=1.0e-3,
    optimality_tolerance=1.0e-2, 
    equality_tolerance=1.0e-2, 
    complementarity_tolerance=1.0e-2,
    slack_tolerance=1.0e-2,
    update_factorization=true,
)

# ## parameters 
parameters = [[state_initial; 1.0; 1.0 * ones(6)], [[1.0; 1.0 * ones(6)] for t = 2:horizon-1]..., 1.0 * ones(1)]

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    parameters=parameters,
    equality=equality,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options);

# ## initialize
configurations = CALIPSO.linear_interpolation(q1, q1, horizon+1)
state_guess = robodojo_state_initialization(sim, configurations, horizon)
action_guess = [1.0 * ones(sim.model.nu - 3) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_controls!(solver, action_guess)

# ## solve
solve!(solver)

# test solution
using Test
@test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

slack_norm = max(
                norm(solver.data.residual.equality_dual, Inf),
                norm(solver.data.residual.cone_dual, Inf),
)
@test slack_norm < solver.options.slack_tolerance

@test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
@test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver)
norm(x_sol[1] - state_initial, Inf)

function policy(θ, x, goal; 
    verbose=false,
    warmstart=true)

    # initialize
    if !warmstart
        fill!(solver.solution.variables, 0.0)
        state_guess = robodojo_state_initialization(sim, configurations, horizon)
        action_guess = [1.0e-1 * ones(sim.model.nu-3) for t = 1:horizon-1]
        initialize_states!(solver, state_guess) 
        initialize_controls!(solver, action_guess)
    end

    q = x[1:sim.model.nq] 
    v = x[sim.model.nq .+ (1:sim.model.nq)] 

    solver.parameters .= [q - v .* sim.h; q; θ]
    
    # solve
    solver.options.verbose = verbose
    solve!(solver)

    return solver.solution.variables[solver.problem.custom.indices.actions[1]] ./ sim.h
end

function policy_jacobian_state(θ, x, goal; 
    verbose=false,
    optimize=true,
    warmstart=true)

    if optimize
        # initialize
        if !warmstart
            fill!(solver.solution.variables, 0.0)
            state_guess = robodojo_state_initialization(sim, configurations, horizon)
            action_guess = [1.0e-1 * ones(sim.model.nu-3) for t = 1:horizon-1] # may need to run more than once to get good trajectory
            initialize_states!(solver, state_guess) 
            initialize_controls!(solver, action_guess)
        end

        q = x[1:sim.model.nq] 
        v = x[sim.model.nq .+ (1:sim.model.nq)] 

        solver.parameters .= [q - v .* sim.h; q; θ]
        
        # solve
        solver.options.verbose = verbose
        solve!(solver)
    end

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], 1:(2 * sim.model.nq)] ./ sim.h
end

function policy_jacobian_parameters(θ, x, goal; 
    verbose=false,
    optimize=true,
    warmstart=true)

    if optimize
        # initialize
        if !warmstart
            fill!(solver.solution.variables, 0.0)
            state_guess = robodojo_state_initialization(sim, configurations, horizon)
            action_guess = [1.0e-1 * ones(sim.model.nu-3) for t = 1:horizon-1] # may need to run more than once to get good trajectory
            initialize_states!(solver, state_guess) 
            initialize_controls!(solver, action_guess)
        end
        
        q = x[1:sim.model.nq] 
        v = x[sim.model.nq .+ (1:sim.model.nq)] 

        solver.parameters .= [q - v .* sim.h; q; θ]
        
        # solve
        solver.options.verbose = verbose
        solve!(solver)
    end

    return solver.data.solution_sensitivity[solver.problem.custom.indices.actions[1], (2 * sim.model.nq) .+ (1:(solver.dimensions.parameters - (2 * sim.model.nq)))] ./ sim.h
end

policy_parameters = vcat([[1.0e-3; 1.0e-3 * ones(6)], [[1.0e-3; 1.0e-3 * ones(6)] for t = 2:horizon-1]..., 1.0 * ones(1)]...)

policy(policy_parameters, state_initial, nothing)
policy(policy_parameters, state_initial, nothing; warmstart=true)

policy_jacobian_state(policy_parameters, state_initial, nothing)
policy_jacobian_state(policy_parameters, state_initial, nothing; warmstart=true)

policy_jacobian_parameters(policy_parameters, state_initial, nothing)
policy_jacobian_parameters(policy_parameters, state_initial, nothing; warmstart=true)

# ## simulate policy
d = RoboDojo.Simulator(RoboDojo.halfcheetah4, 1, h=timestep)
q1 = nominal_configuration(RoboDojo.halfcheetah4)
v1 = zeros(RoboDojo.halfcheetah4.nq)
x_hist = [[q1; q1 - v1 * timestep]]
u_hist = Vector{Float64}[]

T_sim = 31
for t = 1:T_sim
    pt = policy(policy_parameters, x_hist[end], nothing;
        verbose=false,
        warmstart=true)
    push!(u_hist, 
            [
                zeros(3); 
                pt;
            ])
    y = zeros(2 * halfcheetah4.nq)
    RoboDojo.dynamics(d, y, x_hist[end], u_hist[end], zeros(0))
    push!(x_hist, y)
end

s = RoboDojo.Simulator(RoboDojo.halfcheetah4, T_sim-1, h=timestep)
for i = 1:T_sim
    q = x_hist[i][1:halfcheetah4.nq]
    v = x_hist[i][halfcheetah4.nq .+ (1:halfcheetah4.nq)]
    RoboDojo.set_state!(s, q, v, i)
end
RoboDojo.visualize!(vis, s)

vx = [x[halfcheetah4.nq .+ (1:halfcheetah4.nq)][1] for x in x_hist]

########## TRAINING
using IterativeLQR
using RoboDojo
using Plots
using Symbolics
using BenchmarkTools
using LinearAlgebra
using FiniteDiff

const iLQR = IterativeLQR
const RD = RoboDojo

vis = Visualizer()
render(vis)

################################################################################
# Simulation
################################################################################
# ## Initial conditions
q1 = nominal_configuration(RoboDojo.halfcheetah4)
v1 = zeros(RoboDojo.halfcheetah4.nq)

# ## Time
h = 0.1
timestep = h
T = 11

# ## Simulator
s = Simulator(RoboDojo.halfcheetah4, T, h=h)
# s.ip.opts.r_tol = 1e-5
# s.ip.opts.κ_tol = 3e-2
# s.ip.opts.undercut = Inf
# ## Simulate
RD.simulate!(s, q1, v1)
# ## Visualize
RD.visualize!(vis, s)

################################################################################
# Dynamics Model
################################################################################
dynamics_model = Simulator(RoboDojo.halfcheetah4, 1, h=h)
# dynamics_model.ip.opts.r_tol = 1e-5
# dynamics_model.ip.opts.κ_tol = 1e-4
# dynamics_model.ip.opts.undercut = 5.0

nq = dynamics_model.model.nq
nx = 2nq
nu = dynamics_model.model.nu
nw = dynamics_model.model.nw


################################################################################
# iLQR
################################################################################
x_hist = [RD.nominal_state(dynamics_model.model)]
for i = 1:11
    y = zeros(nx)
    RD.dynamics(dynamics_model, y, x_hist[end], [0;0;0;0*ones(6)], zeros(nw))
    push!(x_hist, y)
end
# plot(hcat(x_hist...)'[:,1:3])

s = Simulator(RoboDojo.halfcheetah4, 11-1, h=h)
for i = 1:11
    q = x_hist[i][1:nq]
    v = x_hist[i][nq .+ (1:nq)]
    RD.set_state!(s, q, v, i)
end
visualize!(vis, s)

# ## initialization
x1 = deepcopy(x_hist[1])
xT = deepcopy(x_hist[end])
# xT[1] += 0.30
xT[nq+1] += 3.0
RD.set_robot!(vis, dynamics_model.model, x1)
RD.set_robot!(vis, dynamics_model.model, xT)

u_hover = [0; 0; zeros(nu-2)]

# ## (1-layer) multi-layer perceptron policy
# l_input = nx
# l1 = 6
# l2 = nu - 3
# nθ = l1 * l_input + l2 * l1
nθ = length(policy_parameters)

# function policy(θ, x, goal)
#     shift = 0
#     # input
#     input = x - goal

#     # layer 1
#     W1 = reshape(θ[shift .+ (1:(l1 * l_input))], l1, l_input)
#     z1 = W1 * input
#     o1 = tanh.(z1)
#     shift += l1 * l_input

#     # layer 2
#     W2 = reshape(θ[shift .+ (1:(l2 * l1))], l2, l1)
#     z2 = W2 * o1

#     o2 = z2
#     return o2
# end

# function policy_jacobian_state(θ, x, goal) 
#     FiniteDiff.finite_difference_jacobian(a -> policy(θ, a, goal), x)
# end

# function policy_jacobian_parameters(θ, x, goal) 
#     FiniteDiff.finite_difference_jacobian(a -> policy(a, x, goal), θ)
# end

# ## horizon
T = 11

# ## model
h = timestep

function f1(y, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = u[nu .+ (1:nθ)]
    RD.dynamics(dynamics_model, view(y, 1:nx), x_di, u_ctrl, w)
    @views y[nx .+ (1:nθ)] .= θ
    return nothing
end

function f1x(dx, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = u[nu .+ (1:nθ)]
    dx .= 0.0
    RD.dynamics_jacobian_state(dynamics_model, view(dx, 1:nx, 1:nx), x_di, u_ctrl, w)
    return nothing
end

function f1u(du, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = u[nu .+ (1:nθ)]
    du .= 0.0
    RD.dynamics_jacobian_input(dynamics_model, view(du, 1:nx, 1:nu), x_di, u_ctrl, w)
    @views du[nx .+ (1:nθ), nu .+ (1:nθ)] .= I(nθ)
    return nothing
end

function ft(y, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = x[nx .+ (1:nθ)]
    RD.dynamics(dynamics_model, view(y, 1:nx), x_di, u_ctrl, w)
    @views y[nx .+ (1:nθ)] .= θ
    return nothing
end

function ftx(dx, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = x[nx .+ (1:nθ)]
    dx .= 0.0
    RD.dynamics_jacobian_state(dynamics_model, view(dx, 1:nx, 1:nx), x_di, u_ctrl, w)
    @views dx[nx .+ (1:nθ), nx .+ (1:nθ)] .= I(nθ)
    return nothing
end

function ftu(du, x, u, w)
    @views u_ctrl = u[1:nu]
    @views x_di = x[1:nx]
    @views θ = x[nx .+ (1:nθ)]
    RD.dynamics_jacobian_input(dynamics_model, view(du, 1:nx, 1:nu), x_di, u_ctrl, w)
    return nothing
end


# user-provided dynamics and gradients
dyn1 = iLQR.Dynamics(f1, f1x, f1u, nx + nθ, nx, nu + nθ)
dynt = iLQR.Dynamics(ft, ftx, ftu, nx + nθ, nx + nθ, nu)

dyn = [dyn1, [dynt for t = 2:T-1]...]

# ## objective
function o1(x, u, w)
    J = 0.0
    q = 1e-2 * [1e-6; ones(nq-1); 1e+0; ones(nq-1)]
    r = 1e-0 * ones(nu)
    ex = x - xT
    eu = u[1:nu] - u_hover
    J += 0.5 * transpose(ex) * Diagonal(q) * ex
    J += 0.5 * transpose(eu) * Diagonal(r) * eu
    J += 1e-5 * dot(u[nu .+ (1:nθ)] .- 1.0, u[nu .+ (1:nθ)] .- 1.0)
    return J
end

function ot(x, u, w)
    J = 0.0
    q = 1e-2 * [1e-6; ones(nq-1); 1e+0; ones(nq-1)]
    r = 1e-0 * ones(nu)
    ex = x[1:nx] - xT
    eu = u[1:nu] - u_hover
    J += 0.5 * transpose(ex) * Diagonal(q) * ex
    J += 0.5 * transpose(eu) * Diagonal(r) * eu
    J += 1e-5 * dot(x[nx .+ (1:nθ)] .- 1.0, x[nx .+ (1:nθ)] .- 1.0)
    return J
end

function oT(x, u, w)
    J = 0.0
    J += 1e-5 * dot(x[nx .+ (1:nθ)] .- 1.0, x[nx .+ (1:nθ)] .- 1.0)
    return J
end

c1 = iLQR.Cost(o1, nx, nu + nθ)
ct = iLQR.Cost(ot, nx + nθ, nu)
cT = iLQR.Cost(oT, nx + nθ, 0)
obj = [c1, [ct for t = 2:(T - 1)]..., cT]

# ## constraints
ul = -1.0 * [1e0*ones(3); 1e3ones(nu-3)]
uu = +1.0 * [1e0*ones(3); 1e3ones(nu-3)]

function con1(c, x, u, w)
    θ = u[nu .+ (1:nθ)]
    c .= [
        ul - u[1:nu];
        u[1:nu] - uu;
        1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT));
    ]
end
# con1(rand(nx), rand(nu + nθ), rand(0))
function con1_jacobian_state(cx, x, u, w)
    θ = u[nu .+ (1:nθ)]
    # [
    #     ul - u[1:nu];
    #     u[1:nu] - uu;
    #     1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT));
    # ]
    cx .= [
        zeros(nu + nu, nx);
        1.0e-2 * -1.0 * policy_jacobian_state(θ, x[1:nx], xT);
    ]
end
# con1_jacobian_state(rand(nx), rand(nu + nθ), rand(0))

function con1_jacobian_control(cu, x, u, w)
    θ = u[nu .+ (1:nθ)]
    # [
    #     ul - u[1:nu];
    #     u[1:nu] - uu;
    #     1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT));
    # ]
    cu .= [
        -1.0 * I(nu) zeros(nu, nθ);
         1.0 * I(nu) zeros(nu, nθ);
         1.0e-2 * [zeros(nu - 3, 3) 1.0 * I(nu - 3) -policy_jacobian_parameters(θ, x[1:nx], xT)]
    ]
end
# con1_jacobian_control(rand(nx), rand(nu + nθ), rand(0))

function cont(c, x, u, w)
    θ = x[nx .+ (1:nθ)]
    c .= [
        ul - u[1:nu];
        u[1:nu] - uu;
        1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT))
    ]
end
# cont(rand(nx + nθ), rand(nθ), rand(0))

function cont_jacobian_state(cx, x, u, w)
    θ = x[nx .+ (1:nθ)]
    # [
    #     ul - u[1:nu];
    #     u[1:nu] - uu;
    #     1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT))
    # ]
    cx .= [
        zeros(2 * nu, nx + nθ); 
        1.0e-2 * [-policy_jacobian_state(θ, x[1:nx], xT) -policy_jacobian_parameters(θ, x[1:nx], xT)]
    ]
end
# cont_jacobian_state(rand(nx + nθ), rand(nθ), rand(0))

function cont_jacobian_control(cu, x, u, w)
    θ = x[nx .+ (1:nθ)]
    # [
    #     ul - u[1:nu];
    #     u[1:nu] - uu;
    #     1.0e-2 * (u[3 .+ (1:(nu - 3))] - policy(θ, x[1:nx], xT))
    # ]
    cu .= [
        -1.0 * I(nu); 
         1.0 * I(nu);
         1.0e-2 * [zeros(nu - 3, 3) 1.0 * I(nu - 3)]
    ]
end
# cont_jacobian_control(rand(nx + nθ), rand(nθ), rand(0))

function goal(x, u, w)
    [
        # x[[1,nq+1]] - xT[[1,nq+1]];
        x[nq+1:nq+1] - xT[nq+1:nq+1];
    ]
end
con_policy1 = iLQR.Constraint(con1, con1_jacobian_state, con1_jacobian_control, 3nu - 3, nx, nu + nθ, indices_inequality=collect(1:2nu))
con_policyt = iLQR.Constraint(cont, cont_jacobian_state, cont_jacobian_control, 3nu - 3, nx + nθ, nu, indices_inequality=collect(1:2nu))

cons = [con_policy1, [con_policyt for t = 2:T-1]..., iLQR.Constraint(goal, nx + nθ, 0)]

# ## problem
opts = iLQR.Options(
    line_search=:armijo,
    max_iterations=75,
    max_dual_updates=8,
    objective_tolerance=1e-3,
    lagrangian_gradient_tolerance=1e-3,
    constraint_tolerance=1e-3,
    scaling_penalty=10.0,
    max_penalty=1e7,
    verbose=true)

p = iLQR.Solver(dyn, obj, cons, options=opts)

# ## initialize
u_hist[1]
θ0 = vcat(policy_parameters...)
u_guess = [t == 1 ? [u_hist[t]; θ0] : u_hist[t] for t = 1:T-1]
x_guess = iLQR.rollout(dyn, x1, u_guess)
s = Simulator(RoboDojo.halfcheetah4, T-1, h=h)
for i = 1:T
    q = x_guess[i][1:nq]
    v = x_guess[i][nq .+ (1:nq)]
    RD.set_state!(s, q, v, i)
end
visualize!(vis, s)

iLQR.initialize_controls!(p, u_guess)
iLQR.initialize_states!(p, x_guess)
dynamics_model.ip.opts.r_tol = 1e-5
dynamics_model.ip.opts.κ_tol = 1e-4

# ## callback
scb = Simulator(RoboDojo.halfcheetah4, T-1, h=h)

function continuation_callback!(solver, dymamics_model::RoboDojo.Simulator)
    # dynamics_model.ip.opts.r_tol = max(0.5 * dynamics_model.ip.opts.r_tol, 1e-6)
    dynamics_model.ip.opts.κ_tol = max(0.5 * dynamics_model.ip.opts.κ_tol, 1e-4)
    @printf("r_tol: %9.2e \n, κ_tol: %9.2e", dynamics_model.ip.opts.r_tol, dynamics_model.ip.opts.κ_tol)
    
    x_sol, u_sol = iLQR.get_trajectory(solver)
    for i = 1:T
        q = x_sol[i][1:nq]
        v = x_sol[i][nq .+ (1:nq)]
        RD.set_state!(scb, q, v, i)
    end
    visualize!(vis, scb)
    return nothing
end
local_continuation_callback!(solver) = continuation_callback!(solver, dynamics_model)

# ## solve
@time iLQR.constrained_ilqr_solve!(p, augmented_lagrangian_callback! = local_continuation_callback!)

# ## solution
x_sol, u_sol = iLQR.get_trajectory(p)
θ_sol = u_sol[1][nu .+ (1:nθ)]

using JLD2
@save joinpath(@__DIR__, "halfcheetah_policy.jld") θ_sol
# ## state
# plot(hcat([x[1:nx] for x in x_sol]...)', label="", color=:orange, width=2.0)

# # ## control
# plot(hcat([u[1:nu] for u in u_sol]..., u_sol[end])', linetype = :steppost)

# # ## plot xy
# plot([x[1] for x in x_sol], [x[2] for x in x_sol], label="", color=:black, width=2.0)

# ## visualization
s = Simulator(RoboDojo.halfcheetah4, T-1, h=h)
for i = 1:T
    q = x_sol[i][1:nq]
    v = x_sol[i][nq .+ (1:nq)]
    RD.set_state!(s, q, v, i)
end
visualize!(vis, s)

# ## simulate policy
x_hist = [x1]
u_hist = [u_hover]

for t = 1:5T
    push!(u_hist, [0;0;0; policy(θ_sol, x_hist[end], xT)])
    y = zeros(nx)
    RD.dynamics(dynamics_model, y, x_hist[end], u_hist[end], zeros(nw))
    push!(x_hist, y)
end

s = Simulator(RoboDojo.halfcheetah4, 5T-1, h=h)
for i = 1:5T
    q = x_hist[i][1:nq]
    v = x_hist[i][nq .+ (1:nq)]
    RD.set_state!(s, q, v, i)
end
θ_sol
visualize!(vis, s)
# set_light!(vis)
# set_floor!(vis)
u_hist

# Dojo.convert_frames_to_video_and_gif("halfhyena_single_regularized_open_loop")
# Dojo.convert_frames_to_video_and_gif("halfhyena_single_regularized_policy")