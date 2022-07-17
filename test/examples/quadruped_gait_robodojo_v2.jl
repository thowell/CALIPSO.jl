# @testset "Examples: Quadruped gait (nonlinear friction)" begin 
# ## horizon
horizon = 41

# ## time steps
timestep = 0.01
fixed_timesteps = 3

# ## RoboDojo dynamics 
include("robodojo.jl")
include("quadruped_template.jl")

model = RoboDojo.quadruped4
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)

# ## dynamics
dynamics = [(y, x, u) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon-1]

# ## permutation matrix
perm = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

perm8 = [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
         1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
         0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
         0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
         0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
         0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

state_reference = trotting_gait(model, horizon; 
    timestep=0.01, 
    velocity=0.25, 
    body_height=0.25, 
    body_forward_position=0.05,
    foot_height=0.05)

vis = Visualizer() 
open(vis)
RoboDojo.visualize!(vis, model, state_reference, Δt=timestep)

# ## objective
total_mass = 0.0 
total_mass += model.m_torso
total_mass += model.m_thigh1
total_mass += model.m_calf1
total_mass += model.m_thigh2
total_mass += model.m_calf2
total_mass += model.m_thigh3
total_mass += model.m_calf3
total_mass += model.m_thigh4
total_mass += model.m_calf4

slack_reference = [0.0; total_mass * model.gravity; 0.0]

# x_hist = [copy(state_reference[1])]
# for i = 1:11
#     y = zeros(22)
#     RoboDojo.dynamics(sim, y, x_hist[end], [slack_reference; 0*ones(8)], zeros(0))
#     push!(x_hist, y)
# end
# # using Plots
# # plot(hcat(x_hist...)'[:,1:3])

# s = Simulator(model, 11-1, h=timestep)
# for i = 1:11
#     q = x_hist[i][1:11]
#     v = x_hist[i][11 .+ (1:11)]
#     RoboDojo.set_state!(s, q, v, i)
# end
# visualize!(vis, s)

objective = Function[]
for t = 1:horizon
    push!(objective, (x, u) -> begin 
            J = 0.0 
            # controls
            t < horizon && (J += 1.0e-1 * dot(u[1:3] - slack_reference, u[1:3] - slack_reference))
            t < horizon && (J += 1.0e-3 * dot(u[3 .+ (1:8)], u[3 .+ (1:8)]))
            # kinematic reference
            J += 1.0 * dot(x[11 .+ (1:11)] - state_reference[t][1:11], x[11 .+ (1:11)] - state_reference[t][1:11])
            # velocity 
            # v = (x[11 .+ (1:11)] - x[1:11]) ./ timestep
            # J += 1.0e-3 * dot(v, v)
            return J
        end
    );
end

equality = Function[]

push!(equality, (x, u) -> begin 
        q = x[model.nq .+ (1:model.nq)]
        q̄ = state_reference[1][1:model.nq]
        [
            1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
            1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
            1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
            1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
            10.0 * (x[11 .+ (1:11)] - state_reference[1][1:11]);
            u[1:3];
        ]
    end
);

for t = 2:fixed_timesteps
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[t][1:model.nq]
            [
                1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
                1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
                u[1:3];
            ]
        end
    );
end

for t = (fixed_timesteps + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[t][1:model.nq]
            [
                1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                u[1:3];
            ]
        end
    );
end

push!(equality, (x, u) -> begin  
        [
            10.0 * (x[12] - state_reference[horizon][1]);
        ]
    end
)

equality[horizon](rand(22), rand(11))

function equality_general(z) 
    x1 = z[1:(2 * model.nq)]
    xT = z[sum(num_states[1:horizon-1]) + sum(num_actions) .+ (1:(2 * model.nq))]

    # e = x1 - Array(cat(perm, perm, dims=(1, 2))) * xT 
    e = x1 - xT
    [
        10.0 * e[2:11]; 
        10.0 * e[11 .+ (2:11)];
    ]
end

nonnegative = robodojo_nonnegative(sim, horizon);
second_order = robodojo_second_order(sim, horizon);

# ## options 
options=Options(
        verbose=true, 
        constraint_tensor=true,
        update_factorization=false,  
        # linear_solver=:LU,  
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    # equality_general=equality_general,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options);

# ## callbacks 
vis = Visualizer()
open(vis)
RoboDojo.visualize!(vis, RoboDojo.quadruped4, state_guess, Δt=timestep);

# function callback_inner(trajopt, solver) 
#     println("callback inner")
#     x_sol, u_sol = CALIPSO.get_trajectory(solver)
#     RoboDojo.visualize!(vis, RoboDojo.quadruped4, x_sol, Δt=timestep);
# end

function callback_outer(trajopt, solver) 
    println("callback outer")
    x_sol, u_sol = CALIPSO.get_trajectory(solver)
    RoboDojo.visualize!(vis, RoboDojo.quadruped4, x_sol, Δt=timestep);
end

# solver.options.callback_inner = false
solver.options.callback_outer = true

# ## initialize
state_guess = robodojo_state_initialization(sim, state_reference, horizon)
action_guess = [[slack_reference; 1.0e-3 * randn(8)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_controls!(solver, action_guess)

# ## solve 
solve!(solver)

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver)

# test solution
@test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

slack_norm = max(
                norm(solver.data.residual.equality_dual, Inf),
                norm(solver.data.residual.cone_dual, Inf),
)
@test slack_norm < solver.options.slack_tolerance

@test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
@test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# end 

using JLD2
# @save joinpath(@__DIR__, "quadruped_gait.jld2") x_sol u_sol
@load joinpath(@__DIR__, "quadruped_gait.jld2") x_sol u_sol

# # ## visualize 
function mirror_gait(q, u, horizon)
    qm = [deepcopy(q)...]
    um = [deepcopy(u)...]
    stride = zero(qm[1])
    stride[1] = q[horizon+1][1] - q[2][1]
    for t = 1:horizon-1
        push!(qm, Array(perm) * q[t+2] + stride)
        push!(um, perm8 * u[t])
    end
    return qm, um
end

vis = Visualizer() 
open(vis)
q_vis = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
u_vis = [ut[3 .+ (1:8)] for ut in u_sol]
for i = 1:3
    _horizon = length(q_vis) - 1
    q_vis, u_vis = mirror_gait(q_vis, u_vis, _horizon)
end
RoboDojo.visualize!(vis, model, q_vis, Δt=timestep)

# using Plots
# plot(hcat(u_vis...)')

# ## open-loop rollout 
x_hist = [[q_vis[2]; (q_vis[2] - q_vis[1]) ./ timestep]]
u_vis[1]
for t = 1:horizon-1
    y = zeros(22)
    RoboDojo.dynamics(sim, y, x_hist[end], [zeros(3); u_vis[t] ./ timestep], zeros(0))
    push!(x_hist, y)
end
# using Plots
# plot(hcat(x_hist...)'[:,1:3])

s = Simulator(model, horizon-1, h=timestep)
for i = 1:horizon-1
    q = x_hist[i][1:11]
    v = x_hist[i][11 .+ (1:11)]
    RoboDojo.set_state!(s, q, v, i)
end
visualize!(vis, s)

# ## reference solution 
z̄_mpc = [x_sol[t][11 .+ (1:35)] for t = 2:horizon]
θ̄_mpc = [zeros(length(sim.ip.θ)) for t = 1:horizon-1]
for t = 1:horizon-1 
    RoboDojo.initialize_θ!(θ̄_mpc[t], sim.model, sim.idx_θ, x_sol[t][1:11], x_sol[t][11 .+ (1:11)], [zeros(3); u_sol[t][3 .+ (1:8)]], sim.dist.w, sim.f, sim.h)
end

parameters_mpc = [[z̄_mpc[t]; θ̄_mpc[t]] for t = 1:horizon-1]
num_parameters_mpc = [length(parameters_mpc[t]) for t = 1:horizon-1]

function dynamics_linearized(y, x, u, w)
    z̄ = w[1:35]
    θ̄ = w[35 .+ (1:44)]
    robodojo_linearized_dynamics(sim, z̄, θ̄, y, x, [zeros(3); u])
end

using Symbolics
@variables z̄[1:35] θ̄[1:44] y[1:num_states_mpc[2]] x[1:num_states_mpc[1]] u[1:11]
# robodojo_linearized_dynamics(sim, rand(35), rand(44), rand(num_states_mpc[2]), rand(num_states_mpc[1]), [zeros(3); rand(num_actions[1])])
robodojo_linearized_dynamics(sim, rand(35), rand(44), y, x, u)

sim.ip.z
sim.ip.θ
sim.idx_θ

# ## mpc policy 
horizon_mpc = 3 
num_states_mpc, num_actions_mpc = state_action_dimensions(sim, horizon_mpc)
num_actions_mpc = [8 for t = 1:horizon_mpc-1]

# dynamics
dynamics_mpc = [dynamics_linearized for t = 1:horizon_mpc-1]

# objective
function objt_mpc(x, u, w) 
    x̄ = [w[35 + 11 .+ (1:11)]; w[1:11]] 
    ū = w[35 + 22 .+ (1:8)]
    J = 0.0 
    J += 0.5 * transpose(x[1:22] - x̄) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:22] - x̄)
    J += 0.5 * transpose(u[1:8] - ū) * Diagonal([1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2]) * (u[1:8] - ū)
    return J 
end

function objT_mpc(x, u, w) 
    x̄ = [w[35 + 11 .+ (1:11)]; w[1:11]] 
    ū = w[35 + 22 .+ (1:8)]
    J = 0.0 
    J += 0.5 * transpose(x[1:22] - x̄) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:22] - x̄)
    return J 
end

objective_mpc = [[objt_mpc for t = 1:horizon_mpc-1]..., objT_mpc]
equality_mpc = [(x, u, w) -> x[1:22] - w[35 .+ (1:22)], [empty_constraint for t = 2:horizon_mpc]...]
inequality_mpc = [[(x, u, w) -> [2.0 .- u[1:8]; u[1:8] .+ 2.0] for t = 1:horizon_mpc-1]..., empty_constraint]

options_mpc = Options(
    constraint_tensor=true,
    residual_tolerance=1.0e-3,
    optimality_tolerance=1.0e-2, 
    equality_tolerance=1.0e-2, 
    complementarity_tolerance=1.0e-2,
    slack_tolerance=1.0e-2,
    update_factorization=true,
)

# ## solver 
solver_mpc = Solver(objective_mpc, dynamics_mpc, num_states_mpc, num_actions_mpc;
    parameters=parameters_mpc[1:horizon_mpc],
    equality=equality_mpc,
    options=options_mpc);

# ## initialize
# state_guess = x_mpc[1:horizon_mpc]
# action_guess = u_mpc[1:horizon_mpc-1]
# initialize_states!(solver_mpc, state_guess) 
# initialize_controls!(solver_mpc, action_guess)

# # ## solve 
# solve!(solver_mpc)

function policy(x, τ)

    # set parameters (reference and initial state)
    p = vcat([[x_mpc[t]; u_mpc[t]] for t = 1:horizon_mpc]...)
    p[1:4] .= copy(x) 
    solver_mpc.parameters .= p

    # initialize
    initialize_states!(solver_mpc, x_mpc[1:horizon_mpc]) 
    initialize_controls!(solver_mpc, u_mpc[1:horizon_mpc])

    # optimize
    solve!(solver_mpc)

    # solution 
    x_s, u_s = get_trajectory(solver_mpc)

    # shift reference 
    x_mpc .= [x_mpc[2:end]..., x_mpc[1]]
    u_mpc .= [u_mpc[2:end]..., u_mpc[1]] 

    # return first control
    return u_s[1] 
end