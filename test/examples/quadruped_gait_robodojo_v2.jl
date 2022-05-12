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

perm4 = [0.0 1.0 0.0 0.0;
         1.0 0.0 0.0 0.0;
         0.0 0.0 0.0 1.0;
         0.0 0.0 1.0 0.0]

state_reference = trotting_gait(model, horizon; 
    timestep=0.01, 
    velocity=0.25, 
    body_height=0.25, 
    body_forward_position=0.05,
    foot_height=0.05)

vis = Visualizer() 
render(vis)
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
            t < horizon && (J += 1.0e-3 * dot(u[1:3] - slack_reference, u[1:3] - slack_reference))
            t < horizon && (J += 1.0e-3 * dot(u[3 .+ (1:8)], u[3 .+ (1:8)]))
            # kinematic reference
            J += 100.0 * dot(x[11 .+ (1:11)] - state_reference[t][1:11], x[11 .+ (1:11)] - state_reference[t][1:11])
            # velocity 
            v = (x[11 .+ (1:11)] - x[1:11]) ./ timestep
            J += 1.0e-3 * dot(v, v)
            return J
        end
    );
end

equality = Function[]

push!(equality, (x, u) -> begin 
        q = x[model.nq .+ (1:model.nq)]
        q̄ = state_reference[1][1:model.nq]
        [
            # 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
            # 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
            # 1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
            # 1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
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
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
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
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                # 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
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

    e = x1 - Array(cat(perm, perm, dims=(1, 2))) * xT 

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
# solver = Solver(objective, dynamics, num_states, num_actions; 
#     equality=equality,
#     # equality_general=equality_general,
#     nonnegative=nonnegative,
#     second_order=second_order,
#     options=options);

# ## callbacks 
# vis = Visualizer()
# render(vis)
# RoboDojo.visualize!(vis, RoboDojo.quadruped4, state_guess, Δt=timestep);

# function callback_inner(trajopt, solver) 
#     println("callback inner")
#     x_sol, u_sol = CALIPSO.get_trajectory(solver)
#     RoboDojo.visualize!(vis, RoboDojo.quadruped4, x_sol, Δt=timestep);
# end

# function callback_outer(trajopt, solver) 
#     println("callback outer")
#     x_sol, u_sol = CALIPSO.get_trajectory(solver)
#     RoboDojo.visualize!(vis, RoboDojo.quadruped4, x_sol, Δt=timestep);
# end

# # solver.options.callback_inner = false
# solver.options.callback_outer = true

# # ## initialize
# state_guess = robodojo_state_initialization(sim, state_reference, horizon)
# action_guess = [[slack_reference; 1.0e-3 * randn(8)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
# initialize_states!(solver, state_guess) 
# initialize_controls!(solver, action_guess)

# # ## solve 
# solve!(solver)

# ## solution
# x_sol, u_sol = CALIPSO.get_trajectory(solver)

# # test solution
# @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

# slack_norm = max(
#                 norm(solver.data.residual.equality_dual, Inf),
#                 norm(solver.data.residual.cone_dual, Inf),
# )
# @test slack_norm < solver.options.slack_tolerance

# @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
# @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# end 

using JLD2
# @save joinpath(@__DIR__, "quadruped_gait.jld2") x_sol u_sol
@load joinpath(@__DIR__, "quadruped_gait.jld2") x_sol u_sol

# ## mirror trajectories 
nq = model.nq
nu = model.nu - 3
nc = model.nc
v1_reference = (x_sol[1][nq .+ (1:nq)] - x_sol[1][1:nq]) ./ timestep
q_reference = [x_sol[1][1:nq], [x_sol[t][nq .+ (1:nq)] for t = 1:horizon]...]
u_reference = [u_sol[t][3 .+ (1:nu)] for t = 1:horizon-1]
γ_reference = [x_sol[t][2nq .+ (1:nc)] for t = 2:horizon]
sγ_reference = [x_sol[t][2nq + nc .+ (1:nc)] for t = 2:horizon]
ψ_reference = [x_sol[t][2nq + nc + nc .+ (1:nc)] for t = 2:horizon]
b_reference = [x_sol[t][2nq + nc + nc + nc .+ (1:nc)] for t = 2:horizon]
sψ_reference = [x_sol[t][2nq + nc + nc + nc + nc .+ (1:nc)] for t = 2:horizon]
sb_reference = [x_sol[t][2nq + nc + nc + nc + nc + nc .+ (1:nc)] for t = 2:horizon]

stride_shift = zeros(nq)
stride_shift[1] = q_reference[end][1] - q_reference[2][1]
q_mirror = [q_reference[1:end-2]..., [perm * q + stride_shift for q in q_reference]...]
u_mirror = [u_reference..., [perm8 * u for u in u_reference]...]
γ_mirror = [γ_reference..., [perm4 * γ for γ in γ_reference]...]
sγ_mirror = [sγ_reference..., [perm4 * sγ for sγ in sγ_reference]...]
ψ_mirror = [ψ_reference..., [perm4 * ψ for ψ in ψ_reference]...]
b_mirror = [b_reference..., [perm4 * b for b in b_reference]...]
sψ_mirror = [sψ_reference..., [perm4 * sψ for sψ in sψ_reference]...]
sb_mirror = [sb_reference..., [perm4 * sb for sb in sb_reference]...]

T_mirror = length(q_mirror)

vis = Visualizer() 
render(vis)
RoboDojo.visualize!(vis, model, q_mirror, Δt=timestep)

# ## open-loop rollout 
T_sim = 70
x_hist = [[q_reference[2]; v1_reference]]

for t = 1:T_sim-1
    y = zeros(22)
    RoboDojo.dynamics(sim, y, x_hist[end], [zeros(3); u_mirror[t] ./ timestep], zeros(0))
    push!(x_hist, y)
end
# using Plots
# plot(hcat(x_hist...)'[:,1:3])

s = Simulator(model, T_sim-1, h=timestep)
for i = 1:T_sim
    q = x_hist[i][1:11]
    v = x_hist[i][11 .+ (1:11)]
    RoboDojo.set_state!(s, q, v, i)
end
visualize!(vis, s)

# ## reference solution 
z̄_reference = [zeros(length(sim.ip.z)) for t = 1:T_mirror-2]
θ̄_reference = [zeros(length(sim.ip.θ)) for t = 1:T_mirror-2]
for t = 1:T_mirror-2
    z̄_reference[t] = [
            q_mirror[t+2]; 
            γ_mirror[t];
            sγ_mirror[t]; 
            ψ_mirror[t]; 
            b_mirror[t];
            sψ_mirror[t]; 
            sb_mirror[t];
    ]
    RoboDojo.initialize_θ!(
        θ̄_reference[t], sim.model, sim.idx_θ, 
        q_mirror[t], q_mirror[t+1], 
        [zeros(3); u_mirror[t]], 
        sim.dist.w, sim.f, sim.h,
    )
end

parameters_mpc = [[z̄_reference[t]; θ̄_reference[t]] for t = 1:horizon-1]
num_parameters_mpc = [length(parameters_mpc[t]) for t = 1:horizon-1]

function dynamics_linearized(y, x, u, w)
    z̄ = w[1:35]
    θ̄ = w[35 .+ (1:44)]
    robodojo_linearized_dynamics(sim, z̄, θ̄, y, x, [zeros(3); u])
end

# ## mpc policy 
horizon_mpc = 3 
num_states_mpc, num_actions_mpc = state_action_dimensions(sim, horizon_mpc)
num_actions_mpc = [8 for t = 1:horizon_mpc-1]

# dynamics
dynamics_mpc = [dynamics_linearized for t = 1:horizon_mpc-1]

# objective
function objt_mpc(x, u, w) 
    x̄ = w[35 .+ (1:22)] 
    ū = w[35 + 22 + 3 .+ (1:8)]
    J = 0.0 
    J += 0.5 * transpose(x[1:22] - x̄) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:22] - x̄)
    J += 0.5 * transpose(u[1:8] - ū) * Diagonal([1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2; 1.0e-2]) * (u[1:8] - ū)
    return J 
end

function objT_mpc(x, u, w) 
    x̄ = w[35 .+ (1:22)]
    J = 0.0 
    J += 0.5 * transpose(x[1:22] - x̄) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:22] - x̄)
    return J 
end

objective_mpc = [[objt_mpc for t = 1:horizon_mpc-1]..., objT_mpc]
equality_mpc = [(x, u, w) -> x[1:22] - w[35 .+ (1:22)], [empty_constraint for t = 2:horizon_mpc]...]
nonnegative_contact = robodojo_nonnegative(sim, horizon_mpc, parameters=true);
inequality_mpc = [[(x, u, w) -> [nonnegative_contact[t](x, u, w); 100.0 .- u[1:8]; u[1:8] .+ 100.0] for t = 1:horizon_mpc-1]..., empty_constraint]
second_order_mpc = robodojo_second_order(sim, horizon_mpc, parameters=true);

options_mpc = Options(
    constraint_tensor=true,
    residual_tolerance=1.0e-3,
    optimality_tolerance=1.0e-2, 
    equality_tolerance=1.0e-2, 
    complementarity_tolerance=1.0e-2,
    slack_tolerance=1.0e-2,
    update_factorization=false,
    differentiate=false,
)

# ## solver 
solver_mpc = Solver(objective_mpc, dynamics_mpc, num_states_mpc, num_actions_mpc;
    parameters=parameters_mpc[1:horizon_mpc],
    equality=equality_mpc,
    nonnegative=inequality_mpc,
    second_order=second_order_mpc,
    options=options_mpc);

# ## initialize
state_guess = [t == 1 ? [q_mirror[t+1]; q_mirror[t+2]] : [q_mirror[t+1]; z̄_reference[t]] for t = 1:horizon_mpc]
action_guess = [u_mirror[t] for t = 1:horizon_mpc-1] 
initialize_states!(solver_mpc, state_guess) 
initialize_controls!(solver_mpc, action_guess)

# # ## solve 
solve!(solver_mpc)

function policy(x, τ)

    # set parameters (reference and initial state)
    p = vcat(parameters_mpc[1:horizon_mpc]...)
    # p[1:11] .= #copy(x) 
    # p[]
    solver_mpc.parameters .= p

    state_guess = [t == 1 ? [x[1:nq]; (x[1:nq] - x[nq .+ (1:nq)] * timestep)] : [q_mirror[t+2]; z̄_reference[t]] for t = 1:horizon_mpc]
    action_guess = [u_mirror[t] for t = 1:horizon_mpc-1] 
    initialize_states!(solver_mpc, state_guess) 
    initialize_controls!(solver_mpc, action_guess)

    # optimize
    solve!(solver_mpc)

    # solution 
    x_s, u_s = get_trajectory(solver_mpc)

    # # shift reference 
    # x_mpc .= [z̄_mpc[2:end]..., z̄_mpc[1]]
    # u_mpc .= [θ̄_mpc[2:end]..., θ̄_mpc[1]] 

    # return first control
    return u_s[1] 
end

u_mirror[1]
policy([q_mirror[2]; v1_reference], 1)