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
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    # equality_general=equality_general,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options);

# ## callbacks 
vis = Visualizer()
render(vis)
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

# using JLD2
# @save joinpath(@__DIR__, "quadruped_gait.jld2") x_sol u_sol

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
render(vis)
q_vis = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
u_vis = [ut[3 .+ (1:8)] for ut in u_sol]
for i = 1:3
    horizon = length(q_vis) - 1
    q_vis, u_vis = mirror_gait(q_vis, u_vis, horizon)
end
RoboDojo.visualize!(vis, model, q_vis, Δt=timestep)

using Plots
plot(hcat(u_vis...)')


# ## open-loop rollout 
x_hist = [copy(state_reference[1])]
for i = 1:11
    y = zeros(22)
    RoboDojo.dynamics(sim, y, x_hist[end], [slack_reference; 0*ones(8)], zeros(0))
    push!(x_hist, y)
end
# using Plots
# plot(hcat(x_hist...)'[:,1:3])

s = Simulator(model, 11-1, h=timestep)
for i = 1:11
    q = x_hist[i][1:11]
    v = x_hist[i][11 .+ (1:11)]
    RoboDojo.set_state!(s, q, v, i)
end
visualize!(vis, s)
