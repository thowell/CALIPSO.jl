# @testset "Examples: Quadruped gait (nonlinear friction)" begin 
# ## horizon
horizon = 41

# ## time steps
timestep = 0.01
fixed_timesteps = 5

# ## RoboDojo dynamics 
include("robodojo.jl")
include("quadruped_template.jl")

model = RoboDojo.quadruped4
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)
num_actions = [8 for t = 1:horizon-1]

# ## dynamics
dynamics = [(y, x, u) -> Diagonal([1.0 * ones(model.nq); ones(length(sim.ip.z))]) * robodojo_dynamics(sim, y, x, [zeros(3); u]) for t = 1:horizon-1]

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

# perm8 = [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
#          0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
#          1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
#          0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
#          0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
#          0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
#          0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
#          0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]
state_reference = trotting_gait(model, horizon; 
    timestep=0.01, 
    velocity=0.1, 
    body_height=0.25, 
    body_forward_position=0.025,
    foot_height=0.05)

vis = Visualizer() 
render(vis)
RoboDojo.visualize!(vis, model, state_reference, Δt=timestep)

# ## objective
objective = Function[]

for t = 1:horizon
    push!(objective, (x, u) -> begin 
            J = 0.0 
            t < horizon && (J += 1.0e-3 * dot(u[1:8], u[1:8]))
            J += 100.0 * dot(x[1:22] - state_reference[t+1], x[1:22] - state_reference[t+1])
            return J
        end
    );
end

equality = Function[]

push!(equality, (x, u) -> begin 
        q = x[model.nq .+ (1:model.nq)]
        q̄ = state_reference[model.nq .+ (1:model.nq)]
        [
            10.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
            10.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
            10.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
            10.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
            10.0 * (x[11 .+ (1:11)] - state_reference[1][11 .+ (1:11)]);
            # u[1:3];
        ]
    end
);

for t = 2:fixed_timesteps
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[model.nq .+ (1:model.nq)]
            [
                10.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                10.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                10.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
                10.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
                # u[1:3];
            ]
        end
    );
end

for t = (fixed_timesteps + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[model.nq .+ (1:model.nq)]
            [
                10.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                10.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                # u[1:3];
            ]
        end
    );
end

push!(equality, (x, u) -> begin 
        # q = x[1:11]
        # xT = x[1:22] 
        # x1 = x[46 .+ (1:22)] 
        # e = x1 - Array(cat(perm, perm, dims=(1, 2))) * xT 
        [
            # e[2:11]; 
            # e[11 .+ (2:11)];
            10.0 * (x[12] - state_reference[horizon][12]);
        ]
    end
);

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
        linear_solver=:LU,  
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    equality_general=equality_general,
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
action_guess = [1.0e-3 * randn(8) for t = 1:horizon-1] # may need to run more than once to get good trajectory
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

# # ## visualize 
function mirror_gait(q, horizon)
    qm = [deepcopy(q)...]
    stride = zero(qm[1])
    stride[1] = q[horizon+1][1] - q[2][1]
    for t = 1:horizon-1
        push!(qm, Array(perm) * q[t+2] + stride)
    end
    return qm
end

vis = Visualizer() 
open(vis)
q_vis = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
for i = 1:3
    horizon = length(q_vis) - 1
    q_vis = mirror_gait(q_vis, horizon)
end
RoboDojo.visualize!(vis, model, q_vis, Δt=timestep)

# 
# stride = zero(qm[1])
# @show stride[1] = q[T+1][1] - q[2][1]
# @show 0.5 * strd

# push!(qm, Array(perm) * q[t+2] + stride)
# push!(um, perm8 * u[t])
