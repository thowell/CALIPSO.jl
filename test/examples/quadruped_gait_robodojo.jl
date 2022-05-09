# @testset "Examples: Quadruped gait (nonlinear friction)" begin 
# ## horizon
horizon = 41

# ## time steps
timestep = 0.01
fixed_timesteps = 5

# ## RoboDojo dynamics 
include("robodojo.jl")

model = RoboDojo.quadruped4
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)
for (i, nx) in enumerate(num_states) 
    i == 1 && continue
    num_states[i] += 2 * sim.model.nq  # add states to propagate first time step
end

# ## dynamics
dynamics = [(y, x, u) -> [
                            robodojo_dynamics(sim, y, x, u);
                            y[46 .+ (1:22)] - x[1:22];
                          ],
            [(y, x, u) -> [
                            robodojo_dynamics(sim, y, x, u);
                            y[46 .+ (1:22)] - x[46 .+ (1:22)];
                          ] for t = 2:horizon-1]...
]

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

function initial_configuration(model::RoboDojo.Quadruped4, θ1, θ2, θ3)
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

function ellipse_trajectory(x_start, x_goal, z, horizon)
    dist = x_goal - x_start
    a = 0.5 * dist
    b = z
    z̄ = 0.0
    x = range(x_start, stop = x_goal, length = horizon)
    z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
    return x, z
end

function mirror_gait(q, horizon)
    qm = [deepcopy(q)...]
    stride = zero(qm[1])
    stride[1] = q[horizon+1][1] - q[2][1]
    for t = 1:horizon-1
        push!(qm, Array(perm) * q[t+2] + stride)
    end
    return qm
end

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(model, θ1, θ2, θ3)
q1[2] += 0.0

# ## feet positions
pr1 = RoboDojo.quadruped4_contact_kinematics[1](q1)
pr2 = RoboDojo.quadruped4_contact_kinematics[2](q1)
pf1 = RoboDojo.quadruped4_contact_kinematics[3](q1)
pf2 = RoboDojo.quadruped4_contact_kinematics[4](q1)

stride = 2 * (pr1 - pr2)[1]
qT = Array(perm) * copy(q1)
qT[1] += 0.5 * stride

zh = 0.05

xr1 = [pr1[1] for t = 1:horizon]
zr1 = [pr1[2] for t = 1:horizon]
pr1_ref = [[xr1[t]; zr1[t]] for t = 1:horizon]

xf1 = [pf1[1] for t = 1:horizon]
zf1 = [pf1[2] for t = 1:horizon]
pf1_ref = [[xf1[t]; zf1[t]] for t = 1:horizon]

xr2_el, zr2_el = ellipse_trajectory(pr2[1], pr2[1] + stride, zh, horizon - fixed_timesteps)
xr2 = [[xr2_el[1] for t = 1:fixed_timesteps]..., xr2_el...]
zr2 = [[zr2_el[1] for t = 1:fixed_timesteps]..., zr2_el...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:horizon]

xf2_el, zf2_el = ellipse_trajectory(pf2[1], pf2[1] + stride, zh, horizon - fixed_timesteps)
xf2 = [[xf2_el[1] for t = 1:fixed_timesteps]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:fixed_timesteps]..., zf2_el...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:horizon]

# tr = range(0, stop = tf, length = horizon)
# plot(tr, hcat(pr1_ref...)')
# plot!(tr, hcat(pf1_ref...)')

# plot(tr, hcat(pr2_ref...)')
# plot!(tr, hcat(pf2_ref...)')

# ## objective
objective = Function[]

# t = 1
push!(objective, (x, u) -> begin 
        u_ctrl = u[1:8]
        q = x[11 .+ (1:11)]

        J = 0.0 
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 100.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        return J
    end
)

for t = 2:horizon-1
    push!(objective, (x, u) -> begin 
            u_ctrl = u[1:8]
            q = x[11 .+ (1:11)]

            J = 0.0 
            J += 1.0e-2 * dot(u_ctrl, u_ctrl)
            J += 1.0e-3 * dot(q - qT, q - qT)
            J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
            J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
            J += 100.0 * sum((pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
            J += 100.0 * sum((pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

            return J
        end
    );
end

push!(objective, (x, u) -> begin 
        q = x[11 .+ (1:11)]

        J = 0.0 
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 100.0 * sum((pr2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
        J += 100.0 * sum((pf2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

        return J
    end
);

equality = Function[]

push!(equality, (x, u) -> begin 
        q = x[1:11]
        [
            pr1_ref[1] - RoboDojo.quadruped4_contact_kinematics[1](q);
            pf1_ref[1] - RoboDojo.quadruped4_contact_kinematics[3](q);
            pr2_ref[1] - RoboDojo.quadruped4_contact_kinematics[2](q);
            pf2_ref[1] - RoboDojo.quadruped4_contact_kinematics[4](q);
            x[11 .+ (1:11)] - q1;
        ]
    end
);

for t = 2:fixed_timesteps
    push!(equality, (x, u) -> begin 
            q = x[1:11]
            [
                pr1_ref[t] - RoboDojo.quadruped4_contact_kinematics[1](q);
                pf1_ref[t] - RoboDojo.quadruped4_contact_kinematics[3](q);
                pr2_ref[t] - RoboDojo.quadruped4_contact_kinematics[2](q);
                pf2_ref[t] - RoboDojo.quadruped4_contact_kinematics[4](q);
            ]
        end
    );
end

for t = (fixed_timesteps + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
            q = x[1:11]
            [
                pr1_ref[t] - RoboDojo.quadruped4_contact_kinematics[1](q);
                pf1_ref[t] - RoboDojo.quadruped4_contact_kinematics[3](q);
            ]
        end
    );
end

push!(equality, (x, u) -> begin 
        q = x[1:11]
        xT = x[1:22] 
        x1 = x[46 .+ (1:22)] 
        e = x1 - Array(cat(perm, perm, dims=(1, 2))) * xT 
        [
            e[2:11]; 
            e[11 .+ (2:11)];
            x[11 + 1] - qT[1];
        ]
    end
);

nonnegative = robodojo_nonnegative(sim, horizon);
second_order = robodojo_second_order(sim, horizon);

# ## options 
options=Options(
        verbose=true, 
        constraint_tensor=true,
        update_factorization=false,
        optimality_tolerance=1.0e-3, 
        residual_tolerance=1.0e-3,    
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options);

# ## initialize
configurations = CALIPSO.linear_interpolation(q1, qT, horizon+1)
state_guess = robodojo_state_initialization(sim, configurations, horizon)
state_augmented_guess = Vector{Float64}[] 
for (t, s) in enumerate(state_guess) 
    push!(state_augmented_guess, t == 1 ? s : [s; state_guess[1]])
end
action_guess = [1.0e-3 * randn(sim.model.nu) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_augmented_guess) 
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
