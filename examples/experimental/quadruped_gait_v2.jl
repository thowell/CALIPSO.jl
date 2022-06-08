# @testset "Examples: Quadruped gait (nonlinear friction)" begin 

# ## time 
horizon = 21
T_fix = 2
timestep = 0.625 / (horizon - 1)

# ## RoboDojo dynamics
include("robodojo.jl")
model = RoboDojo.quadruped
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)
for (i, n) in enumerate(num_states) 
    i > 1 && (num_states[i] += 22) # add states to propagate initial state
end

# ## dynamics
function quadruped_dynamics_1(sim, y, x, u)
    [
        robodojo_dynamics(sim, y[1:82], x, u);
        y[82 .+ (1:22)] - x;
    ]
end

function quadruped_dynamics_t(sim, y, x, u)
    [
        robodojo_dynamics(sim, y[1:82], x[1:82], u);
        y[82 .+ (1:22)] - x[82 .+ (1:22)];
    ]
end

dynamics = [
        (y, x, u) -> quadruped_dynamics_1(sim, y, x, u),
        [(y, x, u) -> quadruped_dynamics_t(sim, y, x, u) for t = 2:horizon-1]...,
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

# ## feet positions
pr1 = RoboDojo.quadruped_contact_kinematics[1](q1)
pr2 = RoboDojo.quadruped_contact_kinematics[2](q1)
pf1 = RoboDojo.quadruped_contact_kinematics[3](q1)
pf2 = RoboDojo.quadruped_contact_kinematics[4](q1)

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

xr2_el, zr2_el = ellipse_trajectory(pr2[1], pr2[1] + stride, zh, horizon - T_fix)
xr2 = [[xr2_el[1] for t = 1:T_fix]..., xr2_el...]
zr2 = [[zr2_el[1] for t = 1:T_fix]..., zr2_el...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:horizon]

xf2_el, zf2_el = ellipse_trajectory(pf2[1], pf2[1] + stride, zh, horizon - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:horizon]

# tr = range(0, stop = tf, length = horizon)
# plot(tr, hcat(pr1_ref...)')
# plot!(tr, hcat(pf1_ref...)')

# plot(tr, hcat(pr2_ref...)')
# plot!(tr, hcat(pf2_ref...)')

# ## objective
objective = Function[]
push!(objective, (x, u) -> begin 
        u_ctrl = u[1:8]
        q = x[11 .+ (1:11)]

        J = 0.0 
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 1.0e-5 * dot(x, x)
        J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    return J
end)

for t = 2:horizon-1
    push!(objective, (x, u) -> begin 
            u_ctrl = u[1:8]
            q = x[11 .+ (1:11)]

            J = 0.0 
            J += 1.0e-2 * dot(u_ctrl, u_ctrl)
            J += 1.0e-3 * dot(q - qT, q - qT)
            J += 1.0e-5 * dot(x, x)
            J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
            J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
            J += 1.0e-1 * sum((pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
            J += 1.0e-1 * sum((pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

            return J
        end
    )
end

push!(objective, (x, u) -> begin 
        q = x[11 .+ (1:11)]
        J = 0.0 
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 1.0e-5 * dot(x, x)
        J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 1.0e-1 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 1.0e-1 * sum((pr2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
        J += 1.0e-1 * sum((pf2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)
    end
)

# ## constraints 

# loop constraints 
function loop(x, u) 
    xT = x[1:22] 
    x1 = x[82 .+ (1:22)] 
    e = x1 - Array(cat(perm, perm, dims = (1, 2))) * xT 
    return [e[2:11]; e[11 .+ (2:11)]]
end

equality = Function[]

push!(equality, (x, u) -> begin 
    [
        pr1_ref[1] - RoboDojo.quadruped_contact_kinematics[1](x[1:11]);
        pf1_ref[1] - RoboDojo.quadruped_contact_kinematics[3](x[1:11]);
        pr2_ref[1] - RoboDojo.quadruped_contact_kinematics[2](x[1:11]);
        pf2_ref[1] - RoboDojo.quadruped_contact_kinematics[4](x[1:11]);
        x[11 .+ (1:11)] - q1;
    ]
    end
)

for t = 2:T_fix
    push!(equality, (x, u) -> begin 
        [
            pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](x[1:11]);
            pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](x[1:11]);
            pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](x[1:11]);
            pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](x[1:11]);
        ]
        end
    )
end

for t = (T_fix + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
        [
            pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](x[1:11]);
            pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](x[1:11]);
        ]
        end
    )
end

push!(equality, (x, u) -> begin 
    [
        loop(x, u);
        x[11 + 1] - qT[1];
    ]
    end
)

nonnegative = robodojo_nonnegative(sim, horizon); 
second_order = robodojo_second_order(sim, horizon);

# ## options 
options = Options(
    verbose=true,        
    constraint_tensor=false,
    update_factorization=false,
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options,
);

# ## initialize
q_interp = CALIPSO.linear_interpolation(q1, qT, horizon+1)
x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:horizon]
action_guess = [1.0e-3 * randn(model.nu) for t = 1:horizon-1] # may need to run more than once to get good trajectory
state_guess = [t == 1 ? x_interp[1] : [
    x_interp[t]; 
    ones(model.nc); 
    ones(model.nc); 
    ones(model.nc); 
    0.1 * ones(model.nc); 
    ones(model.nc); 
    0.1 * ones(model.nc); 
    x_interp[1]] for t = 1:horizon]
initialize_states!(solver, state_guess)
initialize_actions!(solver, action_guess)

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

