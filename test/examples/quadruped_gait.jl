# @testset "Examples: Quadruped gait (linear friction)" begin
function quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u) 
    model = RoboDojo.quadruped

    # configurations
    
    q1⁻ = x[1:11] 
    q2⁻ = x[11 .+ (1:11)]
    q2⁺ = y[1:11]
    q3⁺ = y[11 .+ (1:11)]

    # control 
    u_control = u[1:8] 
    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 

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
            timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end

function quadruped_dyn1(mass_matrix, dynamics_bias, timestep, y, x, u)
    model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
        y[22 .+ (1:4)] - u[8 .+ (1:4)];
        y[22 + 4 .+ (1:22)] - x[1:22];
    ]
end

function quadruped_dynt(mass_matrix, dynamics_bias, timestep, y, x, u)
    model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
        y[22 .+ (1:4)] - u[8 .+ (1:4)];
        y[22 + 4 .+ (1:22)] - x[22 + 4 .+ (1:22)];
    ]
end

function contact_constraints_inequality_1(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
        ϕ;
        fc;
    ]
end

function contact_constraints_inequality_t(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    u_control = u[1:8] 
    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    ψ = u[8 + 4 + 8 .+ (1:4)] 
    η = u[8 + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]

    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

    [
        ϕ;
        fc;
    ]
end

function contact_constraints_inequality_T(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[22 .+ (1:4)]

    [
        ϕ;
    ]
end

function contact_constraints_equality_1(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    ψ = u[8 + 4 + 8 .+ (1:4)] 
    η = u[8 + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]

    v = (q3 - q2) ./ timestep[1]
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

function contact_constraints_equality_t(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    ψ = u[8 + 4 + 8 .+ (1:4)] 
    η = u[8 + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[nx .+ (1:4)] 

    v = (q3 - q2) ./ timestep[1]
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

function contact_constraints_equality_T(timestep, x, u) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    ψ = u[8 + 4 + 8 .+ (1:4)] 
    η = u[8 + 4 + 8 + 4 .+ (1:8)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[nx .+ (1:4)] 

    return γ⁻ .* ϕ
end

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

# ## quadruped 
nc = 4
nq = RoboDojo.quadruped.nq
nx = 2 * nq
nu = RoboDojo.quadruped.nu + nc + 8 + nc + 8
nw = RoboDojo.quadruped.nw

# ## time 
horizon = 41 
T_fix = 5
timestep = 0.01

# ## dimensions 
num_states = [nx, [2 * nx + nc for t = 2:horizon]...]
num_actions = [nu for t = 1:horizon-1]

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)
q1[2] += 0.0

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

# ## model
println("codegen dynamics")
# ## model
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
dynamics = [
        (y, x, u) -> quadruped_dyn1(mass_matrix, dynamics_bias, [timestep], y, x, u), 
        [(y, x, u) -> quadruped_dynt(mass_matrix, dynamics_bias, [timestep], y, x, u) for t = 2:horizon-1]...,
]
println("codegen dynamics complete!")

# ## objective
objective = Function[]
push!(objective, (x, u) -> begin 
        u_ctrl = u[1:8]
        q = x[11 .+ (1:11)]

        J = 0.0 
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    return J
end)

for t = 2:horizon-1
    push!(objective, (x, u) -> begin 
            u_ctrl = u[1:8]
            q = x[11 .+ (1:11)]

            J = 0.0 
            J += 1.0e-2 * dot(u_ctrl, u_ctrl)
            J += 1.0e-3 * dot(q - qT, q - qT)
            J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
            J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
            J += 1.0 * sum((pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
            J += 1.0 * sum((pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)

            return J
        end
    )
end

push!(objective, (x, u) -> begin 
        q = x[11 .+ (1:11)]
        J = 0.0 
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 1.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 1.0 * sum((pr2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
        J += 1.0 * sum((pf2_ref[horizon] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)
    end
)

# pinned feet constraints 
function pinned1(x, u, t) 
    q = x[1:11]
    [
        pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](q);
        pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](q);
    ] 
end 

function pinned2(x, u, t) 
    q = x[1:11]
    [
        pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q);
        pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q);
    ]
end

# loop constraints 
function loop(x, u) 
    xT = x[1:22] 
    x1 = x[22 + nc .+ (1:22)] 
    e = x1 - Array(cat(perm, perm, dims = (1,2))) * xT 
    nq = RoboDojo.quadruped.nq
    return [e[2:11]; e[11 .+ (2:11)]]
end

equality = Function[]

push!(equality, (x, u) -> begin 
    [
        pinned1(x, u, 1); 
        pinned2(x, u, 1);
        x[11 .+ (1:11)] - q1;
        contact_constraints_equality_1(timestep, x, u); 
    ]
    end
)

for t = 2:T_fix
    push!(equality, (x, u) -> begin 
        [
            pinned1(x, u, t);
            pinned2(x, u, t);
            contact_constraints_equality_t(timestep, x, u); 
        ]
        end
    )
end

for t = (T_fix + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
        [
            pinned1(x, u, t);
            contact_constraints_equality_t(timestep, x, u); 
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

nonnegative = Function[]

push!(nonnegative, (x, u) -> begin 
    [
        contact_constraints_inequality_1(timestep, x, u);
        u[8 .+ (1:(nu - 8))];
    ]
    end
)

for t = 2:T_fix
    push!(nonnegative, (x, u) -> begin 
        [
            contact_constraints_inequality_t(timestep, x, u);
            u[8 .+ (1:(nu - 8))];
        ]
        end
    )
end

for t = (T_fix + 1):(horizon-1) 
    push!(nonnegative, (x, u) -> begin 
        [
            contact_constraints_inequality_t(timestep, x, u);
            u[8 .+ (1:(nu - 8))];
        ]
        end
    )
end

push!(nonnegative, (x, u) -> begin 
    [
        contact_constraints_inequality_T(timestep, x, u);
    ]
    end
)

# ## options 
options = Options(
    verbose=true,        
    constraint_tensor=false,
    update_factorization=false,
    penalty_initial=1.0, 
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    nonnegative=nonnegative,
    options=options,
);

# ## initialize
q_interp = CALIPSO.linear_interpolation(q1, qT, horizon+1)
x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:horizon]
action_guess = [max.(0.0, 1.0e-3 * randn(nu)) for t = 1:horizon-1] # may need to run more than once to get good trajectory
state_guess = [t == 1 ? x_interp[t] : [x_interp[t]; max.(0.0, 1.0e-3 * randn(nc)); x_interp[t-1]] for t = 1:horizon]
initialize_states!(solver, state_guess);
initialize_controls!(solver, action_guess);
println("solver instantiated and initialized!")

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
# vis = Visualizer() 
# open(vis)
# q_vis = [x_sol[1][1:RoboDojo.quadruped.nq], [x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)] for x in x_sol]...]
# for i = 1:3
#     horizon = length(q_vis) - 1
#     q_vis = mirror_gait(q_vis, horizon)
# end
# length(q_vis)
# RoboDojo.visualize!(vis, RoboDojo.quadruped, q_vis, Δt=timestep)
