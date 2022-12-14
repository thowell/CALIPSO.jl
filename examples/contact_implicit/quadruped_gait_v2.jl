# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RoboDojo 
using LinearAlgebra
using MeshCatMechanisms

RoboDojo.quadruped.friction_foot_world = [0.25, 0.25, 0.25, 0.25]
RoboDojo.quadruped.J_torso = 0.01683 + 4 * 0.696 * 0.183^2.0

function quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
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

    b1 = β[2]
    b2 = β[4]
    b3 = β[6]
    b4 = β[8]

    J = RoboDojo.contact_jacobian(model, q2⁺)
    λ = transpose(J[1:8, :]) * [
                                [b1; γ[1]];
                                [b2; γ[2]];
                                [b3; γ[3]];
                                [b4; γ[4]];
                            ]

    [
        q2⁺ - q2⁻;
        RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
            h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
    ]
end

function quadruped_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
    model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[22 .+ (1:4)] - u[8 .+ (1:4)];
        y[22 + 4 .+ (1:22)] - x[1:22];
        y[22 + 4 + 22 .+ (1:8)] - u[1:8];
        y[22 + 4 + 22 + 8 .+ (1:4)] - u[[14; 16; 18; 20]];
    ]
end

function quadruped_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
    model = RoboDojo.quadruped
    [
        quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
        y[22 .+ (1:4)] - u[8 .+ (1:4)];
        y[22 + 4 .+ (1:22)] - x[22 + 4 .+ (1:22)];
        y[22 + 4 + 22 .+ (1:8)] - u[1:8];
        y[22 + 4 + 22 + 8 .+ (1:4)] - u[[14; 16; 18; 20]];
    ]
end

function contact_constraints_inequality_1(h, x, u, w) 
    model = RoboDojo.quadruped

    q3 = x[11 .+ (1:11)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ = u[8 .+ (1:4)]
    [
        ϕ;
        γ;
    ]
end

function contact_constraints_inequality_t(h, x, u, w) 
    model = RoboDojo.quadruped
    q3 = x[11 .+ (1:11)] 
    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ = u[8 .+ (1:4)]
    [
        ϕ;
        γ;
    ]
end

function contact_constraints_inequality_T(h, x, u, w) 
    model = RoboDojo.quadruped
    q3 = x[11 .+ (1:11)] 
    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    [
        ϕ;
    ]
end

function contact_constraints_equality_1(h, x, u, w) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    η = u[8 + 4 + 8 .+ (1:8)] 

    β1 = β[1:2]
    η1 = η[1:2]

    β2 = β[2 .+ (1:2)]
    η2 = η[2 .+ (1:2)]

    β3 = β[4 .+ (1:2)]
    η3 = η[4 .+ (1:2)]

    β4 = β[6 .+ (1:2)]
    η4 = η[6 .+ (1:2)]

    v = (q3 - q2) ./ h[1]
    vc = vcat([(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] - η[(i - 1) * 2 .+ (1:2)][2] for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - [β[1]; β[3]; β[5]; β[7]]
    
    return [
            fc;
            vc;
            CALIPSO.second_order_product(β1, η1);
            CALIPSO.second_order_product(β2, η2);
            CALIPSO.second_order_product(β3, η3);
            CALIPSO.second_order_product(β4, η4);
    ]
end

function contact_constraints_equality_t(h, x, u, w) 
    model = RoboDojo.quadruped

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[22 .+ (1:4)] 

    q2 = x[1:11] 
    q3 = x[11 .+ (1:11)] 

    γ = u[8 .+ (1:4)] 
    β = u[8 + 4 .+ (1:8)] 
    η = u[8 + 4 + 8 .+ (1:8)] 

    β1 = β[1:2]
    η1 = η[1:2]

    β2 = β[2 .+ (1:2)]
    η2 = η[2 .+ (1:2)]

    β3 = β[4 .+ (1:2)]
    η3 = η[4 .+ (1:2)]

    β4 = β[6 .+ (1:2)]
    η4 = η[6 .+ (1:2)]

    v = (q3 - q2) ./ h[1]
    vc = vcat([(RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] - η[(i - 1) * 2 .+ (1:2)][2] for i = 1:4]...)
    
    μ = RoboDojo.friction_coefficients(model)[1:4]
    fc = μ .* γ[1:4] - [β[1]; β[3]; β[5]; β[7]]
    
    return [
            fc;
            vc;
            ϕ .* γ⁻;
            CALIPSO.second_order_product(β1, η1);
            CALIPSO.second_order_product(β2, η2);
            CALIPSO.second_order_product(β3, η3);
            CALIPSO.second_order_product(β4, η4);
    ]
end

function contact_constraints_equality_T(h, x, u, w) 
    model = RoboDojo.quadruped

    q3 = x[11 .+ (1:11)] 

    ϕ = RoboDojo.signed_distance(model, q3)[1:4]
    γ⁻ = x[22 .+ (1:4)] 

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

function ellipse_trajectory(x_start, x_goal, z, T)
    dist = x_goal - x_start
    a = 0.5 * dist
    b = z
    z̄ = 0.0
    x = range(x_start, stop = x_goal, length = T)
    z = sqrt.(max.(0.0, (b^2) * (1.0 .- ((x .- (x_start + a)).^2.0) / (a^2.0))))
    return x, z
end

# function mirror_gait(q, T)
#     qm = [deepcopy(q)...]
#     stride = zero(qm[1])
#     stride[1] = q[T+1][1] - q[2][1]
#     for t = 1:T-1
#         push!(qm, Array(perm) * q[t+2] + stride)
#     end
#     return qm
# end

# ## quadruped 
nc = 4
nq = RoboDojo.quadruped.nq
nx = 2 * nq
nu = RoboDojo.quadruped.nu + nc + 8 + 8
nw = RoboDojo.quadruped.nw

# ## time 
T = 31 
T_fix = 7
h = 0.015625

# ## initial configuration
θ1 = pi / 4.0
θ2 = pi / 4.0
θ3 = pi / 3.0

q1 = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)
q1[2] += 0.0#
# RoboDojo.signed_distance(RoboDojo.quadruped, q1)[1:4]
# vis = Visualizer()
# open(vis)
# RoboDojo.visualize!(vis, RoboDojo.quadruped, [q1])

# ## feet positions
pr1 = RoboDojo.quadruped_contact_kinematics[1](q1)
pr2 = RoboDojo.quadruped_contact_kinematics[2](q1)
pf1 = RoboDojo.quadruped_contact_kinematics[3](q1)
pf2 = RoboDojo.quadruped_contact_kinematics[4](q1)

stride = 2 * (pr1 - pr2)[1]
qT = Array(perm) * copy(q1)
qT[1] += 0.5 * stride

zh = 0.075

xr1 = [pr1[1] for t = 1:T]
zr1 = [pr1[2] for t = 1:T]
pr1_ref = [[xr1[t]; zr1[t]] for t = 1:T]

xf1 = [pf1[1] for t = 1:T]
zf1 = [pf1[2] for t = 1:T]
pf1_ref = [[xf1[t]; zf1[t]] for t = 1:T]

xr2_el, zr2_el = ellipse_trajectory(pr2[1], pr2[1] + stride, zh, T - T_fix)
xr2 = [[xr2_el[1] for t = 1:T_fix]..., xr2_el...]
zr2 = [[zr2_el[1] for t = 1:T_fix]..., zr2_el...]
pr2_ref = [[xr2[t]; zr2[t]] for t = 1:T]

xf2_el, zf2_el = ellipse_trajectory(pf2[1], pf2[1] + stride, zh, T - T_fix)
xf2 = [[xf2_el[1] for t = 1:T_fix]..., xf2_el...]
zf2 = [[zf2_el[1] for t = 1:T_fix]..., zf2_el...]
pf2_ref = [[xf2[t]; zf2[t]] for t = 1:T]

# tr = range(0, stop = tf, length = T)
# plot(tr, hcat(pr1_ref...)')
# plot!(tr, hcat(pf1_ref...)')

# plot(tr, hcat(pr2_ref...)')
# plot!(tr, hcat(pf2_ref...)')

# ## model
println("codegen dynamics")
mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
d1 = CALIPSO.Dynamics((y, x, u) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, zeros(0)), nx + nc + nx + 8 + 4, nx, nu);
dt = CALIPSO.Dynamics((y, x, u) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, zeros(0)), nx + nc + nx + 8 + 4, nx + nc + nx + 8 + 4, nu);
dyn = [d1, [dt for t = 2:T-1]...];
println("codegen dynamics complete!")

# ## objective
obj = CALIPSO.Cost{Float64}[]

function obj1(x, u)
    u_ctrl = u[1:8]
    q = x[11 .+ (1:11)]

    J = 0.0 
    J += 1.0e-2 * dot(u_ctrl, u_ctrl)
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    v = (q - x[1:11]) ./ h
    J += 1.0e-4 * dot(v, v)
    return J
end
push!(obj, CALIPSO.Cost(obj1, nx, nu))

for t = 2:T-1
    function objt(x, u)
        u_ctrl = u[1:8]
        q = x[11 .+ (1:11)]

        J = 0.0 
        J += 1.0e-2 * dot(u_ctrl, u_ctrl)
        J += 1.0e-3 * dot(q - qT, q - qT)
        J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
        J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
        J += 10.0 * sum((pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
        J += 10.0 * sum((pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)
        v = (q - x[1:11]) ./ h
        J += 1.0e-4 * dot(v, v)

        # prev
        γ = u[8 .+ (1:4)]
        γ_prev = x[22 .+ (1:4)]
        u_prev = x[22 + 4 + 22 .+ (1:8)]
        b = u[[14; 16; 18; 20]]
        b_prev = x[22 + 4 + 22 + 8 .+ (1:4)]

        J += 1.0e-1 * dot(γ - γ_prev, γ - γ_prev);
        J += 1.0e-1 * dot(u_ctrl - u_prev, u_ctrl - u_prev);
        J += 1.0e-1 * dot(b - b_prev, b - b_prev);
        return J
    end
    push!(obj, CALIPSO.Cost(objt, nx + nc + nx + 8 + 4, nu));
end

function objT(x, u)
    q = x[11 .+ (1:11)]

    J = 0.0 
    J += 1.0e-3 * dot(q - qT, q - qT)
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[9](q)[2] - qT[2])
    J += 10.0 * dot(RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2], RoboDojo.quadruped_contact_kinematics[10](q)[2] - qT[2])
    J += 10.0 * sum((pr2_ref[T] - RoboDojo.quadruped_contact_kinematics[2](q)).^2.0)
    J += 10.0 * sum((pf2_ref[T] - RoboDojo.quadruped_contact_kinematics[4](q)).^2.0)
    v = (q - x[1:11]) ./ h
    J += 1.0e-4 * dot(v, v)
    return J
end
push!(obj, CALIPSO.Cost(objT, nx + nc + nx + 8 + 4, 0));

# control limits

# pinned feet constraints 
function pinned1(x, u, w, t) 
    q1 = x[1:11]
    q2 = x[11 .+ (1:11)]
    [
        pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](q1);
        pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](q1);
        pr1_ref[t] - RoboDojo.quadruped_contact_kinematics[1](q2);
        pf1_ref[t] - RoboDojo.quadruped_contact_kinematics[3](q2);
    ] 
end 

function pinned2(x, u, w, t) 
    q1 = x[1:11]
    q2 = x[11 .+ (1:11)]
    [
        pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q1);
        pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q1);
        pr2_ref[t] - RoboDojo.quadruped_contact_kinematics[2](q2);
        pf2_ref[t] - RoboDojo.quadruped_contact_kinematics[4](q2);
    ]
end

# loop constraints 
function loop(x, u, w) 
    xT = x[1:22] 
    x1 = x[22 + nc .+ (1:22)] 
    e = x1 - Array(cat(perm, perm, dims = (1,2))) * xT 
    return [e[2:11]; e[11 .+ (2:11)]]
end

eq = CALIPSO.Constraint{Float64}[]
function equality_1(x, u) 
    w = zeros(0);
    [
        pinned1(x, u, w, 1); 
        pinned2(x, u, w, 1);
        x[11 .+ (1:11)] - q1;
        contact_constraints_equality_1(h, x, u, w); 
    ]
end
push!(eq, CALIPSO.Constraint(equality_1, nx, nu));

for t = 2:T_fix
    function equality_t(x, u) 
        w = zeros(0);
        [
            pinned1(x, u, w, t);
            pinned2(x, u, w, t);
            contact_constraints_equality_t(h, x, u, w); 
        ]
    end
    push!(eq, CALIPSO.Constraint(equality_t, nx + nc + nx + 8 + 4, nu));
end

for t = (T_fix + 1):(T-1) 
    function equality_t(x, u)
        w = zeros(0); 
        [
            pinned1(x, u, w, t);
            contact_constraints_equality_t(h, x, u, w); 
        ]
    end
    push!(eq, CALIPSO.Constraint(equality_t, nx + nc + nx + 8 + 4, nu));
end

function equality_T(x, u) 
    w = zeros(0);
    [
    loop(x, u, w);
    x[11 + 1] - qT[1];
    ]
end
push!(eq, CALIPSO.Constraint(equality_T, nx + nc + nx + 8 + 4, 0));

ineq = CALIPSO.Constraint{Float64}[]
function inequality_1(x, u)
    w = zeros(0); 
    [
        contact_constraints_inequality_1(h, x, u, w);
    ]
end
push!(ineq, CALIPSO.Constraint(inequality_1, nx, nu));

for t = 2:T_fix
    function inequality_t(x, u) 
        w = zeros(0);
        [
            contact_constraints_inequality_t(h, x, u, w);
        ]
    end
    push!(ineq, CALIPSO.Constraint(inequality_t, nx + nc + nx + 8 + 4, nu));
end

for t = (T_fix + 1):(T-1) 
    function inequality_t(x, u) 
        w = zeros(0);
        [
            contact_constraints_inequality_t(h, x, u, w);
        ]
    end
    push!(ineq, CALIPSO.Constraint(inequality_t, nx + nc + nx + 8 + 4, nu));
end

function inequality_T(x, u)
    w = zeros(0); 
    [
        contact_constraints_inequality_T(h, x, u, w);
    ]
end
push!(ineq, CALIPSO.Constraint(inequality_T, nx + nc + nx + 8 + 4, 0));

so = [
        [Constraint((x, u) -> u[8 + (i - 1) * 2 .+ (1:2)], nx, nu) for i = 1:8],
        [[Constraint((x, u) -> u[8 + (i - 1) * 2 .+ (1:2)], 2nx + 4 + 8 + 4, nu) for i = 1:8] for t = 2:T-1]...,
        [Constraint()],
    ] 

# ## initialize
q_interp = CALIPSO.linear_interpolation(q1, qT, T+1)
x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:T]
u_guess = [max.(0.0, 1.0e-3 * randn(nu)) for t = 1:T-1] # may need to run more than once to get good trajectory
x_guess = [t == 1 ? x_interp[t] : [
        x_interp[t]; 
        max.(0.0, 1.0e-3 * randn(nc)); 
        x_interp[t-1];
        1.0e-3 * randn(8);
        max.(0.0, 1.0e-3 * randn(4))] for t = 1:T]

        # ## problem
println("creating solver")
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, CALIPSO.EqualityGeneral(), ineq, so);
trajopt_methods = CALIPSO.ProblemMethods(trajopt);
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# ## solver
solver = Solver(trajopt_methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    custom=trajopt,
    options=Options(
        verbose=true, 
        constraint_tensor=true,
        update_factorization=false,
));

initialize_states!(solver, x_guess);
initialize_actions!(solver, u_guess);
println("solver instantiated and initialized!")

# solve 
solve!(solver)

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver)

# # ## visualize 
using StaticArrays 
StaticArrays.has_size(::Type{SVector}) = false

model = RoboDojo.quadruped
vis = Visualizer() 
open(vis)
q_vis = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
# for i = 1:1
#     horizon = length(q_vis) - 1
#     q_vis = mirror_gait(q_vis, horizon)
# end
RoboDojo.visualize!(vis, model, qm, Δt=h)

# include("visuals/quadruped.jl")
# visualize_meshrobot!(vis, model, q_vis;
#     h=h,
#     anim=MeshCat.Animation(Int(floor(1/h))),
#     name=:quadruped)


# camvis=vis["/Cameras/default/rotated/<object>"]
# setprop!(camvis, "zoom", 25.0)
# MeshCat.settransform!(camvis,
#     MeshCat.compose(
#         MeshCat.LinearMap(Rotations.RotX(-pi / 2.0)),
#         MeshCat.Translation(0.0, -50.0, 0.0),
#     ))
# MeshCat.setvisible!(vis.visualizer["/Grid"], false)
# MeshCat.setvisible!(vis.visualizer["/Axes"], false)

# top_color=RGBA(1,1,1.0)
# bottom_color=RGBA(1,1,1.0)

# MeshCat.setprop!(vis.visualizer["/Background"], "top_color", top_color)
# MeshCat.setprop!(vis.visualizer["/Background"], "bottom_color", bottom_color)



## reference trajectory for CI-MPC

# get trajectories
qq = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
for t = 1:T+1
    qq[t][3] += 0.5 * π
end
uu = [u[1:8] for u in u_sol]
γγ = [u[8 .+ (1:4)] for u in u_sol]
ββ = [u[8 + 4 .+ (1:8)] for u in u_sol]
ηη = [u[8 + 4 + 8 .+ (1:8)] for u in u_sol]
    
# recovered linearized friction terms

function get_βη(βs, ηs)
    βl = zeros(2)
    ηl = [ηs[2] + ηs[1]; -ηs[2] + ηs[1]]
    if (norm(ηl) < 1.0e-3)
        println("hello!!!")
        βl = [1 1; 1 -1] \ βs
        @show βl 
        @show βs
    elseif norm(ηl[1]) < 1.0e-3
        βl[1] = βs[1]
        βl[2] = 0.0
    else
        βl[1] = 0.0
        βl[2] = βs[1]
    end
    return βl, ηl
end

ββl = Vector{Float64}[]
ηηl = Vector{Float64}[]
ψψl = Vector{Float64}[]
for t = 1:T-1
    ββt = Vector{Float64}[] 
    ηηt = Vector{Float64}[]
    ψψt = Vector{Float64}[]
    for i = 1:4
        βt = u_sol[t][8 + 4 + (i - 1) * 2 .+ (1:2)]
        ηt = u_sol[t][8 + 4 + 8 + (i - 1) * 2 .+ (1:2)]
        ψt = u_sol[t][8 + 4 + 8 + (i - 1) * 2 .+ (1:1)]
        βl, ηl = get_βη(βt, ηt)
        push!(ββt, βl)
        push!(ηηt, ηl)
        push!(ψψt, ψt)
    end
    push!(ββl, vcat(ββt...))
    push!(ηηl, vcat(ηηt...))
    push!(ψψl, vcat(ψψt...))
end

perm4 = [0.0 1.0 0.0 0.0;
         1.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 1.0;
		 0.0 0.0 1.0 0.0]

perm8 = [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
         0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
		 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
		 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
		 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]

function mirror_gait(q, u, γ, b, ψ, η, T)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]
	γm = [deepcopy(γ)...]
	bm = [deepcopy(b)...]
	ψm = [deepcopy(ψ)...]
	ηm = [deepcopy(η)...]

	stride = zero(qm[1])
	@show stride[1] = q[T+1][1] - q[2][1]

	for t = 1:T-1
		push!(qm, Array(perm) * q[t+2] + stride)
		push!(um, perm8 * u[t])
		push!(γm, perm4 * γ[t])
		push!(bm, perm8 * b[t])
		push!(ψm, perm4 * ψ[t])
		push!(ηm, perm8 * η[t])
	end

	return qm, um, γm, bm, ψm, ηm
end

qm, um, γm, bm, ψm, ηm = mirror_gait(qq, uu, γγ, ββl, ψψl, ηηl, T)

hm = h
μm = sum(model.friction_foot_world) / length(model.friction_foot_world)

using JLD2
@save joinpath(@__DIR__, "calipso_gait11.jld2") qm um γm bm ψm ηm μm hm
# @load joinpath(@__DIR__, "quadruped_mirror_gait.jld2") qm um γm bm ψm ηm μm hm


using Plots
Plots.plot(hcat(qq...)', color = :black, width = 2.0, label = "")
Plots.plot!(hcat(qm...)', color = :red, width = 1.0, label = "")

Plots.plot(hcat(uu...)', color = :black, width = 2.0, label = "", linetype = :steppost)
Plots.plot!(hcat(um...)', color = :red, width = 1.0, label = "", linetype = :steppost)

Plots.plot(hcat(γγ...)', color = :black, width = 2.0, label = "", linetype = :steppost)
Plots.plot!(hcat(γm...)', color = :red, width = 1.0, label = "", linetype = :steppost)

Plots.plot(hcat(ββl...)', color = :black, width = 2.0, label = "", linetype = :steppost)
Plots.plot!(hcat(bm...)', color = :red, width = 1.0, label = "", linetype = :steppost)

function get_q_viz(q̄)
	q_viz = [q̄...]
	shift_vec = zeros(model.nq)
	shift_vec[1] = q̄[end][1]
	for i = 1:5
		q_update = [q + shift_vec for q in q̄[2:end]]
		push!(q_viz, q_update...)
		shift_vec[1] = q_update[end][1]
	end

	return q_viz
end