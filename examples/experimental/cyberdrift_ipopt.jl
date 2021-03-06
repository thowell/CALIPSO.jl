# @testset "Examples: CYBERDRIFT" begin
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
using DirectTrajectoryOptimization
const DTO = DirectTrajectoryOptimization
using RoboDojo 
const RD = RoboDojo 

"""
    CYBERTRUCK
"""
struct CYBERTRUCK{T} <: RoboDojo.Model{T}
    # dimensions
    nq::Int # generalized coordinates
    nu::Int # controls
    nw::Int # parameters
    nc::Int # contact points

    mass::T
    inertia::T

    kinematics_front::Vector{T} 
    kinematics_rear::Vector{T}

    friction_body_world::Vector{T}
    friction_joint::Vector{T} 
end

# skew-symmetric matrix
function hat(x)
    return [0 -x[3] x[2];
            x[3] 0 -x[1];
        -x[2] x[1] 0]
end

function mm(model::CYBERTRUCK, q) 
    Diagonal([model.mass, model.mass, model.inertia])
end

function db(model::CYBERTRUCK, q, q̇) 
    [0.0; 0.0; 0.0]
end

function input_jacobian(model::CYBERTRUCK, q)
    [
        cos(q[3]) sin(q[3]) 0.0; 
        0.0       0.0       1.0;
    ]
end

function contact_jacobian(model::CYBERTRUCK, q)

    R = [cos(q[3]) -sin(q[3]); sin(q[3]) cos(q[3])] 

    r_front = [R * model.kinematics_front; 0.0] 
    r_rear  = [R * model.kinematics_rear;  0.0]

    C1 = [1.0 0.0 0.0; 0.0 1.0 0.0] * -transpose(hat(r_front)) * [0.0; 0.0; 1.0]
    C2 = [1.0 0.0 0.0; 0.0 1.0 0.0] * -transpose(hat(r_rear))  * [0.0; 0.0; 1.0]

    [
        I(2) C1;
        I(2) C2;
    ]
end

# nominal configuration 
function nominal_configuration(model::CYBERTRUCK)
    [0.0; 0.0; 0.0]
end

# friction coefficients 
friction_coefficients(model::CYBERTRUCK) = model.friction_body_world

function dynamics_discrete(model, mass_matrix, dynamics_bias, timestep, q0, q1, u1, w1, λ1, q2)
    # evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / timestep[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / timestep[1]

    D1L1, D2L1 = RoboDojo.lagrangian_derivatives(mass_matrix, dynamics_bias, qm1, vm1)
    D1L2, D2L2 = RoboDojo.lagrangian_derivatives(mass_matrix, dynamics_bias, qm2, vm2)

    d = 0.5 * timestep[1] * D1L1 + D2L1 + 0.5 * timestep[1] * D1L2 - D2L2 # variational integrator (midpoint)
    d .+= transpose(input_jacobian(model, qm2)) * u1        # control inputs
    d .+= λ1                                                # contact impulses

    return d
end

# ## Dimensions
nq = 3 # configuration dimension
nu = 2 # control dimension
nw = 0 # parameters
nc = 2 # number of contact points

# ## Parameters
body_mass = 1.0
body_inertia = 0.1
friction_body_world = [0.5; 0.25]  # coefficient of friction
kinematics_front = [0.1; 0.0] 
kinematics_rear =  [-0.1; 0.0]

# ## Model
cybertruck = CYBERTRUCK(nq, nu, nw, nc,
        body_mass, body_inertia,
        kinematics_front, kinematics_rear,
        friction_body_world, zeros(0))

# ## horizon
horizon = 15
timestep = 0.1

# ## Optimization Dimensions 
nx = 2 * nq
nu = 2 + nc * 6

num_states = [nx for t = 1:horizon]
num_actions = [nu for t = 1:horizon-1]

# ## dynamics
function cybertruck_dynamics(model::CYBERTRUCK, timestep, y, x, u)
    
    # configurations
    q1⁻ = x[1:3]
    q2⁻ = x[3 .+ (1:3)]
    q2⁺ = y[1:3]
    q3⁺ = y[3 .+ (1:3)]

    # control
    u_control = u[1:2]

    # friction
    β1 = u[2 .+ (1:3)] 
    β2 = u[8 .+ (1:3)]
    b1 = β1[2:3]
    b2 = β2[2:3]

    # contact impulses
    J = contact_jacobian(model, q2⁺)
    λ = transpose(J) * [b1; b2] 

    [
        q2⁺ - q2⁻;
        dynamics_discrete(model, q -> mm(model, q), (q, q̇) -> db(model, q, q̇),
            timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺);
    ]
end

function contact_equality(model, timestep, x, u)
    # configurations
    q2 = x[1:3]
    q3 = x[3 .+ (1:3)]

    # friction primals and duals
    β1 = u[2 .+ (1:3)] 
    η1 = u[2 + 3 .+ (1:3)] 
    β2 = u[2 + 6 .+ (1:3)] 
    η2 = u[2 + 6 + 3 .+ (1:3)]

    # friction coefficient
    μ = model.friction_body_world[1:2]

    # contact point velocities
    v = contact_jacobian(model, q3) * (q3 - q2) ./ timestep[1]

    [
        β1[1] - μ[1] * model.mass * 9.81 * timestep[1];
        β2[1] - μ[2] * model.mass * 9.81 * timestep[1];
        v[1:2] - η1[2:3];
        v[3:4] - η2[2:3];
        CALIPSO.second_order_product(β1, η1);
        CALIPSO.second_order_product(β2, η2);
    ]
end

# ## dynamics
dynamics = [(y, x, u) -> cybertruck_dynamics(cybertruck, [timestep], y, x, u) for t = 1:horizon-1]

# ## states
state_initial = [0.0; 1.5; -0.5 * π; 0.0; 1.5; -0.5 * π] 
state_goal = [3.0; 0.0; 0.5 * π; 3.0; 0.0; 0.5 * π]

# ## objective
function obj1(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    # vc = contact_jacobian(model, x[4:6]) * v 
    # J += 0.5 * 1.0e-5 * v[3]^2.0
    # J += 0.5 * 1.0e-5 * dot(vc, vc)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6 * nc)]) * u
    return J
end

function objt(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0 * ones(2); 1.0e-5 * ones(6 * nc)]) * u
    return J
end

function objT(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 5.0 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - state_goal)
    return J
end

objective = [
    obj1, 
    [objt for t = 2:horizon-1]..., 
    objT,
];

# ## constraints
function equality_1(x, u)
    [
        contact_equality(cybertruck, timestep, x, u);
        x - state_initial;
    ]
end

function equality_t(x, u)
    [
        contact_equality(cybertruck, timestep, x, u);
    ]
end

function equality_T(x, u)
    [
        (x - state_goal)[1:3];
    ]
end

equality = [
    equality_1, 
    [equality_t for t = 2:horizon-1]..., 
    equality_T,
];

num_eq = [length(equality[t](rand(num_states[t]), rand(t == horizon ? 0 : num_actions[t]))) for t = 1:horizon]

u_min = [0.0; -0.5]
u_max = [25.0;  0.5]
p_car1 = [3.0, 1.0 * 0.65]
p_car2 = [3.0, 1.0 * -0.65]
circle_obstacle(x, p; r=0.5) = (x[1] - p[1])^2.0 + (x[2] - p[2])^2.0 - r^2.0
nonnegative = [
        [(x, u) -> -1.0 * [
                u_max - u[1:2]; 
                u[1:2] - u_min;
                circle_obstacle(x, p_car1, r=0.1); 
                circle_obstacle(x, p_car2, r=0.1);
                u[2 .+ (1:3)][1] - norm(u[2 .+ (1:3)][2:end]);
                u[2 + 3 .+ (1:3)][1] - norm(u[2 + 3 .+ (1:3)][2:end]);
                u[2 + 6 .+ (1:3)][1] - norm(u[2 + 6 .+ (1:3)][2:end]);
                u[2 + 9 .+ (1:3)][1] - norm(u[2 + 9 .+ (1:3)][2:end]);
            ] for t = 1:horizon-1]..., 
        empty_constraint,
]

num_ineq = [length(nonnegative[t](rand(num_states[t]), rand(t == horizon ? 0 : num_actions[t]))) for t = 1:horizon]

# second_order = [
#     [
#         [
#             (x, u) -> u[2 .+ (1:3)], 
#             (x, u) -> u[2 + 3 .+ (1:3)],
#             (x, u) -> u[2 + 6 .+ (1:3)],
#             (x, u) -> u[2 + 9 .+ (1:3)],
#         ] for t = 1:horizon-1]..., 
#     [empty_constraint],
# ]

# ## solver 
# solver = Solver(objective, dynamics, num_states, num_actions; 
#     equality=equality,
#     nonnegative=nonnegative,
#     second_order=second_order,
#     );

# ## DTO objects 
eval_hess = true
o1 = DTO.Cost((x, u, w) -> objective[1](x, u), num_states[1], num_actions[1],
    evaluate_hessian=eval_hess)
ot = DTO.Cost((x, u, w) -> objective[2](x, u), num_states[2], num_actions[2],
    evaluate_hessian=eval_hess)
oT = DTO.Cost((x, u, w) -> objective[horizon](x, u), num_states[horizon], 0,
    evaluate_hessian=eval_hess)
obj = [o1, [ot for t = 2:horizon-1]..., oT]

d1 = DTO.Dynamics((y, x, u, w) -> dynamics[1](y, x, u), num_states[2], num_states[1], num_actions[1], 
    evaluate_hessian=eval_hess) 
dt = DTO.Dynamics((y, x, u, w) -> dynamics[2](y, x, u), num_states[3], num_states[2], num_actions[2], 
    evaluate_hessian=eval_hess) 
dyn = [d1, [dt for t = 2:horizon-1]...]

cons = [DTO.Constraint((x, u, w) -> [equality[t](x, u); nonnegative[t](x, u)], num_states[t], t == horizon ? 0 : num_actions[t], indices_inequality=collect(num_eq[t] .+ (1:num_ineq[t])), 
    evaluate_hessian=eval_hess) for t = 1:horizon]

bnd1 = Bound(num_states[1], num_actions[1])
bndt = Bound(num_states[2], num_actions[2])
bndT = Bound(num_states[horizon], 0)
bounds = [bnd1, [bndt for t = 2:horizon-1]..., bndT]

# ## problem 
solver = DTO.Solver(dyn, obj, cons, bounds,
    evaluate_hessian=eval_hess,
    options=DTO.Options(
        max_iter=2000,
        tol=1.0e-4,
        constr_viol_tol=1.0e-4))

# ## solver 
# solver = Solver(objective, dynamics, num_states, num_actions,
#     equality=equality,
#     nonnegative=nonnegative,
#     options=Options()
#     );

state_guess = linear_interpolation(state_initial, state_goal, horizon);
action_guess = [[1.0e-3 * randn(2); vcat([[1.0; 0.1; 0.1] for i = 1:(2 * nc)]...)] for t = 1:horizon-1]; # may need to run more than once to get good trajectory
DTO.initialize_states!(solver, state_guess);
DTO.initialize_actions!(solver, action_guess);


# ## solve
@time DTO.solve!(solver)
x_sol, u_sol = DTO.get_trajectory(solver);


# ## initialize


# ## solve
# solver.options.linear_solver = :LU
# solver.options.residual_tolerance=1.0e-3
# solver.options.optimality_tolerance=1.0e-3
# solver.options.equality_tolerance=1.0e-3
# solver.options.complementarity_tolerance=1.0e-3
# solver.options.slack_tolerance=1.0e-3
# solve!(solver)

# @show solver.problem.objective[1]
# x_sol, u_sol = CALIPSO.get_trajectory(solver)
# @show (x_sol[end][4:6] - x_sol[end][1:3]) ./ timestep

# # test solution?
# @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

# slack_norm = max(
#                 norm(solver.data.residual.equality_dual, Inf),
#                 norm(solver.data.residual.cone_dual, Inf),
# )
# @test slack_norm < solver.options.slack_tolerance

# @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
# @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
    
# @test !CALIPSO.cone_violation(solver.solution.cone_slack, zero(solver.solution.cone_slack), 0.0, solver.indices.cone_nonnegative, solver.indices.cone_second_order)
# @test !CALIPSO.cone_violation(solver.solution.cone_slack_dual, zero(solver.solution.cone_slack_dual), 0.0, solver.indices.cone_nonnegative, solver.indices.cone_second_order)
# end

# # plot
# using Plots
# plot(hcat(x_sol...)[4:6, :]', label="")
# plot(hcat(u_sol...)[1:2, :]', label=["v" "ω"])

# # meshes
vis = Visualizer()
render(vis)

function set_background!(vis::Visualizer; top_color=RoboDojo.RGBA(1,1,1.0), bottom_color=RoboDojo.RGBA(1,1,1.0))
    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize!(vis, model::CYBERTRUCK, q;
    scale=0.1,
    Δt = 0.1)

    # default_background!(vis)
    path_meshes = joinpath(@__DIR__, "..", "..", "robot_meshes")
    meshfile = joinpath(path_meshes, "cybertruck", "cybertruck.obj")
    obj = RoboDojo.MeshCat.MeshFileObject(meshfile);
    
    RoboDojo.MeshCat.setobject!(vis["cybertruck"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    RoboDojo.MeshCat.setobject!(vis["cybertruck1"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck1"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    RoboDojo.MeshCat.setobject!(vis["cybertruck2"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck2"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    anim = RoboDojo.MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

    for t = 1:length(q)
        RoboDojo.MeshCat.atframe(anim, t) do
            RoboDojo.MeshCat.settransform!(vis["cybertruck"], RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(q[t][1:2]..., 0.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(q[t][3]))))
        end
    end

    RoboDojo.MeshCat.settransform!(vis["/Cameras/default"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(2.0, 0.0, 1.0),RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.0))))
    
    # # add parked cars
    
    
    RoboDojo.MeshCat.settransform!(vis["cybertruck1"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car1...; 0.0]),
        RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.5 * π) * RoboDojo.Rotations.RotX(0))))

    RoboDojo.MeshCat.settransform!(vis["cybertruck2"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car2...; 0.0]),
        RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.5 * π) * RoboDojo.Rotations.RotX(0))))

    RoboDojo.MeshCat.setanimation!(vis, anim)
end

visualize!(vis, cybertruck, [[x_sol[1] for t = 1:10]..., x_sol..., [x_sol[end] for t = 1:10]...], Δt=timestep)
set_background!(vis)

RoboDojo.settransform!(vis["/Cameras/default"],
	RoboDojo.compose(RoboDojo.Translation(0.0, 0.0, 10.0), RoboDojo.LinearMap(RoboDojo.RotY(-pi/2.5))))
RoboDojo.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 3)

path_meshes = joinpath(@__DIR__, "..", "..", "..", "robot_meshes")
meshfile = joinpath(path_meshes, "cybertruck", "cybertruck_transparent.obj")
obj = RoboDojo.MeshCat.MeshFileObject(meshfile);

q_sol = [x_sol[1][1:3], [x[4:6] for x in x_sol]...]
for (t, q) in enumerate(q_sol)
    RoboDojo.MeshCat.setobject!(vis["cybertruck_t$t"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck_t$t"]["mesh"], RoboDojo.MeshCat.LinearMap(0.1 * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))
    RoboDojo.MeshCat.settransform!(vis["cybertruck_t$t"], RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(q[1:2]..., 0.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(q[3]))))
end


