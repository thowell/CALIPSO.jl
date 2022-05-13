# @testset "Examples: CYBERDRIFT" begin
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

# function input_jacobian(model::CYBERTRUCK, q)
#     [
#         cos(q[3]) sin(q[3]) 0.0; 
#         0.0       0.0       1.0;
#     ]
# end

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
    d .+= [
        u1[1] * (cos(q1[3]) * cos(u1[2]) - sin(q1[3]) * cos(u1[2]));
        u1[1] * (sin(q1[3]) * sin(u1[2]) + cos(q1[3]) * sin(u1[2]));
        model.kinematics_front[1] * u1[1] * sin(u1[2]);
    ]        # control inputs
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
friction_body_world = [0.25; 0.25]  # coefficient of friction
kinematics_front = [0.1; 0.0] 
kinematics_rear =  [-0.1; 0.0]

# ## Model
cybertruck = CYBERTRUCK(nq, nu, nw, nc,
        body_mass, body_inertia,
        kinematics_front, kinematics_rear,
        friction_body_world, zeros(0))

# ## horizon
horizon = 26
timestep = 0.1

# ## Dimensions 
num_states = [2 * nq, [2 * nq + nc * 6 for t = 2:horizon]...]
num_actions = [2 for t = 1:horizon-1]

# ## dynamics
function cybertruck_dynamics(model::CYBERTRUCK, timestep, y, x, u)
    
    # configurations
    q1⁻ = x[1:3]
    q2⁻ = x[3 .+ (1:3)]
    q2⁺ = y[1:3]
    q3⁺ = y[3 .+ (1:3)]

    # control
    u_control = u[1:2]

    # friction primals and duals
    β1 = y[6 .+ (1:3)] 
    η1 = y[6 + 3 .+ (1:3)] 
    β2 = y[6 + 6 .+ (1:3)] 
    η2 = y[6 + 6 + 3 .+ (1:3)]
    b1 = β1[2:3]
    b2 = β2[2:3]

    # contact impulses
    J = contact_jacobian(model, q2⁺)
    λ = transpose(J) * [b1; b2] 

    # friction coefficient
    μ = model.friction_body_world[1:2]

    # contact point velocities
    v = contact_jacobian(model, q3⁺) * (q3⁺ - q2⁺) ./ timestep[1]

    [
        q2⁺ - q2⁻;
        dynamics_discrete(model, q -> mm(model, q), (q, q̇) -> db(model, q, q̇),
            timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺);
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
state_initial = [0.0; 0.0; 0.0 * π; 0.0; 0.0; 0.0 * π] 
state_goal = [1.0; 0.0; 0.0 * π; 1.0; 0.0; 0.0 * π]

# ## objective
function obj1(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x[1:6] - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:6] - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0; 1.0]) * u
    return J
end

function objt(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x[1:6] - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:6] - state_goal)
    J += 0.5 * 1.0e-3 * transpose(u) * Diagonal([1.0; 1.0]) * u
    return J
end

function objT(x, u)
    J = 0.0
    v = (x[4:6] - x[1:3]) ./ timestep[1]
    J += 0.5 * 1.0e-3 * dot(v, v)
    J += 0.5 * 1.0e-3 * transpose(x[1:6] - state_goal) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:6] - state_goal)
    return J
end

objective = [
    obj1, 
    [objt for t = 2:horizon-1]..., 
    objT,
];

# ## constraints
equality = [
    (x, u) -> x[1:6] - state_initial, 
    [empty_constraint for t = 2:horizon-1]..., 
    (x, u) -> x[1:6] - state_goal,
];

u_min = [-100.0; -0.5 * π]
u_max = [100.0;  0.5 * π]

q1 = state_initial
u1 = [1.0; -0.5 * π]
u1[1] * (cos(q1[3]) * cos(u1[2]) - sin(q1[3]) * cos(u1[2]))
u1[1] * (sin(q1[3]) * sin(u1[2]) + cos(q1[3]) * sin(u1[2]))
cybertruck.kinematics_front[1] * u1[1] * sin(u1[2])



# p_car1 = [3.0, 2.0 * 0.65]
# p_car2 = [3.0, 2.0 * -0.65]
circle_obstacle(x, p; r=0.5) = (x[1] - p[1])^2.0 + (x[2] - p[2])^2.0 - r^2.0
nonnegative = [
        [(x, u) -> [
                u_max - u[1:2]; 
                u[1:2] - u_min;
                # circle_obstacle(x, p_car1, r=0.1); 
                # circle_obstacle(x, p_car2, r=0.1);
            ] for t = 1:horizon-1]..., 
        empty_constraint,
]

second_order = [
    [empty_constraint],
    [
        [
            (x, u) -> x[6 .+ (1:3)], 
            (x, u) -> x[6 + 3 .+ (1:3)],
            (x, u) -> x[6 + 6 .+ (1:3)],
            (x, u) -> x[6 + 9 .+ (1:3)],
        ] for t = 1:horizon-1]..., 
]

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    nonnegative=nonnegative,
    second_order=second_order,
    );

# ## initialize
state_guess = linear_interpolation(state_initial, state_goal, horizon)
state_guess_augmented = [t == 1 ? state_guess[t] : [state_guess[t]; vcat([[1.0; 0.1; 0.1] for i = 1:(2 * nc)]...)] for t = 1:horizon]
action_guess = [1.0e-3 * randn(2) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess_augmented) 
initialize_controls!(solver, action_guess)

# ## solve
solve!(solver)

x_sol, u_sol = CALIPSO.get_trajectory(solver)

# test solution
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

# meshes
# vis = Visualizer()
# render(vis)

# function set_background!(vis::Visualizer; top_color=RoboDojo.RGBA(1,1,1.0), bottom_color=RoboDojo.RGBA(1,1,1.0))
#     RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
#     RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
# end

# function visualize!(vis, model::CYBERTRUCK, q;
#     scale=0.1,
#     Δt = 0.1)

#     # default_background!(vis)
#     path_meshes = joinpath(@__DIR__, "..", "..", "..", "Research/robot_meshes")
#     meshfile = joinpath(path_meshes, "cybertruck", "cybertruck.obj")
#     obj = RoboDojo.MeshCat.MeshFileObject(meshfile);
    
#     RoboDojo.MeshCat.setobject!(vis["cybertruck"]["mesh"], obj)
#     RoboDojo.MeshCat.settransform!(vis["cybertruck"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

#     anim = RoboDojo.MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

#     for t = 1:length(q)
#         RoboDojo.MeshCat.atframe(anim, t) do
#             RoboDojo.MeshCat.settransform!(vis["cybertruck"], RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(q[t][1:2]..., 0.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(q[t][3]))))
#         end
#     end

#     RoboDojo.MeshCat.settransform!(vis["/Cameras/default"],
#         RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(2.0, 0.0, 1.0),RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.0))))
    
#     # # add parked cars
    
#     # RoboDojo.MeshCat.setobject!(vis["cybertruck_park1"], obj)
#     # RoboDojo.MeshCat.settransform!(vis["cybertruck_park1"]["mesh"],
#     #     RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car1...; 0.0]),
#     #     RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(pi + pi / 2) * RoboDojo.Rotations.RotX(pi / 2.0))))

#     # RoboDojo.MeshCat.setobject!(vis["cybertruck_park2"], obj)
#     # RoboDojo.MeshCat.settransform!(vis["cybertruck_park2"]["mesh"],
#     #     RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car2...; 0.0]),
#     #     RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(pi + pi / 2) * RoboDojo.Rotations.RotX(pi / 2.0))))

#     RoboDojo.MeshCat.setanimation!(vis, anim)
# end

# visualize!(vis, cybertruck, state_guess, Δt=timestep)
# set_background!(vis)