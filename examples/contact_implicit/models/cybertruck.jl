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

# skew-symass_matrixetric matrix
function hat(x)
    return [0 -x[3] x[2];
            x[3] 0 -x[1];
        -x[2] x[1] 0]
end

function mass_matrix(model::CYBERTRUCK, q) 
    Diagonal([model.mass, model.mass, model.inertia])
end

function dynamics_bias(model::CYBERTRUCK, q, q̇) 
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
        dynamics_discrete(model, q -> mass_matrix(model, q), (q, q̇) -> dynamics_bias(model, q, q̇),
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


# ## Visuals
function set_background!(vis::Visualizer; top_color=RoboDojo.RGBA(1,1,1.0), bottom_color=RoboDojo.RGBA(1,1,1.0))
    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize!(vis, model::CYBERTRUCK, q;
    scale=0.1,
    Δt = 0.1,
    path_meshes=joinpath(@__DIR__, "..", "..", "..", "..", "robot_meshes"))

    # default_background!(vis)
    
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