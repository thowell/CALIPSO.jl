struct BallInCup{T} 
    num_configuration::Int 
    num_action::Int 
    num_parameter::Int

    mass_cup::T 
    mass_ball::T 
    string_length::T 

    gravity_cup::T 
    gravity_ball::T 
end

function mass_matrix(model::BallInCup, q::Vector)
    Diagonal([
        model.mass_cup,
        model.mass_cup,
        model.mass_ball,
        model.mass_ball,
    ])
end

function dynamics_bias(model::BallInCup, q::Vector, q̇::Vector) 
    [
        0.0;
        model.mass_cup * model.gravity_cup;
        0.0; 
        model.mass_ball * model.gravity_ball;
    ]
end

function signed_distance(model::BallInCup, q::Vector) 
    position_cup = q[1:2] 
    position_ball = q[2 .+ (1:2)] 

    Δ = position_cup - position_ball 
    return [model.string_length^2 - dot(Δ, Δ)]
end

function impact_jacobian(model::BallInCup, q::Vector)
    position_cup = q[1:2] 
    position_ball = q[2 .+ (1:2)] 

    Δ = position_cup - position_ball

    return -2.0 * transpose(Δ) * [1.0 * I(2) -1.0 * I(2)]
end

function lagrangian_derivatives(model, q, v)
    D1L = -1.0 * dynamics_bias(model, q, v)
    D2L = mass_matrix(model, q) * v
    return D1L, D2L
end

function implicit_dynamics(model::BallInCup, h, q0, q1, u1, λ1, q2)
    # evalutate at midpoint
    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(model, qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(model, qm2, vm2)

    d = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
    d .+= [1.0 * I(2); zeros(2, 2)] * u1           
    d .+= λ1                                              

    return d
end

function ballincup_dynamics(model::BallInCup, timestep, y, x, u)
    # configurations
    q1⁻ = x[1:4]
    q2⁻ = x[4 .+ (1:4)]
    q2⁺ = y[1:4]
    q3⁺ = y[4 .+ (1:4)]

    # control
    u_control = u[1:2]

    # impact 
    γ = u[2 .+ (1:1)]
    J = impact_jacobian(model, q2⁺)
    λ = transpose(J) * γ[1]

    [
        q2⁺ - q2⁻;
        implicit_dynamics(model,
            timestep, q1⁻, q2⁺, u_control, λ, q3⁺);
    ]
end

function ballincup_discrete(model::BallInCup, timestep, y, x, u)
    [
        ballincup_dynamics(model, timestep, y, x, u);
        y[8 .+ (1:1)] - u[2 .+ (1:1)];
    ]
end

function contact_constraints_inequality_t(model, timestep, x, u)
    q3 = x[4 .+ (1:4)]
    γ = u[2 .+ (1:1)]
    ϕ = signed_distance(model, q3)
    [
        ϕ; 
        γ;
    ]
end

function contact_constraints_inequality_T(model, timestep, x, u)
    q3 = x[4 .+ (1:4)]
    ϕ = signed_distance(model, q3)
    [
        ϕ; 
    ]
end

function contact_constraints_equality_t(model, timestep, x, u)
    q3 = x[4 .+ (1:4)]
    γ⁻ = x[8 .+ (1:1)]
    ϕ = signed_distance(model, q3)
    
    [
        γ⁻ .* ϕ; 
    ]
end

# model
ballincup = BallInCup(4, 2, 0, 1.0, 0.01, 1.0, 0.0, 10.0)

# visuals
function build_robot!(vis, model::BallInCup;
    r_cup=0.1,
    r_ball=0.05,
    tl=1.0,
    cup_color=RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl),
    ball_color=RoboDojo.Colors.RGBA(1.0, 0.0, 0.0, tl),
    )

    RoboDojo.setobject!(vis["cup"], RoboDojo.GeometryBasics.Sphere(RoboDojo.Point3f0(0),
        convert(Float32, r_cup)),
        RoboDojo.MeshPhongMaterial(color=cup_color))

    RoboDojo.setobject!(vis["ball"], RoboDojo.GeometryBasics.Sphere(RoboDojo.Point3f0(0),
        convert(Float32, r_ball)),
        RoboDojo.MeshPhongMaterial(color=ball_color))
end

function set_robot!(vis, model::BallInCup, q)
    RoboDojo.settransform!(vis["cup"],
        RoboDojo.Translation(q[1], 0.0, q[2]))
    RoboDojo.settransform!(vis["ball"],
        RoboDojo.Translation(q[3], 0.0, q[4]))
end

function set_background!(vis; 
    top_color=RoboDojo.RGBA(1,1,1.0), 
    bottom_color=RoboDojo.RGBA(1,1,1.0))

    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize!(vis, model::BallInCup, q;
    r_cup=0.1,
    r_ball=0.05,
    tl=1.0,
    cup_color=RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl),
    ball_color=RoboDojo.Colors.RGBA(1.0, 0.0, 0.0, tl),
    Δt=0.1,
    fixed_camera=true)

    set_background!(vis)

    build_robot!(vis, model,
        r_cup=r_cup,
        r_ball=r_ball,
        tl=tl,
        cup_color=cup_color,
        ball_color=ball_color,
        ) 

    anim = RoboDojo.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(q)
    for t = 1:T
        RoboDojo.MeshCat.atframe(anim, t) do
            set_robot!(vis, model, q[t])
        end
    end

    if fixed_camera
        RoboDojo.MeshCat.settransform!(vis["/Cameras/default"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(0.0, -50.0, -1.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(-pi / 2.0))))
        RoboDojo.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 25)
    end

    RoboDojo.MeshCat.setvisible!(vis["/Grid"], false)
    RoboDojo.MeshCat.setvisible!(vis["/Axes"], false)

    RoboDojo.MeshCat.setanimation!(vis, anim)
end
