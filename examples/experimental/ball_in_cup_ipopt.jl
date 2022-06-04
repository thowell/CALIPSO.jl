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

# @testset "Examples: Ball-in-cup" begin
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

ballincup = BallInCup(4, 2, 0, 1.0, 0.01, 1.0, 0.0, 10.0)

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
    γ = y[8 .+ (1:1)]
    sγ = y[9 .+ (1:1)]
    ϕ = signed_distance(model, q3⁺)

    J = impact_jacobian(model, q2⁺)
    λ = transpose(J) * γ[1]

    [
        q2⁺ - q2⁻;
        implicit_dynamics(model,
            timestep, q1⁻, q2⁺, u_control, λ, q3⁺);
        sγ - ϕ;
        γ .* sγ; 
    ]
end

# ## horizon
horizon = 21
timestep = 0.075

# ## dimensions
num_states = [8, [10 for t = 2:horizon]...] 
num_actions = [2 for t = 1:horizon-1] 

# ## dynamics
dynamics = [(y, x, u) -> ballincup_dynamics(ballincup, [timestep], y, x, u) for t = 1:horizon-1]

# ## states
x1 = [0.0; 0.0; 0.0; -0.99; 0.0; 0.0; 0.0; -0.99]
xT = [0.0; 0.0; 0.0; 0.125; 0.0; 0.0; 0.0; 0.125]

# ## intermediate states
xM1 = [0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0]
dr = sqrt(0.5 * ballincup.string_length^2)
xM2 = [0.0; 0.0; dr; dr; 0.0; 0.0; dr; dr]
# xM2 = [0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0]
tM1 = 11
tM2 = 16

# ## objective
objective = [
    (x, u) -> begin 
        J = 0.0
        v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
        J += 0.5 * 1.0e-1 * dot(v, v)
        Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
        J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
        J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
        return J
    end,
    [
        (x, u) -> begin
            J = 0.0
            v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
            J += 0.5 * 1.0e-1 * dot(v, v)
            Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
            J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
            J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
            return J
        end for t = 2:horizon-1
    ]...,
    (x, u) -> begin 
        J = 0.0
        v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
        J += 0.5 * 1.0e-1 * dot(v, v)
        Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
        J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
        Δcup_ball = x[4 .+ (1:2)] - x[6 .+ (1:2)]
        J += 0.5 * 1.0 * dot(Δcup_ball, Δcup_ball)
        return J
    end
]

# ## constraints
equality = [
        (x, u) -> begin 
            x - x1;
        end, 
        [t == tM1 ? (x, u) -> begin
            x[6 .+ (1:2)] - xM1[6 .+ (1:2)];
        end : (t == tM2 ? (x, u) -> begin x[6 .+ (1:2)] - xM2[6 .+ (1:2)]; end : empty_constraint) for t = 2:horizon-1]..., 
        (x, u) -> begin 
            [
                x[1:2] - xT[1:2];
                x[4 .+ (1:2)] - xT[4 .+ (1:2)];
                x[6 .+ (1:2)] - xT[6 .+ (1:2)];
            ]
        end,
]

num_eq = [
    length(x1), 
    [t == tM1 ? 2 : (t == tM2 ? 2 : 0) for t = 2:horizon-1]..., 
    6,
]

nonnegative = [
        empty_constraint, 
        [(x, u) -> begin
            [
                -1.0 * x[8 .+ (1:2)];
            ]
        end for t = 2:horizon]..., 
]

num_ineq = [0, [2 for t = 2:horizon]...]

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

# ## initialize
x_interpolation = [linear_interpolation(x1, xM1, 11)..., linear_interpolation(xM1, xM2, 6)[2:end]..., linear_interpolation(xM2, xT, 6)[2:end]...]
state_guess = [x_interpolation[1], [[x_interpolation[t]; 1.0e-3 * ones(2)] for t = 2:horizon]...]
action_guess = [1.0e-3 * randn(2) for t = 1:horizon-1] # may need to run more than once to get good trajectory
DTO.initialize_states!(solver, state_guess) 
DTO.initialize_controls!(solver, action_guess)


# ## solve
@time DTO.solve!(solver)

# ## solution
x_sol, u_sol = DTO.get_trajectory(solver)


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

function set_background!(vis::Visualizer; 
    top_color=RoboDojo.RGBA(1,1,1.0), 
    bottom_color=RoboDojo.RGBA(1,1,1.0))

    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize_ballincup!(vis, model::BallInCup, q;
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

## visualize
vis = Visualizer()
render(vis)

q_sol = [[x[1:4] for x in x_sol]..., x_sol[end][4 .+ (1:4)]]
visualize_ballincup!(vis, ballincup, 
    q_sol, 
    # x_interpolation,
    Δt=timestep,
    r_cup=0.1,
    r_ball=0.05)

minimum([signed_distance(ballincup, q) for q in q_sol])
minimum([q[9] for q in x_sol[2:end-1]])
minimum([q[10] for q in x_sol[2:end-1]])