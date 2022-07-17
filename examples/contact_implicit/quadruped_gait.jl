# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RoboDojo 
using LinearAlgebra
include("models/robodojo.jl")
include("models/quadruped_template.jl")

# ## horizon
horizon = 41

# ## time steps
timestep = 0.01
fixed_timesteps = 5

# ## RoboDojo dynamics 
model = RoboDojo.quadruped4
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)

# ## dynamics
dynamics = [(y, x, u) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon-1]

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

perm4 = [0.0 1.0 0.0 0.0;
         1.0 0.0 0.0 0.0;
         0.0 0.0 0.0 1.0;
         0.0 0.0 1.0 0.0]

# ## reference
state_reference = trotting_gait(model, horizon; 
    timestep=0.01, 
    velocity=0.25, 
    body_height=0.25, 
    body_forward_position=0.05,
    foot_height=0.05)

# ## objective
total_mass = 0.0 
total_mass += model.m_torso
total_mass += model.m_thigh1
total_mass += model.m_calf1
total_mass += model.m_thigh2
total_mass += model.m_calf2
total_mass += model.m_thigh3
total_mass += model.m_calf3
total_mass += model.m_thigh4
total_mass += model.m_calf4

slack_reference = [0.0; total_mass * model.gravity; 0.0]

objective = Function[]
for t = 1:horizon
    push!(objective, (x, u) -> begin 
            J = 0.0 
            ## controls
            t < horizon && (J += 1.0e-1 * dot(u[1:3] - slack_reference, u[1:3] - slack_reference))
            t < horizon && (J += 1.0e-1 * dot(u[3 .+ (1:8)], u[3 .+ (1:8)]))
            ## kinematic reference
            J += 100.0 * dot(x[11 .+ (1:11)] - state_reference[t][1:11], x[11 .+ (1:11)] - state_reference[t][1:11])
            ## velocity 
            v = (x[11 .+ (1:11)] - x[1:11]) ./ timestep
            J += 1.0e-5 * dot(v, v)
            return J
        end
    );
end

# ## equality constraints
equality = Function[]
push!(equality, (x, u) -> begin 
        q = x[model.nq .+ (1:model.nq)]
        q̄ = state_reference[1][1:model.nq]
        [
            ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
            ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
            ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
            ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
            100.0 * (x[11 .+ (1:11)] - state_reference[1][1:11]);
            u[1:3];
        ]
    end
);

for t = 2:fixed_timesteps
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[t][1:model.nq]
            [
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[2](q̄) - RoboDojo.quadruped4_contact_kinematics[2](q));
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[4](q̄) - RoboDojo.quadruped4_contact_kinematics[4](q));
                u[1:3];
            ]
        end
    );
end

for t = (fixed_timesteps + 1):(horizon-1) 
    push!(equality, (x, u) -> begin 
            q = x[model.nq .+ (1:model.nq)]
            q̄ = state_reference[t][1:model.nq]
            [
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[1](q̄) - RoboDojo.quadruped4_contact_kinematics[1](q));
                ## 1.0 * (RoboDojo.quadruped4_contact_kinematics[3](q̄) - RoboDojo.quadruped4_contact_kinematics[3](q));
                u[1:3];
            ]
        end
    );
end

push!(equality, (x, u) -> begin  
        [
            100.0 * (x[12] - state_reference[horizon][1]);
        ]
    end
)

function equality_general(z) 
    x1 = z[1:(2 * model.nq)]
    xT = z[sum(num_states[1:horizon-1]) + sum(num_actions) .+ (1:(2 * model.nq))]

    e = x1 - Array(cat(perm, perm, dims=(1, 2))) * xT 

    [
        100.0 * e[2:11]; 
        100.0 * e[11 .+ (2:11)];
    ]
end

# ## cone constraints
nonnegative = robodojo_nonnegative(sim, horizon);
second_order = robodojo_second_order(sim, horizon);

# ## options 
options=Options(
        verbose=true, 
        constraint_tensor=true,
        update_factorization=false,  
        # linear_solver=:LU,  
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    equality_general=equality_general,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options);

# ## options
solver.options.residual_tolerance = 1.0e-3
solver.options.optimality_tolerance = 1.0e-3
solver.options.equality_tolerance = 1.0e-3
solver.options.complementarity_tolerance = 1.0e-3
solver.options.slack_tolerance = 1.0e-3

# ## initialize
state_guess = robodojo_state_initialization(sim, state_reference, horizon)
action_guess = [[slack_reference; 1.0e-3 * randn(8)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_actions!(solver, action_guess)

# ## solve 
solve!(solver)

# ## solution
using JLD2
x_sol, u_sol = CALIPSO.get_trajectory(solver)
q_sol = [x_sol[1][1:model.nq], [x[model.nq .+ (1:model.nq)] for x in x_sol]...]
@save joinpath(@__DIR__, "visuals/quadruped_gait.jld2") q_sol
@load joinpath(@__DIR__, "visuals/quadruped_gait.jld2") q_sol


# ## visualizer 
vis = Visualizer() 
open(vis)

# ## visualize
RoboDojo.visualize!(vis, model, x_sol, 
    Δt=timestep)

include("visuals/quadruped.jl")
visualize_meshrobot!(vis, model, x_sol;
    h=0.01,
    anim=MeshCat.Animation(Int(floor(1/h))),
    name=:quadruped)

pwd()