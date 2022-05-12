# @testset "Examples: Hopper gait (nonlinear cone)" begin

# ## horizon
horizon = 51

# ## time steps
timestep = 0.01

# ## RoboDojo dynamics 
include("robodojo.jl")

model = RoboDojo.hopper1
sim = RoboDojo.Simulator(model, 1, 
    h=timestep)

# ## dimensions
num_states, num_actions = state_action_dimensions(sim, horizon)

# ## dynamics
dynamics = [(y, x, u) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon-1]

# ## states
q1 = [0.0; 0.5 + model.foot_radius; 0.0; 0.5]
qM = [0.1; 0.5 + model.foot_radius; 0.0; 0.5]
qT = [0.2; 0.5 + model.foot_radius; 0.0; 0.5]
q_ref = [0.1; 0.75 + model.foot_radius; 0.0; 0.25]

state_initial = [q1; q1]
xM = [qM; qM]
state_goal = [qT; qT]
x_ref = [q_ref; q_ref]

# ## objective
objective = [
    [(x, u) -> begin
        J = 0.0 
        v = (x[5:8] - x[1:4]) ./ timestep
        J += 0.5 * 1.0e-2 * dot(v, v)
        J += 0.5 * transpose(x[1:8] - x_ref) * Diagonal([1.0; 5.0; 1.0; 10.0; 1.0; 5.0; 1.0; 10.0]) * (x[1:8] - x_ref) 
        t != 1 && (J += 1.0 * x[9])
        t != horizon && (J += 0.5 * transpose(u[1:2]) * Diagonal(1.0e-2 * ones(model.nu)) * u[1:2])
        return J
    end for t = 1:horizon]..., 
];

# ## constraints
equality = [
    (x, u) -> begin 
        [
            RoboDojo.kinematics_foot(model, x[1:model.nq]) - RoboDojo.kinematics_foot(model, state_initial[1:model.nq]);
            RoboDojo.kinematics_foot(model, x[model.nq .+ (1:model.nq)]) - RoboDojo.kinematics_foot(model, state_initial[model.nq .+ (1:model.nq)]);
            x[1:4] - q1;
        ]
    end,
    [empty_constraint for t = 2:horizon]...,
]


function equality_general(z) 
    x1 = z[1:(2 * model.nq)]
    xT = z[sum(num_states[1:end-1]) + sum(num_actions) .+ (1:(2 * model.nq))] 

    [
        xT[1:model.nq][collect([2, 3, 4])] - x1[1:model.nq][collect([2, 3, 4])];
        xT[model.nq .+ (1:model.nq)][collect([2, 3, 4])] - x1[model.nq .+ (1:model.nq)][collect([2, 3, 4])];
    ]
end

nonnegative_contact = robodojo_nonnegative(sim, horizon)

nonnegative = [ 
    [(x, u) -> begin
        [
            nonnegative_contact[t](x, u);
            u[1:2] - [-10.0; -10.0]; 
            [10.0; 10.0] - u[1:2];
            x[2];
            x[4];
            x[6];
            x[8];
            1.0 - x[4]; 
            1.0 - x[8];
        ]
    end for t = 1:horizon-1]..., 
    (x, u) -> begin 
        [
            nonnegative_contact[horizon](x, u);
            x[1] - 0.1;
            x[model.nq + 1] - 0.1; 
            x[2];
            x[4];
            x[6];
            x[8];
            1.0 - x[4]; 
            1.0 - x[8];
        ]
    end,
];

second_order = robodojo_second_order(sim, horizon)

# ## options 
options = Options(
    optimality_tolerance=1.0e-3,
    residual_tolerance=1.0e-3,
    equality_tolerance=1.0e-3,
    complementarity_tolerance=1.0e-3,
    slack_tolerance=1.0e-3,
)

# ## solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
    equality=equality,
    equality_general=equality_general,
    nonnegative=nonnegative,
    second_order=second_order,
    options=options,
    );

# ## initialize
state_guess = robodojo_state_initialization(sim, linear_interpolation(state_initial, state_goal, horizon), horizon)
action_guess = [[0.0; model.gravity * model.mass_body * 0.5 * timestep] for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess) 
initialize_controls!(solver, action_guess)

# ## solve 
solve!(solver)

# ## solution
x_sol, u_sol = CALIPSO.get_trajectory(solver)

# @test norm((x_sol[1] - x_sol[horizon][1:8])[[2; 3; 4; 6; 7; 8]], Inf) < 1.0e-3

# # test solution
# @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

# slack_norm = max(
#                 norm(solver.data.residual.equality_dual, Inf),
#                 norm(solver.data.residual.cone_dual, Inf),
# )
# @test slack_norm < solver.options.slack_tolerance

# @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
# @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# # end

vis = Visualizer()
render(vis)
RoboDojo.visualize!(vis, RoboDojo.hopper, x_sol, Δt=timestep);

# using JLD2
# @save joinpath(@__DIR__, "hopper_gait.jl") x_sol u_sol
# @load joinpath(@__DIR__, "hopper_gait.jl") x_sol u_sol

# ## mirror trajectories 
nq = model.nq
nu = model.nu
nc = model.nc
v1_reference = (x_sol[1][nq .+ (1:nq)] - x_sol[1][1:nq]) ./ timestep
q_reference = [x_sol[1][1:nq], [x_sol[t][nq .+ (1:nq)] for t = 1:horizon]...]
u_reference = [u_sol[t][1:nu] for t = 1:horizon-1]
γ_reference = [x_sol[t][2nq .+ (1:nc)] for t = 2:horizon]
sγ_reference = [x_sol[t][2nq + nc .+ (1:nc)] for t = 2:horizon]
ψ_reference = [x_sol[t][2nq + nc + nc .+ (1:nc)] for t = 2:horizon]
b_reference = [x_sol[t][2nq + nc + nc + nc .+ (1:nc)] for t = 2:horizon]
sψ_reference = [x_sol[t][2nq + nc + nc + nc + nc .+ (1:nc)] for t = 2:horizon]
sb_reference = [x_sol[t][2nq + nc + nc + nc + nc + nc .+ (1:nc)] for t = 2:horizon]

stride_shift = zeros(nq)
stride_shift[1] = q_reference[end][1] - q_reference[2][1]
q_mirror = [q_reference[1:end-2]..., [q + stride_shift for q in q_reference]...]
u_mirror = [u_reference..., [u for u in u_reference]...]
γ_mirror = [γ_reference..., [γ for γ in γ_reference]...]
sγ_mirror = [sγ_reference..., [sγ for sγ in sγ_reference]...]
ψ_mirror = [ψ_reference..., [ψ for ψ in ψ_reference]...]
b_mirror = [b_reference..., [b for b in b_reference]...]
sψ_mirror = [sψ_reference..., [sψ for sψ in sψ_reference]...]
sb_mirror = [sb_reference..., [sb for sb in sb_reference]...]

T_mirror = length(q_mirror)

vis = Visualizer() 
render(vis)
RoboDojo.visualize!(vis, model, q_mirror, Δt=timestep)

T_sim = T_mirror-2
x_hist = [[q_reference[2]; v1_reference]]

sim_test = RoboDojo.Simulator(RoboDojo.hopper, 1, 
    h=timestep)
    
for t = 1:T_sim-1
    y = zeros(8)
    RoboDojo.dynamics(sim_test, y, x_hist[end], u_mirror[t] ./ timestep, zeros(0))
    push!(x_hist, y)
end
# using Plots
# plot(hcat(x_hist...)'[:,1:3])

s = Simulator(model, T_sim-1, h=timestep)
for i = 1:T_sim
    q = x_hist[i][1:4]
    v = x_hist[i][4 .+ (1:4)]
    RoboDojo.set_state!(s, q, v, i)
end
visualize!(vis, s)


# ## reference solution 
z̄_reference = [zeros(length(sim.ip.z)) for t = 1:T_mirror-2]
θ̄_reference = [zeros(length(sim.ip.θ)) for t = 1:T_mirror-2]
for t = 1:T_mirror-2
    z̄_reference[t] = [
            q_mirror[t+2]; 
            γ_mirror[t];
            sγ_mirror[t]; 
            ψ_mirror[t]; 
            b_mirror[t];
            sψ_mirror[t]; 
            sb_mirror[t];
    ]
    RoboDojo.initialize_θ!(
        θ̄_reference[t], sim.model, sim.idx_θ, 
        q_mirror[t], q_mirror[t+1], 
        u_mirror[t], 
        sim.dist.w, sim.f, sim.h,
    )
end

function dynamics_linearized(y, x, u, w)
    z̄ = w[1:10]
    θ̄ = w[10 .+ (1:13)]
    robodojo_linearized_dynamics(sim, z̄, θ̄, y, x, u)
end

# ## mpc policy 
horizon_mpc = 20
num_states_mpc, num_actions_mpc = state_action_dimensions(sim, horizon_mpc)

# dynamics
# dynamics_mpc = [dynamics_linearized for t = 1:horizon_mpc-1]
dynamics_mpc = [(y, x, u, w) -> robodojo_dynamics(sim, y, x, u) for t = 1:horizon_mpc-1]

# objective
function objt_mpc(x, u, w) 
    x̄ = w[10 .+ (1:8)] 
    ū = w[10 + 8 .+ (1:2)]
    cost_q = w[23 .+ (1:8)] 
    cost_u = w[23 + 8 .+ (1:2)]
    J = 0.0 
    J += 0.5 * transpose(x[1:8] - x̄) * Diagonal(cost_q) * (x[1:8] - x̄)
    J += 0.5 * transpose(u[1:2] - ū) * Diagonal(cost_u) * (u[1:2] - ū)
    return J 
end

function objT_mpc(x, u, w) 
    x̄ = w[10 .+ (1:8)]
    cost_q = w[23 .+ (1:8)]
    J = 0.0 
    J += 0.5 * transpose(x[1:8] - x̄) * Diagonal(cost_q) * (x[1:8] - x̄)
    return J 
end

objective_mpc = [[objt_mpc for t = 1:horizon_mpc-1]..., objT_mpc]
equality_mpc = [(x, u, w) -> x[1:8] - w[10 .+ (1:8)], [empty_constraint for t = 2:horizon_mpc]...]
nonnegative_contact = robodojo_nonnegative(sim, horizon_mpc, parameters=true);
inequality_mpc = [[(x, u, w) -> [nonnegative_contact[t](x, u, w); 1000.0 .- u[1:2]; u[1:2] .+ 1000.0] for t = 1:horizon_mpc-1]..., empty_constraint]
second_order_mpc = robodojo_second_order(sim, horizon_mpc, parameters=true);

options_mpc = Options(
    constraint_tensor=true,
    residual_tolerance=1.0e-3,
    optimality_tolerance=1.0e-2, 
    equality_tolerance=1.0e-2, 
    complementarity_tolerance=1.0e-2,
    slack_tolerance=1.0e-2,
    update_factorization=false,
    differentiate=true,
)

# parameters 
q_cost = ones(2 * model.nq)
u_cost = ones(model.nu)

parameters_mpc = [[q_mirror[t+2]; 
                γ_mirror[t];
                sγ_mirror[t]; 
                ψ_mirror[t]; 
                b_mirror[t];
                sψ_mirror[t]; 
                sb_mirror[t];
                q_mirror[t];
                q_mirror[t+1]; 
                u_mirror[t];
                sim.dist.w; sim.f; sim.h; q_cost; t < horizon_mpc ? u_cost : zeros(0)] for t = 1:horizon_mpc]

# ## solver 
solver_mpc = Solver(objective_mpc, dynamics_mpc, num_states_mpc, num_actions_mpc;
    parameters=parameters_mpc,
    equality=equality_mpc,
    nonnegative=inequality_mpc,
    second_order=second_order_mpc,
    options=options_mpc);

# ## initialize
state_guess = [t == 1 ? [q_mirror[t]; q_mirror[t+1]] : [q_mirror[t+1]; q_mirror[t+2]; 
    γ_mirror[t];
    sγ_mirror[t]; 
    ψ_mirror[t]; 
    b_mirror[t];
    sψ_mirror[t]; 
    sb_mirror[t];] for t = 1:horizon_mpc]
action_guess = [u_mirror[t] for t = 1:horizon_mpc-1] 
initialize_states!(solver_mpc, state_guess) 
initialize_controls!(solver_mpc, action_guess)

solver_mpc.parameters .= vcat([[q_mirror[t+2]; 
    γ_mirror[t];
    sγ_mirror[t]; 
    ψ_mirror[t]; 
    b_mirror[t];
    sψ_mirror[t]; 
    sb_mirror[t];
    q_mirror[t];
    q_mirror[t+1]; 
    u_mirror[t];
    sim.dist.w; sim.f; sim.h; q_cost; t < horizon_mpc ? u_cost : zeros(0)] for t = 1:horizon_mpc]...)

# # ## solve 
solve!(solver_mpc)

# ## policy 

q_mpc = deepcopy(q_mirror[1:end-2])
u_mpc = deepcopy(u_mirror)
γ_mpc = deepcopy(γ_mirror)
sγ_mpc = deepcopy(sγ_mirror)
ψ_mpc = deepcopy(ψ_mirror)
b_mpc = deepcopy(b_mirror)
sψ_mpc = deepcopy(sψ_mirror)
sb_mpc = deepcopy(sb_mirror)

function policy(x, q_mpc, u_mpc, γ_mpc, sγ_mpc, ψ_mpc, b_mpc, sψ_mpc, sb_mpc; 
    reset=false, 
    t_start=2)

    if reset
        println("policy reset")
        q_mpc .= deepcopy(q_mirror[1:end-2])
        u_mpc .= deepcopy(u_mirror)
        γ_mpc .= deepcopy(γ_mirror)
        sγ_mpc .= deepcopy(sγ_mirror)
        ψ_mpc .= deepcopy(ψ_mirror)
        b_mpc .= deepcopy(b_mirror)
        sψ_mpc .= deepcopy(sψ_mirror)
        sb_mpc .= deepcopy(sb_mirror)

        q_mpc .= [q_mpc[t_start:end]..., [q + stride_shift for q in q_mpc[1:(t_start-1)]]...]
        u_mpc .= [u_mpc[(t_start-1):end]..., u_mpc[1:(t_start-2)]...]
        γ_mpc .= [γ_mpc[(t_start-1):end]..., γ_mpc[1:(t_start-2)]...]
        sγ_mpc .= [sγ_mpc[(t_start-1):end]..., sγ_mpc[1:(t_start-2)]...]
        ψ_mpc .= [ψ_mpc[(t_start-1):end]..., ψ_mpc[1:(t_start-2)]...]
        b_mpc .= [b_mpc[(t_start-1):end]..., b_mpc[1:(t_start-2)]...]
        sψ_mpc .= [sψ_mpc[(t_start-1):end]..., sψ_mpc[1:(t_start-2)]...]
        sb_mpc .= [sb_mpc[(t_start-1):end]..., sb_mpc[1:(t_start-2)]...]

        return u_mpc[1] ./ timestep
    else
        # configurations from state 
        q2 = x[1:model.nq] 
        q1 = q2 - x[model.nq .+ (1:model.nq)] .* timestep 

        # set parameters (reference and initial state)
        q_cost = 1.0 * ones(8) 
        u_cost = 1.0 * ones(2)
        solver_mpc.parameters .= vcat([[q_mpc[t+2]; 
            γ_mpc[t];
            sγ_mpc[t]; 
            ψ_mpc[t]; 
            b_mpc[t];
            sψ_mpc[t]; 
            sb_mpc[t];
            t == 1 ? q1 : q_mpc[t];
            t == 2 ? q2 : q_mpc[t+1]; 
            u_mpc[t];
            sim.dist.w; sim.f; sim.h; q_cost; t < horizon_mpc ? u_cost : zeros(0)] for t = 1:horizon_mpc]...)

        state_guess = [t == 1 ? [q1; q2] : [q_mpc[t]; q_mpc[t+1]; 
            γ_mpc[t];
            sγ_mpc[t]; 
            ψ_mpc[t]; 
            b_mpc[t];
            sψ_mpc[t]; 
            sb_mpc[t];] for t = 1:horizon_mpc]
        action_guess = [u_mpc[t] for t = 1:horizon_mpc-1] 
        initialize_states!(solver_mpc, state_guess) 
        initialize_controls!(solver_mpc, action_guess)

        # optimize
        solver_mpc.options.verbose = false
        solver_mpc.options.penalty_initial = 1.0e5
        solver_mpc.options.residual_tolerance=1.0e-3
        solver_mpc.options.optimality_tolerance=1.0e-3
        solver_mpc.options.equality_tolerance=1.0e-3
        solver_mpc.options.complementarity_tolerance=1.0e-3
        solver_mpc.options.slack_tolerance=1.0e-3
        solve!(solver_mpc)

        # solution 
        x_s, u_s = get_trajectory(solver_mpc)

        # uu = copy(u_mpc[1])
        # shift reference 
        q_mpc .= [q_mpc[2:end]..., q_mpc[1] + stride_shift]
        u_mpc .= [u_mpc[2:end]..., u_mpc[1]]
        γ_mpc .= [γ_mpc[2:end]..., γ_mpc[1]]
        sγ_mpc .= [sγ_mpc[2:end]..., sγ_mpc[1]]
        ψ_mpc .= [ψ_mpc[2:end]..., ψ_mpc[1]]
        b_mpc .= [b_mpc[2:end]..., b_mpc[1]]
        sψ_mpc .= [sψ_mpc[2:end]..., sψ_mpc[1]]
        sb_mpc .= [sb_mpc[2:end]..., sb_mpc[1]]

        # return uu ./ timestep
        # return first control
        # @show u_s[1:3]
        return u_s[1] ./ timestep
    end
end

# @show u_mirror[1]
# policy([q_mirror[2]; v1_reference], 1)

T_sim = 10
t_start = 2
x_hist = [[q_mirror[t_start]; (q_mirror[t_start] - q_mirror[t_start-1]) ./ timestep]]

sim_test = RoboDojo.Simulator(RoboDojo.hopper, 1, 
    h=timestep)


u_test = policy(x_hist[1], q_mpc, u_mpc, γ_mpc, sγ_mpc, ψ_mpc, b_mpc, sψ_mpc, sb_mpc, reset=true, t_start=t_start)
u_mirror[t_start-1] ./ timestep



u1 = policy([q_mirror[t_start]; (q_mirror[t_start] - q_mirror[t_start-1]) ./ timestep], q_mpc, u_mpc, γ_mpc, sγ_mpc, ψ_mpc, b_mpc, sψ_mpc, sb_mpc, t_start=t_start)
u_mirror[t_start-1] ./ timestep

u2 = policy([q_mirror[t_start + 1]; (q_mirror[t_start + 1] - q_mirror[t_start-1 + 1]) ./ timestep], q_mpc, u_mpc, γ_mpc, sγ_mpc, ψ_mpc, b_mpc, sψ_mpc, sb_mpc, t_start=t_start)
u_mirror[t_start-1 + 1] ./ timestep
u_mpc[end] ./ timestep

for t = 1:T_sim-1
    y = zeros(8)
    u_open = u_mirror[t_start + t - 2] ./ timestep 
    u_policy = policy(x_hist[end],q_mpc, u_mpc, γ_mpc, sγ_mpc, ψ_mpc, b_mpc, sψ_mpc, sb_mpc,)
    RoboDojo.dynamics(sim_test, y, x_hist[end], 
        u_open,
        zeros(0))

    @show norm(u_open - u_policy, Inf)
    push!(x_hist, y)
end

s = Simulator(model, T_sim-1, h=timestep)
for i = 1:T_sim
    q = x_hist[i][1:4]
    v = x_hist[i][4 .+ (1:4)]
    RoboDojo.set_state!(s, q, v, i)
end

vis = Visualizer() 
render(vis)
visualize!(vis, s)


