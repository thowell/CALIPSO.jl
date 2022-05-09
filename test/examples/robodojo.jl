# ## dimensions
function state_action_dimensions(sim::RoboDojo.Simulator, horizon)
    num_states = [2 * sim.model.nq, [length(sim.ip.z) + sim.model.nq for t = 2:horizon]...]
    num_actions = [sim.model.nu for t = 1:horizon-1]
    return num_states, num_actions 
end

# ## dynamics
function robodojo_dynamics(sim::RoboDojo.Simulator, y, x, u) 
    # dimensions
    nz = length(sim.ip.z) 
    nθ = length(sim.ip.θ)

    # configurations
    q1 = x[1:sim.model.nq]
    q2⁻ = x[sim.model.nq .+ (1:sim.model.nq)] 
    q2⁺ = y[1:sim.model.nq] 

    # residual
    r = zeros(eltype(y), nz)
    z = y[sim.model.nq .+ (1:nz)] 
    θ = zeros(eltype(y), nθ)
    RoboDojo.initialize_θ!(θ, sim.model, sim.idx_θ, q1, q2⁻, u, sim.dist.w, sim.f, sim.h)

    sim.ip.methods.r!(r, z, θ, [0.0])

    [
        r;
        q2⁺ - q2⁻;
    ]
end

# ## nonnegative constraints
function robodojo_nonnegative(sim::RoboDojo.Simulator, horizon)
    idx = [
        empty_constraint,
        [(x, u) -> [
            x[sim.model.nq .+ sim.idx_z.γ];
            x[sim.model.nq .+ sim.idx_z.sγ];
        ] for t = 2:horizon]...,
    ]
    return idx 
end

# ## second order constraints
function robodojo_second_order(sim::RoboDojo.Simulator, horizon)
    idx = [
        [empty_constraint],
        [
            [
                [(x, u) -> x[[sim.model.nq + sim.idx_z.ψ[i]; sim.model.nq + sim.idx_z.b[i]]]  for i = 1:sim.model.nc]...,
                [(x, u) -> x[[sim.model.nq + sim.idx_z.sψ[i]; sim.model.nq + sim.idx_z.sb[i]]] for i = 1:sim.model.nc]...,
        ] for t = 2:horizon]...,
    ]
    return idx
end

function robodojo_state_initialization(sim::RoboDojo.Simulator, configurations, horizon)
    configuration_stack = [[configurations[t]; configurations[t+1]] for t = 1:horizon]
    states = [t == 1 ? configuration_stack[t] : [
            configuration_stack[t]; 
            ones(sim.model.nc); 
            ones(sim.model.nc); 
            ones(sim.model.nc); 
            0.1 * ones(sim.model.nc); 
            ones(sim.model.nc); 
            0.1 * ones(sim.model.nc);
    ] for t = 1:horizon]

    return states 
end