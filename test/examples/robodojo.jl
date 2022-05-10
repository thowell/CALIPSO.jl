# ## dimensions
function state_action_dimensions(sim::RoboDojo.Simulator, horizon)
    num_states = [2 * sim.model.nq, [sim.model.nq + length(sim.ip.z)  for t = 2:horizon]...]
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

# ## linearized dynamics
function robodojo_linearized_dynamics(sim::RoboDojo.Simulator, z̄, θ̄, y, x, u; 
    indices_linearized=collect(1:(sim.model.nq + sim.model.nc + sim.model.nc)))

    # dimensions
    nz = length(sim.ip.z) 
    nθ = length(sim.ip.θ)

    # indices 
    indices_nonlinear = [i for i = 1:nz if !(i in indices_linearized)]

    # configurations
    q1 = x[1:sim.model.nq]
    q2⁻ = x[sim.model.nq .+ (1:sim.model.nq)] 
    q2⁺ = y[1:sim.model.nq] 

    # residual
    r_nonlinear = zeros(eltype(y), nz)
    z = y[sim.model.nq .+ (1:nz)] 
    θ = zeros(eltype(y), nθ)
    RoboDojo.initialize_θ!(θ, sim.model, sim.idx_θ, q1, q2⁻, [zeros(3); u], sim.dist.w, sim.f, sim.h)
    sim.ip.methods.r!(r_nonlinear, z, θ, [0.0])

    # linearization 
    r̄ = zeros(eltype(y), nz)  
    r̄z = zeros(eltype(y), nz, nz)
    r̄θ = zeros(eltype(y), nz, nθ)
    sim.ip.methods.r!(r̄, z̄, θ̄, [0.0])
    sim.ip.methods.rz!(r̄z, z̄, θ̄)
    sim.ip.methods.rθ!(r̄θ, z̄, θ̄)

    r_linearized = r̄ + r̄z * (z - z̄) + r̄θ * (θ - θ̄)

    [
        # r_nonlinear;
        r_linearized[indices_linearized];
        r_nonlinear[indices_nonlinear];
        q2⁺ - q2⁻;
    ]
end

# ## nonnegative constraints
function robodojo_nonnegative(sim::RoboDojo.Simulator, horizon; 
    parameters=false)
    if parameters
        idx = [
            empty_constraint,
            [(x, u, w) -> [
                x[sim.model.nq .+ sim.idx_z.γ];
                x[sim.model.nq .+ sim.idx_z.sγ];
            ] for t = 2:horizon]...,
        ]
    else 
        idx = [
            empty_constraint,
            [(x, u) -> [
                x[sim.model.nq .+ sim.idx_z.γ];
                x[sim.model.nq .+ sim.idx_z.sγ];
            ] for t = 2:horizon]...,
        ]
    end
    return idx 
end

# ## second order constraints
function robodojo_second_order(sim::RoboDojo.Simulator, horizon; 
    parameters=false)
    if parameters 
        idx = [
            [empty_constraint],
            [
                [
                    [(x, u, w) -> x[[sim.model.nq + sim.idx_z.ψ[i]; sim.model.nq + sim.idx_z.b[i]]]  for i = 1:sim.model.nc]...,
                    [(x, u, w) -> x[[sim.model.nq + sim.idx_z.sψ[i]; sim.model.nq + sim.idx_z.sb[i]]] for i = 1:sim.model.nc]...,
            ] for t = 2:horizon]...,
        ]
    else 
        idx = [
            [empty_constraint],
            [
                [
                    [(x, u) -> x[[sim.model.nq + sim.idx_z.ψ[i]; sim.model.nq + sim.idx_z.b[i]]]  for i = 1:sim.model.nc]...,
                    [(x, u) -> x[[sim.model.nq + sim.idx_z.sψ[i]; sim.model.nq + sim.idx_z.sb[i]]] for i = 1:sim.model.nc]...,
            ] for t = 2:horizon]...,
        ]
    end
    return idx
end

function robodojo_state_initialization(sim::RoboDojo.Simulator, states, horizon)
    states_contacts = [t == 1 ? states[t] : [
            states[t]; 
            1.0e-3 * ones(sim.model.nc); 
            1.0e-3 * ones(sim.model.nc); 
            1.0e-3 * ones(sim.model.nc); 
            1.0e-4 * ones(sim.model.nc); 
            1.0e-3 * ones(sim.model.nc); 
            1.0e-4 * ones(sim.model.nc);
    ] for t = 1:horizon]

    return states_contacts
end

function robodojo_configuration_initialization(sim::RoboDojo.Simulator, configurations, horizon)
    robodojo_state_initialization(sim, [[configurations[t]; configurations[t+1]] for t = 1:horizon], horizon)
end

