@testset "Dynamics" begin 
    # ## horizon
    horizon = 3

    # ## dimensions
    num_states = [2 for t = 1:horizon] 
    num_actions = [1 for t = 1:horizon-1] 
    num_parameters = [0 for t = 1:horizon]

    # ## dynamics
    function pendulum_continuous(z, u) 
        mass = 1.0 
        lc = 1.0 
        gravity = 9.81 
        damping = 0.1
        [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
    end

    function pendulum_discrete(y, x, u)
        h = 0.1
        y - (x + h * pendulum_continuous(y, u))
    end

    dynamics = [pendulum_discrete for t = 1:horizon-1]
    dyn = CALIPSO.generate_methods(dynamics, num_states, num_actions, num_parameters, :Dynamics)

    x1 = ones(num_states[1]) 
    u1 = ones(num_actions[1])
    w1 = zeros(num_parameters[1])
    X = [x1 for t = 1:horizon]
    U = [u1 for t = 1:horizon]
    W = [w1 for t = 1:horizon]
    idx_dyn = CALIPSO.constraint_indices(dyn)
    idx_jac = CALIPSO.jacobian_variables_indices(dyn)

    d = zeros(CALIPSO.num_constraint(dyn))
    J = zeros(CALIPSO.num_constraint(dyn), CALIPSO.num_state_action_next_state(dyn))

    sp = CALIPSO.sparsity_jacobian_variables(dyn, num_states, num_actions, row_shift=0)
    j = zeros(length(vcat(sp...)))

    dyn[1].constraint(dyn[1].constraint_cache, x1, x1, u1, w1) 
    # @benchmark $dt.constraint($dt.constraint_cache, $x1, $x1, $u1, $w1) 
    @test norm(dyn[1].constraint_cache - pendulum_discrete(x1, x1, u1)) < 1.0e-6
    dyn[1].jacobian_variables(dyn[1].jacobian_variables_cache, x1, x1, u1, w1) 
    jac_dense = zeros(dyn[1].num_next_state, dyn[1].num_state + dyn[1].num_action + dyn[1].num_next_state)
    for (i, ji) in enumerate(dyn[1].jacobian_variables_cache)
        jac_dense[dyn[1].jacobian_variables_sparsity[1][i], dyn[1].jacobian_variables_sparsity[2][i]] = ji
    end
    jac_fd = FiniteDiff.finite_difference_jacobian(a -> pendulum_discrete(a[num_states[1] + num_actions[1] .+ (1:num_states[2])], a[1:num_states[1]], a[num_states[1] .+ (1:num_actions[1])]), [x1; u1; x1])
    @test norm(jac_dense - jac_fd) < 1.0e-6

    CALIPSO.constraints!(d, idx_dyn, dyn, X, U, W)
    @test norm(vcat(d...) - vcat([pendulum_discrete(X[t+1], X[t], U[t]) for t = 1:horizon-1]...)) < 1.0e-6
    # info = @benchmark CALIPSO.constraints!($d, $idx_dyn, $dynamics, $X, $U, $W) 

    CALIPSO.jacobian_variables!(j, 0, dyn, X, U, W) 
    for (i, idx) in enumerate(vcat(sp...))
        J[idx...] = j[i] 
    end
    @test norm(J - [jac_fd zeros(dyn[2].num_state, dyn[2].num_action + dyn[2].num_next_state); zeros(dyn[2].num_next_state, dyn[1].num_state + dyn[1].num_action) jac_fd]) < 1.0e-6
    # info = @benchmark CALIPSO.jacobian!($j, $idx_jac, $dynamics, $X, $U, $W) 

    x_idx = CALIPSO.state_indices(dyn)
    u_idx = CALIPSO.action_indices(dyn)
    xu_idx = CALIPSO.state_action_indices(dyn)
    xuy_idx = CALIPSO.state_action_next_state_indices(dyn)

    nz = sum([t < horizon ? dyn[t].num_state : dyn[t-1].num_next_state for t = 1:horizon]) + sum([dyn[t].num_action for t = 1:horizon-1])
    z = rand(nz)
    x = [@views z[x_idx[t]] for t = 1:horizon]
    u = [[@views z[u_idx[t]] for t = 1:horizon-1]..., zeros(0)]

    # CALIPSO.trajectory!(x, u, z, x_idx, u_idx)
    z̄ = zero(z)
    for (t, idx) in enumerate(x_idx) 
        z̄[idx] .= x[t] 
    end
    for (t, idx) in enumerate(u_idx) 
        z̄[idx] .= u[t] 
    end

    @test norm(z - z̄) < 1.0e-6
end

