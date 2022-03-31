@testset "Dynamics" begin 
    T = 3
    num_state = 2 
    num_action = 1 
    num_parameter = 0 
    parameter_dimensions = [num_parameter for t = 1:T]

    function pendulum(z, u, w) 
        mass = 1.0 
        lc = 1.0 
        gravity = 9.81 
        damping = 0.1
        [z[2], (u[1] / ((mass * lc * lc)) - gravity * sin(z[1]) / lc - damping * z[2] / (mass * lc * lc))]
    end

    function euler_implicit(y, x, u, w)
        h = 0.1
        y - (x + h * pendulum(y, u, w))
    end

    dt = Dynamics(euler_implicit, num_state, num_state, num_action, 
        num_parameter=num_parameter);
    dynamics = [dt for t = 1:T-1] 

    x1 = ones(num_state) 
    u1 = ones(num_action)
    w1 = zeros(num_parameter)
    X = [x1 for t = 1:T]
    U = [u1 for t = 1:T]
    W = [w1 for t = 1:T]
    idx_dyn = CALIPSO.constraint_indices(dynamics)
    idx_jac = CALIPSO.jacobian_indices(dynamics)

    d = zeros(CALIPSO.num_constraint(dynamics))
    j = zeros(CALIPSO.num_constraint(dynamics), CALIPSO.num_state_action_next_state(dynamics))

    sp = CALIPSO.sparsity_jacobian(dynamics, [num_state for t = 1:T], [num_action for t = 1:T-1], row_shift=0)

    dt.evaluate(dt.evaluate_cache, x1, x1, u1, w1) 
    # @benchmark $dt.evaluate($dt.evaluate_cache, $x1, $x1, $u1, $w1) 
    @test norm(dt.evaluate_cache - euler_implicit(x1, x1, u1, w1)) < 1.0e-8
    dt.jacobian(dt.jacobian_cache, x1, x1, u1, w1) 
    jac_dense = zeros(dt.num_next_state, dt.num_state + dt.num_action + dt.num_next_state)
    for (i, ji) in enumerate(dt.jacobian_cache)
        jac_dense[dt.jacobian_sparsity[1][i], dt.jacobian_sparsity[2][i]] = ji
    end
    jac_fd = ForwardDiff.jacobian(a -> euler_implicit(a[num_state + num_action .+ (1:num_state)], a[1:num_state], a[num_state .+ (1:num_action)], w1), [x1; u1; x1])
    @test norm(jac_dense - jac_fd) < 1.0e-8

    CALIPSO.constraints!(d, idx_dyn, dynamics, X, U, W)
    @test norm(vcat(d...) - vcat([euler_implicit(X[t+1], X[t], U[t], W[t]) for t = 1:T-1]...)) < 1.0e-8
    # info = @benchmark CALIPSO.constraints!($d, $idx_dyn, $dynamics, $X, $U, $W) 

    CALIPSO.jacobian!(j, sp, dynamics, X, U, W) 
    @test norm(j - [jac_fd zeros(dynamics[2].num_state, dynamics[2].num_action + dynamics[2].num_next_state); zeros(dynamics[2].num_next_state, dynamics[1].num_state + dynamics[1].num_action) jac_fd]) < 1.0e-8
    # info = @benchmark CALIPSO.jacobian!($j, $idx_jac, $dynamics, $X, $U, $W) 

    x_idx = CALIPSO.state_indices(dynamics)
    u_idx = CALIPSO.action_indices(dynamics)
    xu_idx = CALIPSO.state_action_indices(dynamics)
    xuy_idx = CALIPSO.state_action_next_state_indices(dynamics)

    nz = sum([t < T ? dynamics[t].num_state : dynamics[t-1].num_next_state for t = 1:T]) + sum([dynamics[t].num_action for t = 1:T-1])
    z = rand(nz)
    x = [zero(z[x_idx[t]]) for t = 1:T]
    u = [[zero(z[u_idx[t]]) for t = 1:T-1]..., zeros(0)]

    CALIPSO.trajectory!(x, u, z, x_idx, u_idx)
    z̄ = zero(z)
    for (t, idx) in enumerate(x_idx) 
        z̄[idx] .= x[t] 
    end
    for (t, idx) in enumerate(u_idx) 
        z̄[idx] .= u[t] 
    end

    @test norm(z - z̄) < 1.0e-8
    # info = @benchmark CALIPSO.trajectory!($x, $u, $z, $x_idx, $u_idx)
end

