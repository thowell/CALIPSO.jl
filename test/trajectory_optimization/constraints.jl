@testset "Constraints" begin 
    # ## horizon
    horizon = 5

    # ## dimensions
    num_states = [2 for t = 1:horizon]
    num_actions = [1 for t = 1:horizon-1] 
    num_parameters = [0 for t = 1:horizon]

    x = [rand(num_states[t]) for t = 1:horizon] 
    u = [[rand(num_actions[t]) for t = 1:horizon-1]..., zeros(0)]
    w = [rand(num_parameters[t]) for t = 1:horizon]

    con = [
        [(x, u) -> [-ones(num_states[t]) - x; x - ones(num_states[t])] for t = 1:horizon-1]...,
        (x, u) -> x,
    ]

    constraints = CALIPSO.generate_methods(con, num_states, num_actions, num_parameters, :Constraint)

    nc = CALIPSO.num_constraint(constraints)
    nj = CALIPSO.num_jacobian_variables(constraints)
    idx_c = CALIPSO.constraint_indices(constraints)
    idx_j = CALIPSO.jacobian_variables_indices(constraints)
    c = zeros(nc) 
    j = zeros(CALIPSO.num_constraint(constraints), sum(num_states) + sum(num_actions))
    sp = CALIPSO.sparsity_jacobian_variables(constraints, num_states, [num_actions..., 0], row_shift=0)
    j = zeros(length(vcat(sp...)))
    constraints[1].constraint(c[idx_c[1]], x[1], u[1], w[1])
    constraints[horizon].constraint(c[idx_c[horizon]], x[horizon], u[horizon], w[horizon])

    CALIPSO.constraints!(c, idx_c, constraints, x, u, w)
    # info = @benchmark CALIPSO.constraints!($c, $idx_c, $constraints, $x, $u, $w)

    @test norm(c - vcat([con[t](x[t], u[t]) for t = 1:horizon-1]..., con[horizon](x[horizon], u[horizon]))) < 1.0e-8

    CALIPSO.jacobian_variables!(j, 0, constraints, x, u, w)
    J = zeros(CALIPSO.num_constraint(constraints), sum(num_states) + sum(num_actions))
    for (i, idx) in enumerate(vcat(sp...)) 
        J[idx...] = j[i] 
    end
    # info = @benchmark CALIPSO.jacobian!($j, $idx_j, $constraints, $x, $u, $w)

    dct = [-I zeros(num_states[1], num_actions[1]); I zeros(num_states[1], num_actions[1])]
    dcT = Diagonal(ones(num_states[horizon]))
    dc = cat([dct for t = 1:horizon-1]..., dcT, dims=(1,2))

    @test norm(J - dc) < 1.0e-8
end