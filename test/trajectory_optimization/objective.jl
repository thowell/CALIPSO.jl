@testset "Objective" begin
    # ## horizon
    horizon = 3

    # ## dimension
    num_states = [2 for t = 1:horizon]
    num_actions = [1 for t = 1:horizon-1] 
    num_parameters = [0 for t = 1:horizon] 

    # ## objective
    objective = [
            [(x, u) -> dot(x, x) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
            (x, u) -> 10.0 * dot(x, x),
    ]

    obj = CALIPSO.generate_methods(objective, num_states, num_actions, num_parameters, :Cost)

    J = [0.0]
    gradient = zeros(sum(num_states) + sum(num_actions))
    idx_xu = [sum(num_states[1:(t-1)]) + sum(num_actions[1:(t-1)]) .+ (1:(num_states[t] + (t == horizon ? 0 : num_actions[t]))) for t = 1:horizon]
    x1 = ones(num_states[1]) 
    u1 = ones(num_actions[1])
    w1 = zeros(num_parameters[1]) 
    X = [x1 for t = 1:horizon]
    U = [t < horizon ? u1 : zeros(0) for t = 1:horizon]
    W = [w1 for t = 1:horizon]

    obj[1].cost(obj[1].cost_cache, x1, u1, w1)
    obj[1].gradient_variables(obj[1].gradient_variables_cache, x1, u1, w1)
    @test obj[1].cost_cache[1] ≈ objective[1](x1, u1)
    @test norm(obj[1].gradient_variables_cache - [2.0 * x1; 0.2 * u1]) < 1.0e-8

    obj[horizon].cost(obj[horizon].cost_cache, x1, u1, w1)
    obj[horizon].gradient_variables(obj[horizon].gradient_variables_cache, x1, u1, w1)
    @test obj[horizon].cost_cache[1] ≈ objective[horizon](x1, u1)
    @test norm(obj[horizon].gradient_variables_cache - 20.0 * x1) < 1.0e-8

    cc = zeros(1)
    CALIPSO.cost(cc, obj, X, U, W)
    @test cc[1] - sum([objective[t](X[t], U[t]) for t = 1:horizon-1]) - objective[horizon](X[horizon], U[horizon]) ≈ 0.0
    CALIPSO.gradient_variables!(gradient, idx_xu, obj, X, U, W) 
    @test norm(gradient - vcat([[2.0 * x1; 0.2 * u1] for t = 1:horizon-1]..., 20.0 * x1)) < 1.0e-8

end
