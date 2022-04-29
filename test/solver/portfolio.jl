@testset "Solver: Minimum variance portfolio" begin 
    """ 
    Create a portfolio optimization problem with p dimensions 
    https://arxiv.org/pdf/1705.00772.pdf
    """
    # ## setup
    p = 10
    E = randn(p, p)
    Σ = E' * E
    Σ½ = sqrt(Σ)
    c = [zeros(p); 1.0]

    G1 = [2.0 * Σ½ zeros(p, 1); zeros(1, p) -1.0] 
    h = [zeros(p); 1.0] 
    q = [zeros(p); 1.0] 
    z = 1.0 

    G2 = [ones(1, p) 0.0] 
    G3 = [-ones(1, p) 0.0] 

    A = [G2; G3; -q'; -G1]
    b = [1.0; -1.0; z; h]

    # ## problem 
    objective(x) = dot(c, x)
    cone(x) = b - A * x

    # ## variables 
    num_variables = p + 1 

    # ## cone indices
    nonnegative_indices = collect(1:2) 
    second_order_indices = [collect(2 .+ (1:12))]

    # ## solver
    solver = Solver(objective, empty_constraint, cone, num_variables;
        nonnegative_indices=nonnegative_indices,
        second_order_indices=second_order_indices,)

    # ## initialize
    x0 = randn(num_variables)
    initialize!(solver, x0)

    # ## solve 
    solve!(solver)

    # ## solution
    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

    @test norm(solver.problem.cone_product, Inf) < solver.options.complementarity_tolerance
    @test all(solver.solution.cone_slack[1:2] .> -1.0e-5)
    @test norm(solver.solution.cone_slack[4:14]) < solver.solution.cone_slack[3] + 1.0e-5
    @test norm(b - A * solver.solution.variables - solver.solution.cone_slack, Inf) < solver.options.equality_tolerance
end