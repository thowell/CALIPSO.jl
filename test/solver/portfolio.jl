@testset "Solver: Minimum variance portfolio" begin 
    """ 
    Create a portfolio optimization problem with p dimensions 
    https://arxiv.org/pdf/1705.00772.pdf
    """

    p = 10
    E = randn(p, p)
    Σ = E' * E

    Σ½ = sqrt(Σ)

    # setup for cone problem
    c = [zeros(p); 1.0]

    G1 = [2.0 * Σ½ zeros(p, 1); zeros(1, p) -1.0] 
    h = [zeros(p); 1.0] 
    q = [zeros(p); 1.0] 
    z = 1.0 

    G2 = [ones(1, p) 0.0] 
    G3 = [-ones(1, p) 0.0] 

    A = [G2; G3; -q'; -G1]
    b = [1.0; -1.0; z; h]

    num_variables = p + 1 
    num_parameters = 0
    num_equality = 0 
    num_cone = 14 

    idx_ineq = collect(1:2) 
    idx_soc = [collect(2 .+ (1:12))]

    obj(x, θ) = dot(c, x)
    eq(x, θ) = zeros(0)
    cone(x, θ) = b - A * x


    # methods.cone_jacobian_variables(zeros(num_cone), x0, zeros(num_parameters))
    # methods.cone_jacobian_variables(problem.cone_jacobian_variables, x, θ)
    # solver
    method = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(method, num_variables, num_parameters, num_equality, num_cone;
        nonnegative_indices=idx_ineq,
        second_order_indices=idx_soc,)

    x0 = randn(num_variables)
    initialize!(solver, x0)

    # solve 
    solve!(solver)

    # test solution
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