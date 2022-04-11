@testset "Solver: Minimum variable portfolio" begin 
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
    num_equality = 0 
    num_cone = 14 

    idx_ineq = collect(1:2) 
    idx_soc = [collect(2 .+ (1:12))]

    obj(x) = dot(c, x)
    eq(x) = zeros(0)
    cone(x) = b - A * x

    # solver
    methods = ProblemMethods(num_variables, obj, eq, cone)
    solver = Solver(methods, num_variables, num_equality, num_cone;
        nonnegative_indices=idx_ineq,
        second_order_indices=idx_soc,)

    x0 = randn(num_variables)
    initialize!(solver, x0)

    # solve 
    solve!(solver)

    @test norm(solver.data.residual, Inf) < 1.0e-5
    @test norm(solver.problem.cone_product, Inf) < 1.0e-3
    @test all(solver.variables[solver.indices.cone_slack][1:2] .> -1.0e-5)
    @test norm(solver.variables[solver.indices.cone_slack][4:14]) < solver.variables[solver.indices.cone_slack][3] + 1.0e-5
    @test norm(b - A * solver.variables[solver.indices.variables] - solver.variables[solver.indices.cone_slack], Inf) < 1.0e-5
end