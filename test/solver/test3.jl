@testset "Solver problem: Test 3" begin
    num_variables = 2
    num_parameters = 0
    num_equality = 0
    num_cone = 2
    x0 = rand(num_variables)

    obj(x, θ) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
    eq(x, θ) = zeros(0)
    cone(x, θ) = [-(x[1] - 1.0)^3 + x[2] - 1.0;
                -x[1] - x[2] + 2.0]

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)

    # test solution
    opt_norm = max(
        norm(solver.data.residual[solver.indices.variables], Inf),
        norm(solver.data.residual[solver.indices.cone_slack], Inf),
        # norm(λ - y, Inf),
    )
    @test opt_norm < solver.options.optimality_tolerance

    slack_norm = max(
                    norm(solver.data.residual[solver.indices.equality_dual], Inf),
                    norm(solver.data.residual[solver.indices.cone_dual], Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end
