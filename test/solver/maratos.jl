@testset "Solver problem: Maratos" begin
    num_variables = 2
    num_parameters = 0
    num_equality = 1
    num_cone = 0
    x0 = [2.0; 1.0]

    obj(x, θ) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
    eq(x, θ) = [x[1]^2 + x[2]^2 - 1.0]
    cone(x, θ) = zeros(0)

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)

    # test solution
    @test norm(solver.data.residual, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual[solver.indices.equality_dual], Inf),
                    norm(solver.data.residual[solver.indices.cone_dual], Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end