@testset "Solver problem: Wachter" begin
    num_variables = 3
    num_parameters = 0
    num_equality = 2
    num_cone = 2
    x0 = [-2.0, 3.0, 1.0]

    obj(x, θ) = x[1]
    eq(x, θ) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
    cone(x, θ) = x[2:3]

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
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

    @test norm(solver.solution.variables[1:3] - [1.0; 0.0; 0.5], Inf) < 1.0e-3
end