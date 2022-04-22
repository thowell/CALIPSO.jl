@testset "Solver problem: Test 4" begin 
    num_variables = 3
    num_parameters = 0
    num_equality = 0
    num_cone = 1
    x0 = rand(num_variables)
    obj(x, θ) = x[1] - 2*x[2] + x[3] + sqrt(6)
    eq(x, θ) = zeros(0)
    cone(x, θ) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

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
end
