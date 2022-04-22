@testset "Solver problem: Test 1" begin
    num_variables = 50
    num_parameters = 0
    num_equality = 30
    num_cone = 3

    x0 = ones(num_variables)

    obj(x, θ) = transpose(x) * x
    eq(x, θ) = x[1:30].^2 .- 1.2
    cone(x, θ) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

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