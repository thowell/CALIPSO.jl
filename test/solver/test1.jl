@testset "Solver problem: Test 1" begin
    # ## problem 
    objective(x) = transpose(x) * x
    equality(x) = x[1:30].^2 .- 1.2
    cone(x) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

    # ## variables
    num_variables = 50

    # ## solver
    solver = Solver(objective, equality, cone, num_variables);

    # ## initialize 
    x0 = ones(num_variables)
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
end