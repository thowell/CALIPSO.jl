@testset "Solver problem: Test 4" begin 
    # ## problem
    objective(x) = x[1] - 2.0 * x[2] + x[3] + sqrt(6)
    cone(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

    # ## variables 
    num_variables = 3

    # ## solver
    solver = Solver(objective, empty_constraint, cone, num_variables);
    
    # ## initialize 
    x0 = rand(num_variables)
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
