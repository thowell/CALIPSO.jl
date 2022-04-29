@testset "Solver problem: Test 2" begin
    # ## problem 
    objective(x) = -x[1] * x[2] + 2.0 / (3.0 * sqrt(3))
    equality(x) = zeros(0)
    cone(x) = [-x[1] - x[2]^2 + 1.0;
                x[1] + x[2]]

    # ## variables 
    num_variables = 2

    # ## solver
    solver = Solver(objective, equality, cone, num_variables);
    
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
