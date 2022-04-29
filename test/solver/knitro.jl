@testset "Solver problem: Knitro" begin
    # ## problem
    objective(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
    equality(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
                3*x[1] - x[2] - 3.0 - x[6];
                -x[1] + 0.5*x[2] + 4.0 - x[7];
                -x[1] - x[2] + 7.0 - x[8];
                x[3]*x[6];
                x[4]*x[7];
                x[5]*x[8];]
    cone(x) = x #.- 1.0e-5

    # ## variables 
    num_variables = 8

    # ## solver
    solver = Solver(objective, equality, cone, num_variables);
    
    # ## initialize
    x0 = zeros(num_variables)
    initialize!(solver, x0)

    # ## solve 
    solve!(solver)
    x = solver.solution.variables[1:8]

    # ## solution
    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

    # @test abs(x[4]) < 1.0e-4
    # @test abs(x[5]) < 1.0e-4
    # @test abs(x[6]) < 1.0e-4
    # @test abs(x[3] - 2.0) < 1.0e-4
    # @test abs(x[7] - 3.0) < 1.0e-4 
    # @test abs(x[8] - 6.0) < 1.0e-4
end
