@testset "Solver problem: Wachter" begin
    # ## problem
    objective(x) = x[1]
    equality(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
    cone(x) = x[2:3]

    # ## variables 
    num_variables = 3

    # ## solver
    solver = Solver(objective, equality, cone, num_variables);

    # ## initialize
    x0 = [-2.0, 3.0, 1.0]
    initialize!(solver, x0)

    # ## solve 
    solve!(solver)

    # ## callbacks 
    solver.options.callback_inner = true 
    solver.options.callback_outer = true 

    function callback_inner(custom, solver)
        println("inner callback")
    end

    function callback_outer(custom, solver)
        println("outer callback")
    end

    # ## solution 
    @show solver.solution.variables # x* = [1.0, 0.0, 0.5]

    # ## test solution
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
