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

    @test norm(solver.data.residual, Inf) < 1.0e-3
    @test norm(solver.variables[1:3] - [1.0; 0.0; 0.5], Inf) < 1.0e-3
end