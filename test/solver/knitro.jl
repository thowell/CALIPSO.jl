@testset "Solver problem: Knitro" begin
    num_variables = 8
    num_parameters = 0
    num_equality = 7
    num_cone = 8
    x0 = zeros(num_variables)

    obj(x, θ) = (x[1] - 5)^2 + (2*x[2] + 1)^2
    eq(x, θ) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
                3*x[1] - x[2] - 3.0 - x[6];
                -x[1] + 0.5*x[2] + 4.0 - x[7];
                -x[1] - x[2] + 7.0 - x[8];
                x[3]*x[6];
                x[4]*x[7];
                x[5]*x[8];]
    cone(x, θ) = x

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    x = solver.variables[1:8]

    @test norm(solver.data.residual, Inf) < solver.options.residual_tolerance
    @test abs(x[4]) < 1.0e-4
    @test abs(x[5]) < 1.0e-4
    @test abs(x[6]) < 1.0e-4
    @test abs(x[3] - 2.0) < 1.0e-4
    @test abs(x[7] - 3.0) < 1.0e-4 
    @test abs(x[8] - 6.0) < 1.0e-4
end
