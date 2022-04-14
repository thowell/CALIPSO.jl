@testset "Solver problem: Maratos" begin
    num_variables = 2
    num_parameters = 0
    num_equality = 1
    num_cone = 0
    x0 = [2.0; 1.0]

    obj(x, θ) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
    eq(x, θ) = [x[1]^2 + x[2]^2 - 1.0]
    cone(x, θ) = zeros(0)

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end