@testset "Solver problem: Test 3" begin
    num_variables = 2
    num_parameters = 0
    num_equality = 0
    num_cone = 2
    x0 = rand(num_variables)

    obj(x, θ) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
    eq(x, θ) = zeros(0)
    cone(x, θ) = [-(x[1] -1)^3 + x[2] - 1;
                -x[1] - x[2] + 2]

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < solver.options.residual_tolerance
end
