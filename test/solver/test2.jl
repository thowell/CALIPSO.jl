@testset "Solver problem: Test 2" begin
    num_variables = 2
    num_parameters = 0
    num_equality = 0
    num_cone = 2
    x0 = rand(num_variables)

    obj(x, θ) = -x[1]*x[2] + 2/(3*sqrt(3))
    eq(x, θ) = zeros(0)
    cone(x, θ) = [-x[1] - x[2]^2 + 1.0;
                x[1] + x[2]]

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end
