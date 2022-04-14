@testset "Solver problem: Test 4" begin 
    num_variables = 3
    num_parameters = 0
    num_equality = 0
    num_cone = 1
    x0 = rand(num_variables)
    obj(x, θ) = x[1] - 2*x[2] + x[3] + sqrt(6)
    eq(x, θ) = zeros(0)
    cone(x, θ) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

    # solver
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end
