@testset "Solver problem: Test 3" begin
    num_variables = 2
    num_equality = 0
    num_inequality = 2
    x0 = rand(num_variables)

    obj(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
    eq(x) = zeros(0)
    ineq(x) = [-(x[1] -1)^3 + x[2] - 1;
                -x[1] - x[2] + 2]

    # solver
    methods = ProblemMethods(num_variables, obj, eq, ineq)
    solver = Solver(methods, num_variables, num_equality, num_inequality)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end
