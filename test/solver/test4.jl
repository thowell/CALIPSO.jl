@testset "Solver problem: Test 4" begin 
    num_variables = 3
    num_equality = 0
    num_inequality = 1
    x0 = rand(num_variables)
    obj(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
    eq(x) = zeros(0)
    ineq(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

    # solver
    methods = ProblemMethods(num_variables, obj, eq, ineq)
    solver = Solver(methods, num_variables, num_equality, num_inequality)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end
