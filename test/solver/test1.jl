@testset "Solver problem: Test 1" begin
    num_variables = 50
    num_equality = 30
    num_inequality = 3

    x0 = ones(num_variables)

    obj(x) = transpose(x) * x
    eq(x) = x[1:30].^2 .- 1.2
    ineq(x) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

    # solver
    methods = ProblemMethods(num_variables, obj, eq, ineq)
    solver = Solver(methods, num_variables, num_equality, num_inequality)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end