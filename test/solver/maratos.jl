@testset "Solver problem: Maratos" begin
    num_variables = 2
    num_equality = 1
    num_inequality = 0
    x0 = [2.0; 1.0]

    obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
    eq(x) = [x[1]^2 + x[2]^2 - 1.0]
    ineq(x) = zeros(0)

    # solver
    methods = ProblemMethods(num_variables, obj, eq, ineq)
    solver = Solver(methods, num_variables, num_equality, num_inequality)
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    @test norm(solver.data.residual, Inf) < 1.0e-5
end