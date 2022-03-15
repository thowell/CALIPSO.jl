@testset "LU solver" begin
    n = 20
    m = 10
    A = rand(n, n)
    X = rand(n, m)
    B = rand(n, m)
    x = rand(n)
    b = rand(n)

    solver = CALIPSO.empty_solver(A) 
    @test solver isa CALIPSO.LinearSolver

    solver = CALIPSO.lu_solver(A)
    CALIPSO.linear_solve!(solver, X, A, B)
    @test norm(A * X - B, Inf) < 1.0e-10

    solver = CALIPSO.lu_solver(A)
    CALIPSO.linear_solve!(solver, x, A, b)
    @test norm(A * x - b, Inf) < 1.0e-10


end