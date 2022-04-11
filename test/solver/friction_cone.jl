@testset "Solver: Second-order cone for friction" begin 
    """
        minimize    v' b
        subject to  ||b|| <= μn

        minimize    [0 v]' x 
        subject to  x[1] = μn
                    x in K
    """

    num_variables = 3 
    num_equality = 1 
    num_cone = 3
    idx_ineq = Int[]
    idx_soc = [collect(1:3)]

    V = [[0.0; 0.0; 0.0], [0.0; 1.0; 0.0], [0.0; 0.0; 1.0], [0.0; 1.0; 1.0], [0.0; 10.0; 1.0]]
    friction_coefficients = [0.0, 0.5, 1.0] 
    impact_impulse = [0.0, 1.0] 

    for v in V 
        for μ in friction_coefficients 
            for γ in impact_impulse

                obj(x) = transpose(v) * x
                eq(x) = [x[1] - μ * γ]
                cone(x) = x

                # solver
                methods = ProblemMethods(num_variables, obj, eq, cone)
                solver = Solver(methods, num_variables, num_equality, num_cone;
                    nonnegative_indices=idx_ineq,
                    second_order_indices=idx_soc,)

                x0 = randn(num_variables)
                initialize!(solver, x0)

                # solve 
                solve!(solver)

                @test norm(solver.data.residual, Inf) < 1.0e-3 
                @test !CALIPSO.cone_violation(solver.variables, 
                    solver.indices.cone_nonnegative, solver.indices.cone_second_order)
                if norm(v[2:3]) > 0.0 && γ > 0.0 && μ > 0.0
                    v_dir = v[2:3] ./ norm(v[2:3]) 
                    b_dir = solver.variables[2:3] ./ norm(solver.variables[2:3])
                    @test norm(v_dir + b_dir, Inf) < 1.0e-3
                    @test norm(solver.variables[2:3]) <= μ * γ 
                end
            end
        end
    end
end