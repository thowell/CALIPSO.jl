@testset "Solver: Second-order cone for friction" begin 
    """
        minimize    v' b
        subject to  ||b|| <= μn

        minimize    [0 v]' x 
        subject to  x[1] = μn
                    x in K
    """

    num_variables = 3
    num_parameters = 0 
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
                obj(x, θ) = transpose(v) * x
                eq(x, θ) = [x[1] - μ * γ]
                cone(x, θ) = x

                # solver
                methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
                solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone;
                    nonnegative_indices=idx_ineq,
                    second_order_indices=idx_soc,)

                x0 = randn(num_variables)
                initialize!(solver, x0)

                # solve 
                solve!(solver)

                # test solution
                @test norm(solver.data.residual, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

                slack_norm = max(
                                norm(solver.data.residual[solver.indices.equality_dual], Inf),
                                norm(solver.data.residual[solver.indices.cone_dual], Inf),
                )
                @test slack_norm < solver.options.slack_tolerance

                @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
                @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

                @test !CALIPSO.cone_violation(solver.variables, zero(solver.variables), 0.0,
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