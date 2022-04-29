@testset "Solver: Second-order cone for friction" begin 
    """
        minimize    v' b
        subject to  ||b|| <= μn

        minimize    [0 v]' x 
        subject to  x[1] = μn
                    x in K
    """

    # ## variables
    num_variables = 3

    # ## cone indices
    nonnegative_indices = collect(1:0)
    second_order_indices = [collect(1:3)]

    # ## problem settings
    V = [[0.0; 0.0; 0.0], [0.0; 1.0; 0.0], [0.0; 0.0; 1.0], [0.0; 1.0; 1.0], [0.0; 10.0; 1.0]]
    friction_coefficients = [0.0, 0.5, 1.0] 
    impact_impulse = [0.0, 1.0] 

    for v in V 
        for μ in friction_coefficients 
            for γ in impact_impulse

                # ## problem
                objective(x) = transpose(v) * x
                equality(x) = [x[1] - μ * γ]
                cone(x) = x

                # solver
                solver = Solver(objective, equality, cone, num_variables;
                    nonnegative_indices=nonnegative_indices,
                    second_order_indices=second_order_indices,)

                # ## initialize
                x0 = randn(num_variables)
                initialize!(solver, x0)

                # ## solve 
                solve!(solver)

                # ## solution
                @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

                slack_norm = max(
                                norm(solver.data.residual.equality_dual, Inf),
                                norm(solver.data.residual.cone_dual, Inf),
                )
                @test slack_norm < solver.options.slack_tolerance

                @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
                @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 

                @test !CALIPSO.cone_violation(solver.solution.all, zero(solver.solution.all), 0.0,
                    solver.indices.cone_nonnegative, solver.indices.cone_second_order)
                if norm(v[2:3]) > 0.0 && γ > 0.0 && μ > 0.0
                    v_dir = v[2:3] ./ norm(v[2:3]) 
                    b_dir = solver.solution.variables[2:3] ./ norm(solver.solution.variables[2:3])
                    @test norm(v_dir + b_dir, Inf) < 1.0e-3
                    @test norm(solver.solution.variables[2:3]) <= μ * γ 
                end
            end
        end
    end
end