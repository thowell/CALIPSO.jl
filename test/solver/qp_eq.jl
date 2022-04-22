@testset "Solver problem: Quadratic program (w/ equality constraints)" begin
    num_variables = 10
    num_equality = 5
    num_cone = 0
    num_parameters = num_variables + num_variables + num_equality * num_variables + num_equality

    x̂ = max.(0.0, randn(num_variables))
    Q = rand(num_variables, num_variables) 
    P = Diagonal(diag(Q' * Q))
    p = randn(num_variables)
    A = rand(num_equality, num_variables)
    b = A * x̂

    parameters = [diag(P); p; vec(A); b]

    function obj(x, θ) 
        Pθ = Diagonal(θ[1:num_variables])
        pθ = θ[num_variables .+ (1:num_variables)]
        
        J = 0.0
        J += 0.5 * transpose(x) * Pθ * x 
        J += transpose(pθ) * x

        return J 
    end

    function eq(x, θ)
        Aθ = reshape(θ[num_variables + num_variables .+ (1:(num_equality * num_variables))], num_equality, num_variables)
        bθ = θ[num_variables + num_variables + num_equality * num_variables .+ (1:num_equality)]
        return Aθ * x - bθ 
    end

    cone(x, θ) = zeros(0)

    # solver
    x0 = randn(num_variables)
    methods = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone;
        parameters=parameters,
        options=Options(
            residual_tolerance=1.0e-8, 
            equality_tolerance=1.0e-6,
            complementarity_tolerance=1.0e-6,
            differentiate=true))
    initialize!(solver, x0)

    # solve 
    solve!(solver)
    
    # solution
    # test solution
    @test norm(solver.data.residual, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual[solver.indices.equality_dual], Inf),
                    norm(solver.data.residual[solver.indices.cone_dual], Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
    
    @test norm(A * solver.solution.variables - b, Inf) < solver.options.equality_tolerance

    # sensitivity
    @variables x[1:num_variables] y[1:num_equality] θ[1:num_parameters]
    function f1(x, θ) 
        Pθ = Diagonal(θ[1:num_variables])
        pθ = θ[num_variables .+ (1:num_variables)]

        return Pθ * x + pθ
    end

    function f2(x, θ) 
        eq(x, θ)
    end

    function f3(y, θ) 
        Aθ = reshape(θ[num_variables + num_variables .+ (1:(num_equality * num_variables))], num_equality, num_variables)
        return transpose(Aθ) * y
    end

    f1θ = Symbolics.jacobian(Symbolics.gradient(obj(x, θ), x), θ)
    f2θ = Symbolics.jacobian(f2(x, θ), θ)
    f3θ = Symbolics.jacobian(f3(y, θ), θ)
    f1θ_func = eval(Symbolics.build_function(f1θ, x, θ)[2])
    f2θ_func = eval(Symbolics.build_function(f2θ, x, θ)[2])
    f3θ_func = eval(Symbolics.build_function(f3θ, y, θ)[2])

    Pxpθ = zeros(num_variables, num_parameters) 
    Aᵀyθ = zeros(num_variables, num_parameters) 
    Axbθ = zeros(num_equality, num_parameters)

    f1θ_func(Pxpθ, solver.solution.variables, parameters)
    f2θ_func(Axbθ, solver.solution.variables, parameters)
    f3θ_func(Aᵀyθ, solver.solution.equality_dual, parameters)

    @test norm(solver.problem.objective_jacobian_variables_parameters[:, 1:num_variables] - Pxpθ[:, 1:num_variables], Inf) < 1.0e-4
    @test norm(solver.problem.equality_dual_jacobian_variables_parameters - Aᵀyθ, Inf) < 1.0e-4
    @test norm(solver.problem.equality_jacobian_parameters - Axbθ, Inf) < 1.0e-4

    rz = [
            P A'; 
            A zeros(num_equality, num_equality);
    ]

    rθ = [
        Pxpθ + Aᵀyθ; 
        Axbθ; 
    ]

    sensitivity = -1.0 * rz \ rθ
    sensitivity_full = -1.0 * solver.data.jacobian_variables \ solver.data.jacobian_parameters
    sensitivity_solver = solver.data.solution_sensitivity

    @test norm(sensitivity[1:num_variables, :] - sensitivity_full[1:num_variables, :], Inf) < 1.0e-2
    @test norm(sensitivity[1:num_variables, :] - sensitivity_solver[1:num_variables, :], Inf) < 1.0e-2
    @test norm(sensitivity_full[1:num_variables, :] - sensitivity_solver[1:num_variables, :], Inf) < 1.0e-2
end