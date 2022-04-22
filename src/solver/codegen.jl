function generate_gradients(func::Function, num_variables::Int, num_parameters::Int, mode::Symbol)
    @variables x[1:num_variables] θ[1:num_parameters]

    if mode == :scalar 
        f = [func(x, θ)]
        
        fx = Symbolics.gradient(f[1], x)
        fθ = Symbolics.gradient(f[1], θ)
        fxx = Symbolics.jacobian(fx, x)
        fxθ = Symbolics.jacobian(fx, θ)

        f_expr = eval(Symbolics.build_function(f, x, θ)[2])
        fx_expr = eval(Symbolics.build_function(fx, x, θ)[2])
        fθ_expr = eval(Symbolics.build_function(fθ, x, θ)[2])
        fxx_expr = eval(Symbolics.build_function(fxx, x, θ)[2])
        fxθ_expr = eval(Symbolics.build_function(fxθ, x, θ)[2])

        return f_expr, fx_expr, fθ_expr, fxx_expr, fxθ_expr
    elseif mode == :vector 
        f = func(x, θ)
        
        fx = Symbolics.jacobian(f, x)
        fθ = Symbolics.jacobian(f, θ)

        @variables y[1:length(f)]
        fᵀy = sum(transpose(f) * y)
        fᵀyx = Symbolics.gradient(fᵀy, x)
        fᵀyxx = Symbolics.jacobian(fᵀyx, x) 
        fᵀyxθ = Symbolics.jacobian(fᵀyx, θ) 

        f_expr = eval(Symbolics.build_function(f, x, θ)[2])
        fx_expr = eval(Symbolics.build_function(fx, x, θ)[2])
        fθ_expr = eval(Symbolics.build_function(fθ, x, θ)[2])
        fᵀy_expr = eval(Symbolics.build_function([fᵀy], x, θ, y)[2])
        fᵀyx_expr = eval(Symbolics.build_function(fᵀyx, x, θ, y)[2])
        fᵀyxx_expr = eval(Symbolics.build_function(fᵀyxx, x, θ, y)[2])
        fᵀyxθ_expr = eval(Symbolics.build_function(fᵀyxθ, x, θ, y)[2])

        return f_expr, fx_expr, fθ_expr, fᵀy_expr, fᵀyx_expr, fᵀyxx_expr, fᵀyxθ_expr
    end
end

# using Convex 
# using ECOS, SCS 
# using MathOptInterface

function generate_random_qp(num_variables, num_equality, num_cone;
    # check_feasible=true,
    # optimizer=ECOS.Optimizer,
    # silent_solver=true
    )

    n = num_variables
    m = num_cone
    p = num_equality

    P = randn(n, n)
    P = P' * P
    q = randn(n)
    G = randn(m, n)
    x = randn(n)
    h = G * x + rand(m)
    A = randn(p, n)
    b = A * x

    objective(z, θ) = transpose(z) * P * z + transpose(q) * z 
    constraint_equality(z, θ) = A * z - b 
    constraint_cone(z, θ) = h - G * z
    
    flag = true

    # if check_feasible
    #     x = Convex.Variable(n)
    #     problem = minimize(0.5 * quadform(x, P) + q' * x,
    #                     [G * x <= h,
    #                     A * x == b])

    #     Convex.solve!(problem, optimizer; 
    #         silent_solver=silent_solver)

    #     # optimal value
    #     @show problem.optval

    #     # solution 
    #     @show x.value

    #     # Check the status of the problem
    #     flag = problem.status == MathOptInterface.OPTIMAL
    # end

    return objective, constraint_equality, constraint_cone, flag
end

