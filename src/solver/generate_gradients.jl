# using Convex 
# using ECOS, SCS 
# using MathOptInterface

function generate_gradients(func::Function, num_variables::Int, num_parameters::Int, mode::Symbol)

    @variables x[1:num_variables] θ[1:num_parameters]

    if mode == :scalar 
        f = func(x, θ)
        
        fx = Symbolics.gradient(f, x) 
        fθ = Symbolics.gradient(f, θ)
        fxx = Symbolics.jacobian(fx, x)
        fxθ = Symbolics.jacobian(fx, θ)

        f_func = eval(Symbolics.build_function(f, x, θ))
        fx_func = eval(Symbolics.build_function(fx, x, θ)[2])
        fθ_func = eval(Symbolics.build_function(fθ, x, θ)[2])
        fxx_func = eval(Symbolics.build_function(fxx, x, θ)[2])
        fxθ_func = eval(Symbolics.build_function(fxθ, x, θ)[2])

        return f_func, fx_func, fθ_func, fxx_func, fxθ_func
    elseif mode == :vector 
        f = func(x, θ)
        
        fx = Symbolics.jacobian(f, x)
        fθ = Symbolics.jacobian(f, θ)

        @variables y[1:length(f)]
        fᵀy = sum(transpose(f) * y)
        fᵀyx = Symbolics.gradient(fᵀy, x)
        fᵀyxx = Symbolics.jacobian(fᵀyx, x) 
        fᵀyxθ = Symbolics.jacobian(fᵀyx, θ) 

        f_func = eval(Symbolics.build_function(f, x, θ)[2])
        fx_func = eval(Symbolics.build_function(fx, x, θ)[2])
        fθ_func = eval(Symbolics.build_function(fθ, x, θ)[2])
        fᵀy_func = eval(Symbolics.build_function(fᵀy, x, θ, y))
        fᵀyx_func = eval(Symbolics.build_function(fᵀyx, x, θ, y)[2])
        fᵀyxx_func = eval(Symbolics.build_function(fᵀyxx, x, θ, y)[2])
        fᵀyxθ_func = eval(Symbolics.build_function(fᵀyxθ, x, θ, y)[2])

        return f_func, fx_func, fθ_func, fᵀy_func, fᵀyx_func, fᵀyxx_func, fᵀyxθ_func
    end
end

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

