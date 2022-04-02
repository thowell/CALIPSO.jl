# using Convex 
# using ECOS, SCS 
# using MathOptInterface

function generate_gradients(func::Function, num_variables::Int, mode::Symbol; 
    output=:inplace)

    @variables x[1:num_variables]

    if mode == :scalar 
        f = func(x)
        fx = Symbolics.gradient(f, x) 
        fxx = Symbolics.jacobian(fx, x) 

        f_func = eval(Symbolics.build_function(f, x))
        fx_func = eval(Symbolics.build_function(fx, x)[output == :inplace ? 2 : 1])
        fxx_func = eval(Symbolics.build_function(fxx, x)[output == :inplace ? 2 : 1])

        return f_func, fx_func, fxx_func
    elseif mode == :vector 
        f = func(x)
        fx = Symbolics.jacobian(f, x)
        dim = length(f)
        @variables y[1:dim]
        fyxx = Symbolics.hessian(dot(f, y), x) 

        f_func = eval(Symbolics.build_function(f, x)[output == :inplace ? 2 : 1])
        fx_func = eval(Symbolics.build_function(fx, x)[output == :inplace ? 2 : 1])
        fyxx_func = eval(Symbolics.build_function(fyxx, x, y)[output == :inplace ? 2 : 1])

        return f_func, fx_func, fyxx_func
    end
end

using Symbolics

function generate_random_qp(num_variables, num_equality, num_inequality;
    # check_feasible=true,
    # optimizer=ECOS.Optimizer,
    # silent_solver=true
    )

    n = num_variables
    m = num_inequality
    p = num_equality

    P = randn(n, n)
    P = P' * P
    q = randn(n)
    G = randn(m, n)
    x = randn(n)
    h = G * x + rand(m)
    A = randn(p, n)
    b = A * x

    objective(z) = transpose(z) * P * z + transpose(q) * z 
    constraint_equality(z) = A * z - b 
    constraint_inequality(z) = h - G * z
    
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

    return objective, constraint_equality, constraint_inequality, flag
end

