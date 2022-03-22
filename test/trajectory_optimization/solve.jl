@testset "Solve: acrobot" begin 
    # horizon 
    T = 101 

    # acrobot 
    num_state = 4 
    num_action = 1 
    num_parameter = 0 
    parameter_dimensions = [num_parameter for t = 1:T]

    function acrobot(x, u, w)
        # dimensions
        n = 4
        m = 1
        d = 0

        # link 1
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        # link 2
        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        # mass matrix
        function M(x, w)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return [a b; b c]
        end

        # dynamics bias
        function τ(x, w)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x, w)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        # input Jacobian
        function B(x, w)
            [0.0; 1.0]
        end

        # dynamics
        q = view(x, 1:2)
        v = view(x, 3:4)

        qdd = M(q, w) \ (-1.0 * C(x, w) * v
                + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function midpoint_implicit(y, x, u, w)
        h = 0.05 # timestep 
        y - (x + h * acrobot(0.5 * (x + y), u, w))
    end

    dt = Dynamics(midpoint_implicit, num_state, num_state, num_action, num_parameter=num_parameter)
    dyn = [dt for t = 1:T-1] 

    # initial state 
    x1 = [0.0; 0.0; 0.0; 0.0] 

    # goal state
    xT = [0.0; π; 0.0; 0.0] 

    # interpolation 
    x_interpolation = linear_interpolation(x1, xT, T)

    # objective 
    ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
    cT = Cost(oT, num_state, 0, num_parameter=num_parameter)
    obj = [[ct for t = 1:T-1]..., cT]

    # constraints
    bnd1 = Bound(num_state, num_action, state_lower=x1, state_upper=x1)
    bndt = Bound(num_state, num_action)
    bndT = Bound(num_state, 0, state_lower=xT, state_upper=xT)
    bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

    cons = [Constraint() for t = 1:T]

    # problem 
    p = Solver(dyn, obj, cons, bounds)

    # initialize
    initialize_states!(p, x_interpolation)
    initialize_controls!(p, [randn(num_action) for t = 1:T-1])

    # solve
    solve!(p)

    # solution 
    x_sol, u_sol = get_trajectory(p) 

    @test norm(x_sol[1] - x1) < 1.0e-3
    @test norm(x_sol[T] - xT) < 1.0e-3
end

@testset "Solve: user-provided dynamics gradients" begin 
    # ## horizon 
    T = 11 

    # ## double integrator 
    num_state = 2
    num_action = 1 
    num_parameter = 0 

    function double_integrator(d, y, x, u, w)
        A = [1.0 1.0; 0.0 1.0] 
        B = [0.0; 1.0] 
        d .= y - (A * x + B * u[1])
    end

    function double_integrator_grad(dz, y, x, u, w) 
        A = [1.0 1.0; 0.0 1.0] 
        B = [0.0; 1.0] 
        dz .= [-A -B I]
    end

    # ## fast methods
    function double_integrator(y, x, u, w)
        A = [1.0 1.0; 0.0 1.0] 
        B = [0.0; 1.0] 
        y - (A * x + B * u[1])
    end

    function double_integrator_grad(y, x, u, w) 
        A = [1.0 1.0; 0.0 1.0] 
        B = [0.0; 1.0] 
        [-A -B I]
    end

    @variables y[1:num_state] x[1:num_state] u[1:num_action] w[1:num_parameter]

    di = double_integrator(y, x, u, w) 
    diz = double_integrator_grad(y, x, u, w) 
    di_func = eval(Symbolics.build_function(di, y, x, u, w)[2])
    diz_func = eval(Symbolics.build_function(diz, y, x, u, w)[2])

    # ## model
    dt = Dynamics(di_func, diz_func, num_state, num_state, num_action)
    dyn = [dt for t = 1:T-1] 

    # ## initialization
    x1 = [0.0; 0.0] 
    xT = [1.0; 0.0] 

    # ## objective 
    ot = (x, u, w) -> 0.1 * dot(x, x) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x, x)
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter)
    cT = Cost(oT, num_state, 0, num_parameter=num_parameter)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints
    bnd1 = Bound(num_state, num_action, 
        state_lower=x1, 
        state_upper=x1)
    bndt = Bound(num_state, num_action)
    bndT = Bound(num_state, 0, 
        state_lower=xT, 
        state_upper=xT)
    bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

    cons = [Constraint() for t = 1:T]

    # ## problem 
    p = Solver(dyn, obj, cons, bounds)

    # ## initialize
    x_interpolation = linear_interpolation(x1, xT, T)
    u_guess = [1.0 * randn(num_action) for t = 1:T-1]

    initialize_states!(p, x_interpolation)
    initialize_controls!(p, u_guess)

    # ## solve
    solve!(p)

    # ## solution
    x_sol, u_sol = get_trajectory(p)
    @test norm(x_sol[1] - x1) < 1.0e-3
    @test norm(x_sol[T] - xT) < 1.0e-3
end

@testset "Solve: general constraint" begin 
    # ## 
    evaluate_hessian = true 

    # ## horizon 
    T = 11 

    # ## double integrator 
    num_state = 2
    num_action = 1 
    num_parameter = 0 

    function double_integrator(y, x, u, w)
        A = [1.0 1.0; 0.0 1.0] 
        B = [0.0; 1.0] 
        y - (A * x + B * u[1])
    end

    # ## model
    dt = Dynamics(double_integrator, num_state, num_state, num_action, evaluate_hessian=evaluate_hessian)
    dyn = [dt for t = 1:T-1] 
    dyn[1].hessian_cache

    # ## initialization
    x1 = [0.0; 0.0] 
    xT = [1.0; 0.0] 

    # ## objective 
    ot = (x, u, w) -> 0.1 * dot(x, x) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x, x)
    ct = Cost(ot, num_state, num_action, num_parameter=num_parameter, 
        evaluate_hessian=evaluate_hessian)
    cT = Cost(oT, num_state, 0, num_parameter=num_parameter, 
        evaluate_hessian=evaluate_hessian)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints
    bnd1 = Bound(num_state, num_action, 
        state_lower=x1, 
        state_upper=x1)
    bndt = Bound(num_state, num_action)
    bndT = Bound(num_state, 0)#, state_lower=xT, state_upper=xT)
    bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

    cons = [Constraint() for t = 1:T]

    gc = GeneralConstraint((z, w) -> z[(end-1):end] - xT, num_state * T + num_action * (T-1), 0, 
        evaluate_hessian=true)

    # ## problem 
    p = Solver(dyn, obj, cons, bounds, 
        general_constraint=gc,
        evaluate_hessian=true,
        options=Options())

    # ## initialize
    x_interpolation = linear_interpolation(x1, xT, T)
    u_guess = [1.0 * randn(num_action) for t = 1:T-1]

    initialize_states!(p, x_interpolation)
    initialize_controls!(p, u_guess)

    # ## solve
    solve!(p)

    # ## solution
    x_sol, u_sol = get_trajectory(p)
    @test norm(x_sol[1] - x1) < 1.0e-3
    @test norm(x_sol[T] - xT) < 1.0e-3
end