# @testset "Examples: Pendulum" begin 
    # ## horizon 
    T = 11 

    # ## acrobot 
    num_state = 2
    num_action = 1 
    num_parameter = 0 

    function pendulum(x, u, w)
        mass = 1.0
        length_com = 0.5
        gravity = 9.81
        damping = 0.1

        [
            x[2],
            (u[1] / ((mass * length_com * length_com))
                - gravity * sin(x[1]) / length_com
                - damping * x[2] / (mass * length_com * length_com))
        ]
    end

    function midpoint_implicit(y, x, u, w)
        h = 0.05 # timestep 
        y - (x + h * pendulum(0.5 * (x + y), u, w))
    end

    # ## model
    dt = Dynamics(
        midpoint_implicit, 
        num_state, 
        num_state, 
        num_action, 
        num_parameter=num_parameter)
    dyn = [dt for t = 1:T-1] 

    # ## initialization
    x1 = [0.0; 0.0] 
    xT = [π; 0.0] 

    # ## objective 
    ot = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u)
    oT = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2])
    ct = Cost(ot, num_state, num_action)
    cT = Cost(oT, num_state, 0)
    obj = [[ct for t = 1:T-1]..., cT]

    # ## constraints 
    eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
    eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
    eq = [eq1, [Constraint() for t = 2:T-1]..., eqT]

    ineq = [Constraint() for t = 1:T]

    soc = [[Constraint()] for t = 1:T]

    # ## problem 
    trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc)

    # ## initialize
    x_interpolation = linear_interpolation(x1, xT, T)
    u_guess = [1.0 * randn(num_action) for t = 1:T-1]

    methods = ProblemMethods(trajopt)
    idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

    # ## solver
    solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
        nonnegative_indices=idx_nn, 
        second_order_indices=idx_soc,
        custom=trajopt,
        options=Options(verbose=true))
    initialize_states!(solver, trajopt, x_interpolation) 
    initialize_controls!(solver, trajopt, u_guess)

    # ## solve 
    solve!(solver)

    # test solution
    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end
