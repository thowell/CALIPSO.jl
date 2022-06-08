@testset "Examples: Acrobot" begin 
    # ## horizon
    horizon = 51 

    # ## dimensions 
    num_states = [4 for t = 1:horizon] 
    num_actions = [1 for t = 1:horizon-1] 

    # ## dynamics
    function acrobot_continuous(x, u)
        mass1 = 1.0  
        inertia1 = 0.33  
        length1 = 1.0 
        lengthcom1 = 0.5 

        mass2 = 1.0  
        inertia2 = 0.33  
        length2 = 1.0 
        lengthcom2 = 0.5 

        gravity = 9.81 
        friction1 = 0.1 
        friction2 = 0.1

        function M(x)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return [a b; b c]
        end

        function Minv(x)
            a = (inertia1 + inertia2 + mass2 * length1 * length1
                + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

            b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

            c = inertia2

            return 1.0 / (a * c - b * b) * [c -b; -b a]
        end

        function τ(x)
            a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
                - mass2 * gravity * (length1 * sin(x[1])
                + lengthcom2 * sin(x[1] + x[2])))

            b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

            return [a; b]
        end

        function C(x)
            a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
            c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
            d = 0.0

            return [a b; c d]
        end

        function B(x)
            [0.0; 1.0]
        end

        q = x[1:2]
        v = x[3:4]

        qdd = Minv(q) * (-1.0 * C(x) * v
                + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

        return [x[3]; x[4]; qdd[1]; qdd[2]]
    end

    function acrobot_discrete(x, u)
        h = 0.05 # timestep 
        x + h * acrobot_continuous(x + 0.5 * h * acrobot_continuous(x, u), u)
    end

    function acrobot_discrete(y, x, u)
        y - acrobot_discrete(x, u)
    end

    dynamics = [acrobot_discrete for t = 1:horizon-1]
   
    # ## states
    state_initial = [0.0; 0.0; 0.0; 0.0] 
    state_goal = [π; 0.0; 0.0; 0.0] 

    # ## objective 
    objective = [
        [(x, u) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
        (x, u) -> 0.1 * dot(x[3:4], x[3:4]),
    ];

    # ## constraints 
    equality = [
        (x, u) -> x - state_initial, 
        [empty_constraint for t = 2:horizon-1]..., 
        (x, u) -> x - state_goal,
    ];

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality);

    # ## initialize
    state_guess = linear_interpolation(state_initial, state_goal, horizon)
    action_guess = [0.01 * ones(num_actions[t]) for t = 1:horizon-1]
    initialize_states!(solver, state_guess) 
    initialize_actions!(solver, action_guess)

    # ## solve 
    solve!(solver)

    # ## test solution
    state_solution, action_solution = get_trajectory(solver);
    state_solution[end]

    @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

    slack_norm = max(
                    norm(solver.data.residual.equality_dual, Inf),
                    norm(solver.data.residual.cone_dual, Inf),
    )
    @test slack_norm < solver.options.slack_tolerance

    @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
    @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
end
