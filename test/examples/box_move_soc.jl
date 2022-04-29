@testset "Examples: Box move (nonlinear friction cone)" begin 
    function box_dyn(mass_matrix, dynamics_bias, timestep, y, x, u)
        model = RoboDojo.box
        
        # configurations
        q1⁻ = x[1:3]
        q2⁻ = x[3 .+ (1:3)]
        q2⁺ = y[1:3]
        q3⁺ = y[3 .+ (1:3)]

        # control
        u_control = u[1:3]
        γ = u[3 .+ (1:4)]
        β = u[3 + 4 .+ (1:8)]
        J = RoboDojo.contact_jacobian(model, q2⁺)
        λ = transpose(J) * [[β[2]; γ[1]];
                            [β[4]; γ[2]];
                            [β[6]; γ[3]];
                            [β[8]; γ[4]]]
        [
        q2⁺ - q2⁻;
        RoboDojo.dynamics(model, mass_matrix, dynamics_bias,
            timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
        ]
    end

    function box_dyn1(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
        box_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
        y[6 .+ (1:4)] - u[3 .+ (1:4)];
        y[6 + 4 .+ (1:6)] - x;
        ]
    end

    function box_dynt(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
        box_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
        y[6 .+ (1:4)] - u[3 .+ (1:4)];
        y[6 + 4 .+ (1:6)] - x[6 + 4 .+ (1:6)];
        ]
    end

    function contact_constraints_inequality_1(timestep, x, u)
        model = RoboDojo.box
        q3 = x[3 .+ (1:3)]
        ϕ = RoboDojo.signed_distance(model, q3)
        [
            ϕ;
        ]
    end

    function contact_constraints_inequality_t(timestep, x, u)
        model = RoboDojo.box
        q3 = x[3 .+ (1:3)]
        ϕ = RoboDojo.signed_distance(model, q3)
        
        [
            ϕ;
        ]
    end

    function contact_constraints_inequality_T(timestep, x, u)
        model = RoboDojo.box
        q3 = x[3 .+ (1:3)]
        ϕ = RoboDojo.signed_distance(model, q3)
        [
            ϕ;
        ]
    end

    function contact_constraints_equality_1(timestep, x, u)
        model = RoboDojo.box
        q2 = x[1:3]
        q3 = x[3 .+ (1:3)]

        γ = u[3 .+ (1:4)]
        β = u[3 + 4 .+ (1:8)]
        η = u[3 + 4 + 8 .+ (1:8)]
        
        v = (q3 - q2) ./ timestep[1]
        vc = vcat([(RoboDojo.box_contact_kinematics_jacobians[i](q3) * v)[1] - η[(i - 1) * 2 + 2] for i = 1:4]...)

        μ = model.friction_body_world    
        fc = μ .* γ[1:4] - [β[1]; β[3]; β[5]; β[7]]

        [
            fc;
            vc;
            CALIPSO.second_order_product(β, η);
        ]
    end

    function contact_constraints_equality_t(timestep, x, u)
        model = RoboDojo.box
        q2 = x[1:3]
        q3 = x[3 .+ (1:3)]
        γ = u[3 .+ (1:4)]
        β = u[3 + 4 .+ (1:8)]
        η = u[3 + 4 + 8 .+ (1:8)]
        ϕ = RoboDojo.signed_distance(model, q3)
        γ⁻ = x[nx .+ (1:4)]

        γ = u[3 .+ (1:4)]
        β = u[3 + 4 .+ (1:8)]
        η = u[3 + 4 + 8 .+ (1:8)]
        
        v = (q3 - q2) ./ timestep[1]
        vc = vcat([(RoboDojo.box_contact_kinematics_jacobians[i](q3) * v)[1] - η[(i - 1) * 2 + 2] for i = 1:4]...)

        μ = model.friction_body_world    
        fc = μ .* γ[1:4] - [β[1]; β[3]; β[5]; β[7]]

        [
            fc;
            vc;
            CALIPSO.second_order_product(β, η);
            γ⁻ .* ϕ; 
        ]
    end

    function contact_constraints_equality_T(timestep, x, u)
        model = RoboDojo.box
        q2 = x[1:3]
        q3 = x[3 .+ (1:3)]
        ϕ = RoboDojo.signed_distance(model, q3)
        γ⁻ = x[6 .+ (1:4)]
        [
            γ⁻ .* ϕ;
        ]
    end

    # ## horizon
    horizon = 11
    timestep = 0.1

    # ## dimensions
    nx = 2 * RoboDojo.box.nq
    nu = RoboDojo.box.nu + 4 + 8 + 8

    num_states = [nx, [2 * nx + 4 for t = 2:horizon]...]
    num_actions = [nu for t = 1:horizon-1]

    # ## dynamics
    mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.box)

    dynamics = [
            (y, x, u) -> box_dyn1(mass_matrix, dynamics_bias, [timestep], y, x, u), 
            [(y, x, u) -> box_dynt(mass_matrix, dynamics_bias, [timestep], y, x, u) for t = 2:horizon-1]...,
    ];

    # ## states
    q1 = [0.0; 0.5; 0.0]
    qM = [0.0; 0.5; 0.0]
    # qT = [0.0; 0.5 + 1.0; 0.0]
    qT = [1.0; 0.5; 0.0]

    q_ref = qT
    state_initial = [q1; q1]
    xM = [qM; qM]
    state_goal = [qT; qT]
    x_ref = [q_ref; q_ref]

    # ## objective
    function obj1(x, u)
        J = 0.0
        J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - x_ref)
        J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(RoboDojo.box.nu); 1.0e-5 * ones(nu - RoboDojo.box.nu)]) * u
        return J
    end

    function objt(x, u)
        J = 0.0
        J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
        J += 0.5 * transpose(u) * Diagonal([1.0e-2 * ones(RoboDojo.box.nu); 1.0e-5 * ones(nu - RoboDojo.box.nu)]) * u
        return J
    end

    function objT(x, u)
        J = 0.0
        J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
        return J
    end

    objective = [
            obj1, 
            [objt for t = 2:horizon-1]..., 
            objT,
    ];

    # ## constraints
    function equality_1(x, u)
        [
            contact_constraints_equality_1(timestep, x, u);
            x - state_initial;
        ]
    end

    function equality_t(x, u)
        [
            contact_constraints_equality_t(timestep, x, u);
        ]
    end

    function equality_T(x, u)
        [
            contact_constraints_equality_T(timestep, x, u);
            x[1:6] - state_goal;
        ]
    end

    equality = [
            equality_1, 
            [equality_t for t = 2:horizon-1]..., 
            equality_T,
    ];

    function inequality_1(x, u)
        [
            contact_constraints_inequality_1(timestep, x, u);
            u[1:7] - [-10.0; -10.0; -10.0; zeros(4)];
            [10.0; 10.0; 10.0] - u[1:3];
        ]
    end

    function inequality_t(x, u)
        [
            contact_constraints_inequality_t(timestep, x, u);
            u[1:7] - [-10.0; -10.0; -10.0; zeros(4)];
            [10.0; 10.0; 10.0] - u[1:3];
        ]
    end

    function inequality_T(x, u)
        [
            contact_constraints_inequality_T(timestep, x, u);
        ]
    end

    nonnegative = [
            inequality_1, 
            [inequality_t for t = 2:horizon-1]..., 
            inequality_T,
    ];

    second_order = [
        [(x, u) -> u[7 + (i - 1) * 2 .+ (1:2)] for i = 1:8],
        [[(x, u) -> u[7 + (i - 1) * 2 .+ (1:2)] for i = 1:8] for t = 2:horizon-1]...,
        [empty_constraint],
    ]

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        nonnegative=nonnegative,
        second_order=second_order);


    # ## initialize
    state_interpolation = linear_interpolation(state_initial, state_goal, horizon)
    state_guess = [state_interpolation[1], [[state_interpolation[t]; zeros(4); state_interpolation[t]] for t = 2:horizon]...]
    action_guess = [[0.0 * randn(RoboDojo.box.nu); 1.0e-1 * ones(nu - RoboDojo.box.nu)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
    initialize_states!(solver, state_guess) 
    initialize_controls!(solver, action_guess)

    # ## solve
    solve!(solver)

    # ## solution
    x_sol, u_sol = CALIPSO.get_trajectory(solver)

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

# # ## visualize
# vis = Visualizer()
# render(vis)
# RoboDojo.visualize!(vis, RoboDojo.box, x_sol, Δt=timestep, r=0.5);

