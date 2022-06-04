@testset "Examples: Hopper gait (nonlinear cone)" begin
    function hopper_dyn(mass_matrix, dynamics_bias, timestep, y, x, u) 
        model = RoboDojo.hopper

        # configurations
        q1⁻ = x[1:4] 
        q2⁻ = x[4 .+ (1:4)]
        q2⁺ = y[1:4]
        q3⁺ = y[4 .+ (1:4)]

        # control 
        u_control = u[1:2] 
        γ = u[2 .+ (1:4)] 
        β = u[2 + 4 .+ (1:4)] 
        
        J = RoboDojo.contact_jacobian(model, q2⁺)
        λ = transpose(J) * [[β[2]; γ[1]];
                            [β[4]; γ[2]];
                                γ[3:4];
                            ]
        λ[3] += model.body_radius * β[2] # friction on body creates a moment

        [
            q2⁺ - q2⁻;
            RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
            timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
        ]
    end

    function hopper_dyn1(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
            hopper_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
            y[8 .+ (1:4)] - u[2 .+ (1:4)];
            y[8 + 4 .+ (1:8)] - x
        ]
    end

    function hopper_dynt(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
            hopper_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
            y[8 .+ (1:4)] - u[2 .+ (1:4)];
            y[8 + 4 .+ (1:8)] - x[8 + 4 .+ (1:8)]
        ]
    end

    function contact_constraints_inequality_1(timestep, x, u) 
        model = RoboDojo.hopper
        q3 = x[4 .+ (1:4)] 
        ϕ = RoboDojo.signed_distance(model, q3) 
        [
            ϕ; 
        ]
    end

    function contact_constraints_inequality_t(timestep, x, u) 
        model = RoboDojo.hopper
        q3 = x[4 .+ (1:4)] 
        ϕ = RoboDojo.signed_distance(model, q3) 
        [
            ϕ; 
        ]
    end

    function contact_constraints_inequality_T(timestep, x, u) 
        model = RoboDojo.hopper
        q3 = x[4 .+ (1:4)] 
        ϕ = RoboDojo.signed_distance(model, q3) 
        [
            ϕ; 
        ]
    end


    function contact_constraints_equality_1(timestep, x, u) 
        model = RoboDojo.hopper

        q2 = x[1:4] 
        q3 = x[4 .+ (1:4)] 

        u_control = u[1:2] 
        γ = u[2 .+ (1:4)] 
        β = u[2 + 4 .+ (1:4)] 
        η = u[2 + 4 + 4 .+ (1:4)] 
        
        μ = [model.friction_body_world; model.friction_foot_world]
        fc = μ .* γ[1:2] - [β[1]; β[3]]

        v = (q3 - q2) ./ timestep[1]
        vT_body = v[1] + model.body_radius * v[3]
        vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
        vc = [vT_body - η[2]; vT_foot - η[4]]
        
        [
            fc;
            vc;
            CALIPSO.second_order_product(β, η);
        ]
    end

    function contact_constraints_equality_t(timestep, x, u) 
        model = RoboDojo.hopper

        q2 = x[1:4] 
        q3 = x[4 .+ (1:4)] 

        γ = u[2 .+ (1:4)] 
        β = u[2 + 4 .+ (1:4)] 
        η = u[2 + 4 + 4 .+ (1:4)] 

        ϕ = RoboDojo.signed_distance(model, q3) 
        γ⁻ = x[8 .+ (1:4)] 
        γ = u[2 .+ (1:4)] 
        β = u[2 + 4 .+ (1:4)] 
        η = u[2 + 4 + 4 .+ (1:4)] 

        μ = [model.friction_body_world; model.friction_foot_world]
        fc = μ .* γ[1:2] - [β[1]; β[3]]

        v = (q3 - q2) ./ timestep[1]
        vT_body = v[1] + model.body_radius * v[3]
        vT_foot = (RoboDojo.kinematics_foot_jacobian(model, q3) * v)[1]
        vc = [vT_body - η[2]; vT_foot - η[4]]
        
        [
            fc; 
            vc;
            γ⁻ .* ϕ;
            CALIPSO.second_order_product(β, η);
        ]
    end

    function contact_constraints_equality_T(timestep, x, u) 
        model = RoboDojo.hopper

        q3 = x[4 .+ (1:4)] 

        ϕ = RoboDojo.signed_distance(model, q3) 
        γ⁻ = x[8 .+ (1:4)] 

        [
            γ⁻ .* ϕ;
        ]
    end

    # ## horizon 
    horizon = 21 
    timestep = 0.05

    # ## dimensions 
    nx = 2 * RoboDojo.hopper.nq
    nu = RoboDojo.hopper.nu + 4 + 4 + 4

    num_states = [nx, [2 * nx + 4 for t = 1:horizon-1]...]
    num_actions = [nu for t = 1:horizon-1]

    # ## dynamics
    mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.hopper)
    dynamics = [
        (y, x, u) -> hopper_dyn1(mass_matrix, dynamics_bias, [timestep], y, x, u), 
        [(y, x, u) -> hopper_dynt(mass_matrix, dynamics_bias, [timestep], y, x, u) for t = 2:horizon-1]...,
    ];

    # ## states
    q1 = [0.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
    qM = [0.5; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
    qT = [1.0; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.5]
    q_ref = [0.5; 0.5 + RoboDojo.hopper.foot_radius; 0.0; 0.25]

    state_initial = [q1; q1]
    xM = [qM; qM]
    state_goal = [qT; qT]
    x_ref = [q_ref; q_ref]

    # ## gate 
    GAIT = 1 
    GAIT = 2 
    GAIT = 3

    if GAIT == 1 
        r_cost = 1.0e-1 
        q_cost = 1.0e-1
    elseif GAIT == 2 
        r_cost = 1.0
        q_cost = 1.0
    elseif GAIT == 3 
        r_cost = 1.0e-3
        q_cost = 1.0e-1
    end

    # ## objective
    function obj1(x, u)
        J = 0.0 
        J += 0.5 * transpose(x - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x - x_ref) 
        J += 0.5 * transpose(u) * Diagonal([r_cost * ones(RoboDojo.hopper.nu); zeros(nu - RoboDojo.hopper.nu)]) * u
        return J
    end

    function objt(x, u)
        J = 0.0 
        J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal(q_cost * [1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
        J += 0.5 * transpose(u) * Diagonal([r_cost * ones(RoboDojo.hopper.nu); zeros(nu - RoboDojo.hopper.nu)]) * u
        return J
    end

    function objT(x, u)
        J = 0.0 
        J += 0.5 * transpose(x[1:nx] - x_ref) * Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]) * (x[1:nx] - x_ref)
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
            # equality (8)
            RoboDojo.kinematics_foot(RoboDojo.hopper, x[1:RoboDojo.hopper.nq]) - RoboDojo.kinematics_foot(RoboDojo.hopper, state_initial[1:RoboDojo.hopper.nq]);
            RoboDojo.kinematics_foot(RoboDojo.hopper, x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]) - RoboDojo.kinematics_foot(RoboDojo.hopper, state_initial[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)]);
            contact_constraints_equality_1(timestep, x, u); 
            # initial condition 4 
            x[1:4] - q1;
        ]
    end

    function equality_t(x, u) 
        [
            # equality (4)
            contact_constraints_equality_t(timestep, x, u); 
        ]
    end

    function equality_T(x, u) 
        θ = x[8 + 4 .+ (1:8)]
        [
            contact_constraints_equality_T(timestep, x, u); 
            # equality (6)
            x[1:RoboDojo.hopper.nq][collect([2, 3, 4])] - θ[1:RoboDojo.hopper.nq][collect([2, 3, 4])];
            x[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])] - θ[RoboDojo.hopper.nq .+ (1:RoboDojo.hopper.nq)][collect([2, 3, 4])];
        ]
    end

    equality = [
        equality_1, 
        [equality_t for t = 2:horizon-1]..., 
        equality_T,
    ];

    function inequality_1(x, u) 
        [
            # inequality (12)
            contact_constraints_inequality_1(timestep, x, u);
            # + 17 + 2 inequality 
            u[1:6] - [-10.0; -10.0; zeros(4)]; 
            [10.0; 10.0] - u[1:2];
            # + 6 state bounds 
            x[2];
            x[4];
            x[6];
            x[8];
            1.0 - x[4]; 
            1.0 - x[8];
        ]
    end

    function inequality_t(x, u) 
        [
        # equality (4)
            contact_constraints_inequality_t(timestep, x, u);
            # + 17 + 2 inequality 
            u[1:6] - [-10.0; -10.0; zeros(4)]; 
            [10.0; 10.0] - u[1:2];
            # + 6 state bounds 
            x[2];
            x[4];
            x[6];
            x[8];
            1.0 - x[4]; 
            1.0 - x[8];
        ]
    end

    function inequality_T(x, u) 
        x_travel = 0.5
        θ = x[8 + 4 .+ (1:8)]
        [
            (x[1] - θ[1]) - x_travel;
            (x[RoboDojo.hopper.nq + 1] - θ[RoboDojo.hopper.nq + 1]) - x_travel; 
            contact_constraints_inequality_T(timestep, x, u);
            # + 6 state bounds 
            x[2];
            x[4];
            x[6];
            x[8];
            1.0 - x[4]; 
            1.0 - x[8];
        ]
    end

    nonnegative = [
        inequality_1, 
        [inequality_t for t = 2:horizon-1]..., 
        inequality_T,
    ];

    second_order = [
            [(x, u) -> u[6 + (i - 1) * 2 .+ (1:2)] for i = 1:4],
            [[(x, u) -> u[6 + (i - 1) * 2 .+ (1:2)] for i = 1:4] for t = 2:horizon-1]...,
            [empty_constraint],
    ];

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        nonnegative=nonnegative,
        second_order=second_order,
        );

    # ## initialize
    state_guess = [state_initial, [[state_initial; zeros(4); state_initial] for t = 2:horizon]...]
    action_guess = [[0.0; RoboDojo.hopper.gravity * RoboDojo.hopper.mass_body * 0.5 * timestep[1]; 1.0e-1 * ones(nu - 2)] for t = 1:horizon-1] # may need to run more than once to get good trajectory
    initialize_states!(solver, state_guess) 
    initialize_controls!(solver, action_guess)

    # ## solve 
    solve!(solver)

    # ## solution
    x_sol, u_sol = CALIPSO.get_trajectory(solver)

    @test norm((x_sol[1] - x_sol[horizon][1:nx])[[2; 3; 4; 6; 7; 8]], Inf) < 1.0e-3

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

# vis = Visualizer()
# render(vis)
# RoboDojo.visualize!(vis, RoboDojo.hopper, x_sol, Δt=timestep);
