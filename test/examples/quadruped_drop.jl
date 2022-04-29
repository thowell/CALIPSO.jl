@testset "Examples: Quadruped drop" begin
    function quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u) 
        model = RoboDojo.quadruped

        # configurations
        
        q1⁻ = x[1:11] 
        q2⁻ = x[11 .+ (1:11)]
        q2⁺ = y[1:11]
        q3⁺ = y[11 .+ (1:11)]

        # control 
        u_control = u[1:8] 
        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 

        E = [1.0 -1.0] # friction mapping 
        J = RoboDojo.contact_jacobian(model, q2⁺)
        λ = transpose(J[1:8, :]) * [
                                    [E * β[1:2]; γ[1]];
                                    [E * β[3:4]; γ[2]];
                                    [E * β[5:6]; γ[3]];
                                    [E * β[7:8]; γ[4]];
                                ]

        [
            q2⁺ - q2⁻;
            RoboDojo.dynamics(model, mass_matrix, dynamics_bias, 
                timestep, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
        ]
    end

    function quadruped_dyn1(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
            quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
            y[22 .+ (1:4)] - u[8 .+ (1:4)];
        ]
    end

    function quadruped_dynt(mass_matrix, dynamics_bias, timestep, y, x, u)
        [
            quadruped_dyn(mass_matrix, dynamics_bias, timestep, y, x, u);
            y[22 .+ (1:4)] - u[8 .+ (1:4)];
        ]
    end

    function contact_constraints_inequality_1(timestep, x, u) 
        model = RoboDojo.quadruped
        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]
        
        μ = RoboDojo.friction_coefficients(model)[1:4]
        fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

        [
            ϕ;
            fc;
        ]
    end

    function contact_constraints_inequality_t(timestep, x, u) 
        model = RoboDojo.quadruped

        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 
        ψ = u[8 + 4 + 8 .+ (1:4)] 
        η = u[8 + 4 + 8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]

        μ = RoboDojo.friction_coefficients(model)[1:4]
        fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)

        [
            ϕ;
            fc;
        ]
    end

    function contact_constraints_inequality_T(timestep, x, u) 
        model = RoboDojo.quadruped
        q3 = x[11 .+ (1:11)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]

        [
            ϕ;
        ]
    end

    function contact_constraints_equality_1(timestep, x, u) 
        model = RoboDojo.quadruped

        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 
        ψ = u[8 + 4 + 8 .+ (1:4)] 
        η = u[8 + 4 + 8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]

        v = (q3 - q2) ./ timestep[1]
        E = [1.0; -1.0]
        vT = vcat([E * (RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
        ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
        
        μ = RoboDojo.friction_coefficients(model)[1:4]
        fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)
        
        return [
                η - vT - ψ_stack;
                β .* η;
                ψ .* fc;
        ]
    end

    function contact_constraints_equality_t(timestep, x, u) 
        model = RoboDojo.quadruped

        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 
        ψ = u[8 + 4 + 8 .+ (1:4)] 
        η = u[8 + 4 + 8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]
        γ⁻ = x[22 .+ (1:4)] 

        v = (q3 - q2) ./ timestep[1]
        E = [1.0; -1.0]
        vT = vcat([E * (RoboDojo.quadruped_contact_kinematics_jacobians[i](q3) * v)[1] for i = 1:4]...)
        ψ_stack = vcat([ψ[i] * ones(2) for i = 1:4]...)
        
        μ = RoboDojo.friction_coefficients(model)[1:4]
        fc = μ .* γ[1:4] - vcat([sum(β[(i-1) * 2 .+ (1:2)]) for i = 1:4]...)
        
        return [
            η - vT - ψ_stack;
            γ⁻ .* ϕ;
            β .* η; 
            ψ .* fc;
        ]
    end

    function contact_constraints_equality_T(timestep, x, u) 
        model = RoboDojo.quadruped

        q3 = x[11 .+ (1:11)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]
        γ⁻ = x[22 .+ (1:4)] 

        return [
            γ⁻ .* ϕ;
        ]
    end

    function initial_configuration(model::RoboDojo.Quadruped, θ1, θ2, θ3)
        q1 = zeros(11)
        q1[3] = 0.0
        q1[4] = -θ1
        q1[5] = θ2

        q1[8] = -θ1
        q1[9] = θ2

        q1[2] = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])

        q1[10] = -θ3
        q1[11] = acos((q1[2] - model.l_thigh2 * cos(q1[10])) / model.l_calf2)

        q1[6] = -θ3
        q1[7] = acos((q1[2] - model.l_thigh2 * cos(q1[6])) / model.l_calf2)

        return q1
    end

    # ## quadruped 
    nc = 4
    nq = RoboDojo.quadruped.nq
    nx = 2 * nq
    nu = RoboDojo.quadruped.nu + nc + 8 + nc + 8
    nw = RoboDojo.quadruped.nw

    # ## time 
    horizon = 6
    timestep = 0.1

    # ## dimensions 
    num_states = [nx, [nx + nc for t = 2:horizon]...]
    num_actions = [nu for t = 1:horizon-1]

    # ## initial configuration
    θ1 = pi / 4.0
    θ2 = pi / 4.0
    θ3 = pi / 3.0

    q1 = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)
    q1[2] += 0.25#
    q1[3] += 0.2
    qT = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)

    # ## dynamics
    mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
    dynamics = [
            (y, x, u) -> quadruped_dyn1(mass_matrix, dynamics_bias, [timestep], y, x, u), 
            [(y, x, u) -> quadruped_dynt(mass_matrix, dynamics_bias, [timestep], y, x, u) for t = 2:horizon-1]...,
    ]

    # ## objective
    function obj1(x, u)
        u_ctrl = u[1:RoboDojo.quadruped.nu]
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += 1.0 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end

    function objt(x, u)
        u_ctrl = u[1:RoboDojo.quadruped.nu]
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += 1.0 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end

    function objT(x, u)
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += 1.0 * dot(q - qT, q - qT)

        return J
    end

    objective = [
        obj1, 
        [objt for t = 2:horizon-1]..., 
        objT,
    ];

    # control limits
    function equality_1(x, u) 
        [
            x[1:11] - q1;
            x[11 .+ (1:11)] - q1;
            contact_constraints_equality_1(timestep, x, u); 
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
            u[RoboDojo.quadruped.nu .+ (1:(nu - RoboDojo.quadruped.nu))];
        ]
    end

    function inequality_t(x, u) 
        [
            contact_constraints_inequality_t(timestep, x, u);
            u[RoboDojo.quadruped.nu .+ (1:(nu - RoboDojo.quadruped.nu))];
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

    # ## options 
    options = Options(
            verbose=true,        
            constraint_tensor=false,
            update_factorization=false,
            penalty_initial=1.0, 
    )

    # ## solver 
    solver = Solver(objective, dynamics, num_states, num_actions; 
        equality=equality,
        nonnegative=nonnegative,
        options=options,
        );

    # ## initialize
    q_interp = CALIPSO.linear_interpolation(q1, q1, horizon+1)
    x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:horizon]
    action_guess = [max.(0.0, 1.0e-3 * randn(nu)) for t = 1:horizon-1] # may need to run more than once to get good trajectory
    state_guess = [t == 1 ? x_interp[1] : [x_interp[t]; max.(0.0, 1.0e-3 * randn(nc))] for t = 1:horizon]
    initialize_states!(solver, x_guess);
    initialize_controls!(solver, u_guess);

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
# RoboDojo.visualize!(vis, RoboDojo.quadruped, x_sol, Δt=timestep);