@testset "Examples: Quadruped drop" begin
    function quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w) 
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
                h, q1⁻, q2⁺, u_control, zeros(model.nw), λ, q3⁺)
        ]
    end

    function quadruped_dyn1(mass_matrix, dynamics_bias, h, y, x, u, w)
        [
            quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
            y[22 .+ (1:4)] - u[8 .+ (1:4)];
        ]
    end

    function quadruped_dynt(mass_matrix, dynamics_bias, h, y, x, u, w)
        [
            quadruped_dyn(mass_matrix, dynamics_bias, h, y, x, u, w);
            y[22 .+ (1:4)] - u[8 .+ (1:4)];
        ]
    end

    function contact_constraints_inequality_1(h, x, u, w) 
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

    function contact_constraints_inequality_t(h, x, u, w) 
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

    function contact_constraints_inequality_T(h, x, u, w) 
        model = RoboDojo.quadruped
        q3 = x[11 .+ (1:11)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]

        [
            ϕ;
        ]
    end

    function contact_constraints_equality_1(h, x, u, w) 
        model = RoboDojo.quadruped

        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 
        ψ = u[8 + 4 + 8 .+ (1:4)] 
        η = u[8 + 4 + 8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]

        v = (q3 - q2) ./ h[1]
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

    function contact_constraints_equality_t(h, x, u, w) 
        model = RoboDojo.quadruped

        q2 = x[1:11] 
        q3 = x[11 .+ (1:11)] 

        γ = u[8 .+ (1:4)] 
        β = u[8 + 4 .+ (1:8)] 
        ψ = u[8 + 4 + 8 .+ (1:4)] 
        η = u[8 + 4 + 8 + 4 .+ (1:8)] 

        ϕ = RoboDojo.signed_distance(model, q3)[1:4]
        γ⁻ = x[22 .+ (1:4)] 

        v = (q3 - q2) ./ h[1]
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

    function contact_constraints_equality_T(h, x, u, w) 
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

    # RoboDojo.quadruped.friction_joint .*= 0.001

    # ## time 
    T = 6
    h = 0.1

    # ## initial configuration
    θ1 = pi / 4.0
    θ2 = pi / 4.0
    θ3 = pi / 3.0

    q1 = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)
    q1[2] += 0.25#
    q1[3] += 0.2
    qT = initial_configuration(RoboDojo.quadruped, θ1, θ2, θ3)

    # ## model
    # println("codegen dynamics")
    mass_matrix, dynamics_bias = RoboDojo.codegen_dynamics(RoboDojo.quadruped)
    d1 = CALIPSO.Dynamics((y, x, u, w) -> quadruped_dyn1(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc, nx, nu)
    dt = CALIPSO.Dynamics((y, x, u, w) -> quadruped_dynt(mass_matrix, dynamics_bias, [h], y, x, u, w), nx + nc, nx + nc, nu)
    dyn = [d1, [dt for t = 2:T-1]...]
    # println("codegen dynamics: complete!")

    # ## objective
    obj = CALIPSO.Cost{Float64}[]

    function obj1(x, u, w)
        u_ctrl = u[1:RoboDojo.quadruped.nu]
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += 1.0 * dot(u_ctrl, u_ctrl)
        J += 1.0 * dot(q - qT, q - qT)
        return J
    end
    push!(obj, CALIPSO.Cost(obj1, nx, nu))

    for t = 2:T-1
        function objt(x, u, w)
            u_ctrl = u[1:RoboDojo.quadruped.nu]
            q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

            J = 0.0 
            J += 1.0 * dot(u_ctrl, u_ctrl)
            J += 1.0 * dot(q - qT, q - qT)
            return J
        end
        push!(obj, CALIPSO.Cost(objt, nx + nc, nu))
    end

    function objT(x, u, w)
        q = x[RoboDojo.quadruped.nq .+ (1:RoboDojo.quadruped.nq)]

        J = 0.0 
        J += 1.0 * dot(q - qT, q - qT)

        return J
    end
    push!(obj, CALIPSO.Cost(objT, nx + nc, 0))

    # control limits
    eq = CALIPSO.Constraint{Float64}[]
    function equality_1(x, u, w) 
        [
            x[1:11] - q1;
            x[11 .+ (1:11)] - q1;
            contact_constraints_equality_1(h, x, u, w); 
        ]
    end
    push!(eq, CALIPSO.Constraint(equality_1, nx, nu))

    for t = 2:T-1
        function equality_t(x, u, w) 
            [
                contact_constraints_equality_t(h, x, u, w); 
            ]
        end
        push!(eq, CALIPSO.Constraint(equality_t, nx + nc, nu))
    end

    function equality_T(x, u, w) 
        [
        contact_constraints_equality_T(h, x, u, w); 
        ]
    end
    push!(eq, CALIPSO.Constraint(equality_T, nx + nc, 0))

    ineq = CALIPSO.Constraint{Float64}[]
    function inequality_1(x, u, w) 
        [
            contact_constraints_inequality_1(h, x, u, w);
            u[RoboDojo.quadruped.nu .+ (1:(nu - RoboDojo.quadruped.nu))];
        ]
    end
    push!(ineq, CALIPSO.Constraint(inequality_1, nx, nu))

    for t = 2:T-1
        function inequality_t(x, u, w) 
            [
                contact_constraints_inequality_t(h, x, u, w);
                u[RoboDojo.quadruped.nu .+ (1:(nu - RoboDojo.quadruped.nu))];
            ]
        end
        push!(ineq, CALIPSO.Constraint(inequality_t, nx + nc, nu))
    end

    function inequality_T(x, u, w) 
        [
            contact_constraints_inequality_T(h, x, u, w);
        ]
    end
    push!(ineq, CALIPSO.Constraint(inequality_T, nx + nc, 0))

    so = [[Constraint()] for t = 1:T]

    # ## initialize
    q_interp = CALIPSO.linear_interpolation(q1, q1, T+1)
    x_interp = [[q_interp[t]; q_interp[t+1]] for t = 1:T]
    u_guess = [max.(0.0, 1.0e-3 * randn(nu)) for t = 1:T-1] # may need to run more than once to get good trajectory
    x_guess = [t == 1 ? x_interp[1] : [x_interp[t]; max.(0.0, 1.0e-3 * randn(nc))] for t = 1:T]

    # ## problem
    # println("solver")
    trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so);
    methods = ProblemMethods(trajopt);
    idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

    # ## solver
    solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
        nonnegative_indices=idx_nn, 
        second_order_indices=idx_soc,
        options=Options(
            verbose=true,        
            constraint_hessian=false,
            # residual_tolerance=1.0e-3, 
            # equality_tolerance=1.0e-2,
            # complementarity_tolerance=1.0e-2,
            update_factorization=false,
            penalty_initial=1.0, 
    ));

    initialize_states!(solver, trajopt, x_guess);
    initialize_controls!(solver, trajopt, u_guess);

    # println("solver instantiated and initialized")

    # solve 
    solve!(solver)

    # ## solution
    x_sol, u_sol = CALIPSO.get_trajectory(solver, trajopt)

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
# RoboDojo.visualize!(vis, RoboDojo.quadruped, x_sol, Δt=h);