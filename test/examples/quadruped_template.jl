function trotting_configuration(model::RoboDojo.Quadruped4, Tm::Int, mode::Symbol; timestep=0.01,
    velocity=0.15, body_height=0.30, body_forward_position=0.00,
    foot_height=0.08)

    nq = model.nq
    nx = 2nq
    foot_x = model.l_torso1
    foot_y = 0.0
    stride = velocity * timestep * (Tm - 1)


    orientations = [zeros(1) for i=1:Tm]
    body_positions = [[body_forward_position; body_height] for i=1:Tm]

    flight_positions = [zeros(3) for i=1:Tm]
    for (i,θ) in enumerate(range(0, stop=π, length=Tm))
        x = -stride/2 * cos(θ)
        z = foot_height * sin(θ)
        flight_positions[i] = [x; 0; z]
    end
    stance_positions = [[s;0;0] for s in range(stride/2, stop=-stride/2, length=Tm)]

    foot_1_positions = [[+foot_x; +foot_y; 0.0] for i=1:Tm]
    foot_2_positions = [[+foot_x; -foot_y; 0.0] for i=1:Tm]
    foot_3_positions = [[-foot_x; +foot_y; 0.0] for i=1:Tm]
    foot_4_positions = [[-foot_x; -foot_y; 0.0] for i=1:Tm]

    if mode == :left
        foot_1_positions .+= flight_positions
        foot_2_positions .+= stance_positions
        foot_3_positions .+= stance_positions
        foot_4_positions .+= flight_positions
    else
        foot_1_positions .+= stance_positions
        foot_2_positions .+= flight_positions
        foot_3_positions .+= flight_positions
        foot_4_positions .+= stance_positions
    end

    foot_1_joint_angles = [
        inverse_kinematics(model, [body_positions[i]; orientations[i]], foot_1_positions[i][[1,3]], leg=:leg1)
        for i=1:Tm]
    foot_2_joint_angles = [
        inverse_kinematics(model, [body_positions[i]; orientations[i]], foot_2_positions[i][[1,3]], leg=:leg2)
        for i=1:Tm]
    foot_3_joint_angles = [
        inverse_kinematics(model, [body_positions[i]; orientations[i]], foot_3_positions[i][[1,3]], leg=:leg3)
        for i=1:Tm]
    foot_4_joint_angles = [
        inverse_kinematics(model, [body_positions[i]; orientations[i]], foot_4_positions[i][[1,3]], leg=:leg4)
        for i=1:Tm]

    configurations = [[
        body_positions[i];
        orientations[i];
        foot_1_joint_angles[i];
        foot_2_joint_angles[i];
        foot_3_joint_angles[i];
        foot_4_joint_angles[i];
        ] for i=1:Tm]
    return configurations
end

function inverse_kinematics(model::RoboDojo.Quadruped4, body_pose::Vector{T},
        goal_foot_position::Vector{T}; leg::Symbol=:none, tolerance=1e-8, max_iterations=40) where T
    # body_pose = [x_body, y_body, θ]
    # foot_position = [x_foot, y_foot]

    function res(θ)
        q = [body_pose; zeros(8)]
        if leg == :leg1
            q[4:5] .= θ
        elseif leg == :leg2
            q[6:7] .= θ
        elseif leg == :leg3
            q[8:9] .= θ
        elseif leg == :leg4
            q[10:11] .= θ
        else
            @warn "invalid leg index"
        end
        foot_position = RoboDojo.kinematics_calf(model, q; leg=leg, mode=:ee)
        return goal_foot_position - foot_position
    end

    θ = [-0.8, 0.8]
    for i = 1:max_iterations
        r = res(θ)
        (norm(r, Inf) < tolerance) && break
        J = FiniteDiff.finite_difference_jacobian(θ -> res(θ), θ)
        θ = θ - 0.90 * J \ r
        (i == max_iterations) && @warn "failed inverse kinematics"
        # @show norm(r, Inf)
    end
    return θ
end

function trotting_gait(model::RoboDojo.Quadruped4, Tm::Int; timestep=0.01, velocity=0.15, kwargs...)

    T = 2Tm - 1
    nq = model.nq

    configurations_left = trotting_configuration(model, Tm, :left; timestep=timestep, velocity=velocity, kwargs...)
    configurations_right = trotting_configuration(model, Tm, :right; timestep=timestep, velocity=velocity, kwargs...)

    configurations = [configurations_left[1:end-1]; configurations_right]
    # add forward velocity
    for (i,q) in enumerate(configurations)
        q[1] += (i-1) * velocity * timestep
    end
    # compute finite difference velocity
    states = [[q; zeros(nq)] for q in configurations]
    for i = 2:T
        states[i][nq .+ (1:nq)] = (configurations[i] - configurations[i-1]) / timestep
    end
    states[1][nq .+ (1:nq)] = states[end][nq .+ (1:nq)]
    return states
end