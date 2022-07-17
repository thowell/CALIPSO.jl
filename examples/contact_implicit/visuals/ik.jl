# RigidBodyDynamics.jl
# ## Inverse kinematics 
function jacobian_transpose_ik!(state::MechanismState,
    body::RigidBody,
    point::Point3D,
    desired::Point3D;
    α=0.1,
    iterations=1000,
    verbose=false)
    
    mechanism = state.mechanism
    world = root_frame(mechanism)

    # Compute the joint path from world to our target body
    p = path(mechanism, root_body(mechanism), body)
    # Allocate the point jacobian (we'll update this in-place later)
    Jp = point_jacobian(state, p, transform(state, point, world))

    q = copy(configuration(state))

    for i in 1:iterations
        # Update the position of the point
        point_in_world = transform(state, point, world)
        # Update the point's jacobian
        point_jacobian!(Jp, state, p, point_in_world)
        dx = (transform(state, desired, world) - point_in_world).v
        if norm(dx)<1e-3
            verbose && (@info "success")
            break
        end
        # Compute an update in joint coordinates using the jacobian transpose
        Δq = α * Array(Jp)' * dx
        # Apply the update
        q .= configuration(state) .+ Δq
        set_configuration!(state, q)
    end
    state
end