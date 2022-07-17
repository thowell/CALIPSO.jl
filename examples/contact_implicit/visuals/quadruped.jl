function build_meshrobot!(vis::Visualizer, model::Quadruped; name::Symbol=:Quadruped, α=1.0)
	default_background!(vis)

	urdf = joinpath(@__DIR__, "..", "meshes", "a1.urdf")
	package_path = joinpath(@__DIR__, "..", "meshes")
	build_meshrobot!(vis, model, urdf, package_path; name=name, α=α)
end

function convert_config(model::Quadruped, q::AbstractVector)
    # Quadruped configuration
    # 1.  position long axis +x
    # 2.  position long axis +z
    # 3.  trunk rotation along -y
    # 4.  back  left  shoulder rotation along -y
    # 5.  back  left  elbow    rotation along -y
    # 6.  back  right shoulder rotation along -y
    # 7.  back  right elbow    rotation along -y
    # 8.  front right shoulder rotation along -y
    # 9.  front right elbow    rotation along -y
	# 10. front left  shoulder rotation along -y
	# 11. front left  elbow    rotation along -y

    # URDF configuration
    # 1.  front right clavicle rotation along +x
    # 2.  front left  clavicle rotation along +x
    # 3.  back  right clavicle rotation along +x
    # 4.  back  left  clavicle rotation along +x
    # 5.  front right shoulder rotation along +y
    # 6.  front left  shoulder rotation along +y
    # 7.  back  right shoulder rotation along +y
    # 8.  back  left  shoulder rotation along +y
    # 9.  front right elbow    rotation relative to shoulder along +y
    # 10. front left  elbow    rotation relative to shoulder along +y
    # 11. back  right elbow    rotation relative to shoulder along +y
    # 12. back  left  elbow    rotation relative to shoulder along +y
    
    q_ = zeros(12)
    x, z, θ = q[1:3]
    q_[5]  = -q[10] + θ - π/2
    q_[6]  = -q[8] + θ - π/2
	q_[7]  = -q[4] + θ - π/2
    q_[8]  = -q[6] + θ - π/2
    q_[9]  = +q[10]-q[11]
    q_[10] = +q[8]-q[9]
	q_[11] = +q[4]-q[5]
    q_[12] = +q[6]-q[7]

	r_foot = 0.02
	trunk_length = 2*0.183
	trunk_width = 2*0.132
    tform = compose(
		Translation(x , trunk_width/2, z+r_foot),
		compose(
			LinearMap(AngleAxis(π/2-θ, 0, 1.0, 0)),
			Translation(trunk_length/2, 0,0)
			)
		)
    return q_, tform
end

function build_meshrobot!(vis::Visualizer, model, urdf::String,
    package_path::String; name::Symbol=model_name(model), α=1.0)
    default_background!(vis)

    mechanism = MeshCatMechanisms.parse_urdf(urdf)
    visuals = URDFVisuals(urdf, package_path=[package_path])
    state = MeshCatMechanisms.MechanismState(mechanism)
    vis_el = MeshCatMechanisms.visual_elements(mechanism, visuals)
    set_alpha!(vis_el,α)

    mvis = MechanismVisualizer(state, vis[name, :world])
    MeshCatMechanisms._set_mechanism!(mvis, vis_el)
    MeshCatMechanisms._render_state!(mvis)
    return mvis
end

function set_alpha!(visuals::Vector, α)
    for el in visuals
        c = el.color
        c_new = RGBA(red(c),green(c),blue(c),α)
        el.color = c_new
    end
end

function set_meshrobot!(vis::Visualizer, mvis::MechanismVisualizer, model,
    q::AbstractVector; name::Symbol=:quadruped)

    q_mesh, tform = convert_config(model, q)
    set_configuration!(mvis, q_mesh)
    settransform!(vis[name, :world], tform)

    return nothing
end

function animate_meshrobot!(vis::Visualizer, mvis::MechanismVisualizer, anim::MeshCat.Animation,
    model, q::AbstractVector; name::Symbol=:quadruped)
    for t in 1:length(q)
        MeshCat.atframe(anim, t) do
            set_meshrobot!(vis, mvis, model, q[t], name=name)
        end
    end
    setanimation!(vis, anim)
    return nothing
end

function visualize_meshrobot!(vis::Visualizer, model, q::AbstractVector;
    h=0.01,
    anim::MeshCat.Animation=MeshCat.Animation(Int(floor(1/h))),
    name::Symbol=model_name(model),
    α=α)

    mvis = build_meshrobot!(vis, model, name=name, α=α)
    animate_meshrobot!(vis, mvis, anim, model, q, name=name)

    return anim
end

function visualize_meshrobot!(vis::Visualizer, model, traj;
    h=0.01,
    anim::MeshCat.Animation=MeshCat.Animation(Int(floor(1/h))),
    name::Symbol=:quadruped)

    anim = visualize_meshrobot!(vis, model, traj; 
        anim=anim, 
        name=name, 
        h=h, 
        α=1.0)
    
    return anim
end