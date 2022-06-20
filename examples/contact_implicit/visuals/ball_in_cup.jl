using RigidBodyDynamics
using LinearAlgebra
using StaticArrays
using MeshCat, MeshCatMechanisms, Blink
using GeometryBasics, Colors, Rotations
using JLD2
using RobotVisualizer

cd(joinpath(dirname(@__FILE__),"kuka"))
urdf = "/Users/taylor/Research/robot_meshes/kuka/kuka.urdf"

meshfile = joinpath(@__DIR__,"traffic_cone/traffic_cone.obj")
obj = MeshFileObject(meshfile);
setobject!(vis["traffic_cone"], obj)
scale = 0.004
settransform!(vis["traffic_cone"], LinearMap(scale*RotX(pi/2)))

meshfile = joinpath(@__DIR__,"mtb/mtb.obj")
obj = MeshFileObject(meshfile);
setobject!(vis["mountain_bike"]["base"], obj)
scale = 0.25
settransform!(vis["mountain_bike"]["base"], Translation([0,0,0.7]) ∘ LinearMap(scale*RotZ(pi/2)*RotX(pi/2)))

state = MechanismState(mechanism)
mechanism

visuals = URDFVisuals(urdf, package_path=["/Users/taylor/Research/robot_meshes/kuka"])
state = MeshCatMechanisms.MechanismState(mechanism)
vis_el = MeshCatMechanisms.visual_elements(mechanism, visuals);

@load "/Users/taylor/Research/CALIPSO.jl/test/examples/ballincup.jld2"

body = findbody(mechanism, "iiwa_link_7")
# here is the point we are trying to control (position is relative to "body" link)
point = Point3D(default_frame(body), 0., 0, 0)

vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
render(vis)

visuals = URDFVisuals(urdf, package_path=["/home/taylor/Research/robot_meshes/kuka"])
state = MeshCatMechanisms.MechanismState(mechanism)
vis_el = MeshCatMechanisms.visual_elements(mechanism, visuals)

function set_alpha!(visuals::Vector, α)
    for el in visuals
        c = el.color
        c_new = RGBA(red(c),green(c),blue(c),α)
        el.color = c_new
    end
end

handle = GeometryBasics.Cylinder(Point3f0(0.0, 0.0, 0.0),
    Point3f0(0.1, 0.0, 0.0),
    convert(Float32, 0.02))
basin = GeometryBasics.Cylinder(Point3f0(-0.05, 0.0, 0.0),
    Point3f0(-0.0, 0.0, 0.0),
    convert(Float32, 0.06))
setobject!(vis["world/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/after_iiwa_joint_7"][:cup][:handle], handle, MeshPhongMaterial(color=RGBA(238 / 255, 213 / 255, 174 / 255, 1.0)))
setobject!(vis["world/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/after_iiwa_joint_7"][:cup][:basin], basin, MeshPhongMaterial(color=RGBA(238 / 255, 213 / 255, 174 / 255, 1.0)))
settransform!(vis["world/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/after_iiwa_joint_7"][:cup][:handle], MeshCat.Translation(-0.01, 0.0, 0.05))
settransform!(vis["world/iiwa_link_1/iiwa_link_2/iiwa_link_3/iiwa_link_4/iiwa_link_5/iiwa_link_6/iiwa_link_7/after_iiwa_joint_7"][:cup][:basin], MeshCat.Translation(0.0, 0.0, 0.05))


setobject!(vis["ball"], GeometryBasics.Sphere(Point3f0(0),
        convert(Float32, 0.03)),
    MeshPhongMaterial(color=RGBA(1.0, 0.0, 0.0, 1.0)))

my_new_p = [q_sol[1][1]; 0.0; q_sol[1][2]] + shift
desired_tip_location = Point3D(root_frame(mechanism), my_new_p...)
jacobian_transpose_ik!(state, body, point, desired_tip_location)
set_configuration!(vis, configuration(state))

my_new_p = [q_sol[1][1]; 0.0; q_sol[1][2]] + shift
my_new_b = [q_sol[1][3]; 0.0; q_sol[1][4]] + shift

settransform!(vis["cup"], MeshCat.Translation(my_new_p))
settransform!(vis["ball"], MeshCat.Translation(my_new_b))

function jacobian_transpose_ik!(state::MechanismState,
    body::RigidBody,
    point::Point3D,
    desired::Point3D;
    α=0.1,
    iterations=1000)
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
            @info "success"
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

@load "/home/taylor/Research/CALIPSO.jl/test/examples/ballincup.jld2" q_sol


# new position I want to drive my point to 
my_new_p = [q_sol[1][1]; 0.0; q_sol[1][2]] + shift
desired_tip_location = Point3D(root_frame(mechanism), my_new_p...)

jacobian_transpose_ik!(state, body, point, desired_tip_location)
set_configuration!(vis, configuration(state))

anim = MeshCat.Animation(convert(Int, floor(1.0 / 0.075)))
RobotVisualizer.build_rope(vis.visualizer, N=50, name=:rope2, rope_radius=0.01)

for (t, q) in enumerate([[q_sol[1] for t = 1:10]..., q_sol..., [q_sol[end] for t = 1:10]...])
    MeshCat.atframe(anim, t) do
        my_new_p = [q[1]; 0.0; q[2]] + shift + 1.0e-3 * ones(3)
        my_new_b = [q[3]; 0.0; q[4]] + shift - 1.0e-3 * ones(3)
        
        if t > 20 + 10 
            my_new_b = my_new_p + [0.05; 0.0; 0.05]
        end
#         # check for caught ball
#         if norm(q[1:2] - q[3:4]) < 0.2
#             flag = true 
#         end
        
#         if flag 
#             my_new_ball = my_new_p 
#         end
        
        desired_tip_location = Point3D(root_frame(mechanism), my_new_p...)
        jacobian_transpose_ik!(state, body, point, desired_tip_location)
        set_configuration!(vis, configuration(state))

        settransform!(vis["cup"], MeshCat.Translation(my_new_p))
        settransform!(vis["ball"], MeshCat.Translation(my_new_b))

        set_loose_rope(vis.visualizer, t > 20 + 10 ? Array(my_new_b) + [-0.025; 1.0e-3; 1.0e-3] : Array(my_new_p), Array(my_new_b), rope_length=1.0, N=50, min_altitude=-10.0, name=:rope2)


#         @show norm(transform(state, point, root_frame(mechanism)).v - my_new_p, Inf)
    end
end
MeshCat.setanimation!(vis, anim)

# MeshCat.settransform!(vis.visualizer["/Cameras/default"],
# MeshCat.compose(MeshCat.Translation(0.0, -50.0, -1.0), MeshCat.LinearMap(Rotations.RotZ(-pi / 2.0))))
# setprop!(vis.visualizer["/Cameras/default/rotated/<object>"], "zoom", 25)
camvis=vis["/Cameras/default/rotated/<object>"]
setprop!(camvis, "zoom", 25.0)
MeshCat.settransform!(camvis,
    MeshCat.compose(
        MeshCat.LinearMap(Rotations.RotX(-pi / 2.0)),
        MeshCat.Translation(0.0, -50.0, 0.0),
    ))
MeshCat.setvisible!(vis.visualizer["/Grid"], false)
MeshCat.setvisible!(vis.visualizer["/Axes"], false)

top_color=RGBA(1,1,1.0)
bottom_color=RGBA(1,1,1.0)

MeshCat.setprop!(vis.visualizer["/Background"], "top_color", top_color)
MeshCat.setprop!(vis.visualizer["/Background"], "bottom_color", bottom_color)

set_light!(vis.visualizer)
# set_floor!(vis.visualizer)

# build_rope(vis, N=50, name=:rope1)
# RobotVisualizer.build_rope(vis.visualizer, N=10, name=:rope2)
# set_straight_rope(vis.visualizer, [0,0,0.0], [1,1,1.0], N=50, name=:rope1)
# set_loose_rope(vis.visualizer, [0,0,0.0], [1,1,1.0], rope_length=10, N=50, min_altitude=-0.0, name=:rope2)
goal_traj = [Array([q[1], 0, q[2]] + shift + [1.0e-3; 1.0e-3; 1.0e-3]) for q in q_sol]
start_traj =  [Array([q[3], 0, q[4]] + shift - [1.0e-3; 1.0e-3; 1.0e-3]) for q in q_sol]

RobotVisualizer.build_rope(vis.visualizer, N=50, name=:rope2, rope_radius=0.01)
# set_straight_rope(vis.visualizer, [0,0,0.0], [1,1,1.0], N=10, name=:rope1)

t = 21
set_loose_rope(vis.visualizer, Array(start_traj[t]), Array(goal_traj[t]), rope_length=2.0, N=50, min_altitude=-10.0, name=:rope2)


# anim = Animation(30)
# # vis, anim = animate_straight_rope(vis, start_traj, goal_traj, name=:rope1)
# vis, anim = animate_loose_rope(vis.visualizer, start_traj, goal_traj, rope_length=1.0,
#     anim=anim, name=:rope2, min_altitude=-1.0)

# start_traj = [[-1-sin(2π*i/100), +1+0.00i,  2-0.0i] for i = 1:100];
# goal_traj =  [[+1+sin(2π*i/100), -1-0.00i,  1.8+0.2sin(2π*i/100)] for i = 1:100];

# vis_visualizer, anim = animate_loose_rope(vis.visualizer, start_traj, goal_traj, rope_length=2.0,
#     anim=anim, name=:rope2, min_altitude=-10.0)

convert_frames_to_video_and_gif("ballincup", true)

render_static(vis.visualizer)


open("/home/taylor/Downloads/ballincup.html", "w") do file
    write(file, static_html(vis.visualizer))
end

