using RigidBodyDynamics
using JLD2
using LinearAlgebra
using MeshCat, MeshCatMechanisms
using StaticArrays
using Rotations
using GeometryBasics, Colors
include("ik.jl")
include("rope.jl") 

# ## fix static arrays 
StaticArrays.has_size(::Type{SVector}) = false

# ## URDF 
cd(joinpath(@__DIR__, "..", "..", "meshes"))
urdf = "kuka.urdf"
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

# ## Visualizer 
vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
open(vis)

# ## end-effector
body = findbody(mechanism, "iiwa_link_7")
point = Point3D(default_frame(body), 0., 0, 0)

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

# ## Initial configuration 
set_configuration!(state, deg2rad.([0.0, -45.0, 0.0, -145.0, 0.0, 30.0, 0.0]))
set_configuration!(vis, configuration(state))

# ## Shift 
@show current_point = transform(state, point, root_frame(mechanism))
shift = current_point.v + [0.0; 0.0; -0.0]

# ## Animation
anim = MeshCat.Animation(convert(Int, floor(1.0 / 0.075)))
build_rope(vis.visualizer, N=50, name=:rope2, rope_radius=0.01)

for (t, q) in enumerate([[q_sol[1] for t = 1:10]..., q_sol..., [q_sol[end] for t = 1:10]...])
    MeshCat.atframe(anim, t) do
        my_new_p = [q[1]; 0.0; q[2]] + shift + 1.0e-3 * ones(3)
        my_new_b = [q[3]; 0.0; q[4]] + shift - 1.0e-3 * ones(3)
        
        if t > 20 + 10 
            my_new_b = my_new_p + [0.05; 0.0; 0.05]
        end

        desired_tip_location = Point3D(root_frame(mechanism), my_new_p...)
        jacobian_transpose_ik!(state, body, point, desired_tip_location)
        set_configuration!(vis, configuration(state))

        settransform!(vis["cup"], MeshCat.Translation(my_new_p))
        settransform!(vis["ball"], MeshCat.Translation(my_new_b))

        set_loose_rope(vis.visualizer, t > 20 + 10 ? Array(my_new_b) + [-0.025; 1.0e-3; 1.0e-3] : Array(my_new_p), Array(my_new_b), rope_length=1.0, N=50, min_altitude=-10.0, name=:rope2)
    end
end

MeshCat.setanimation!(vis, anim)

# ## set scene
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

# reset directory
cd(@__DIR__)
