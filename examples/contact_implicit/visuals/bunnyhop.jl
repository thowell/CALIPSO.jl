using RigidBodyDynamics
using JLD2
using LinearAlgebra
using MeshCat, MeshCatMechanisms
using Interpolations
using StaticArrays
using Rotations
using RoboDojo
using GeometryBasics

# ## fix static arrays 
StaticArrays.has_size(::Type{SVector}) = false

# ## URDF 
cd(joinpath(@__DIR__, "..", "..", "meshes"))
urdf = "atlas.urdf"
mechanism = parse_urdf(urdf)
state = MechanismState(mechanism)

# ## Visualizer 
vis = MechanismVisualizer(mechanism, URDFVisuals(urdf))
open(vis)

# ## Atlas configuration
q = configuration(state)

q[1] = 0.0 * π
q[2] = 0.0 * π
q[3] = 0.0 * π
q[4] = 0.25 * π # torso bend
q[5] = 0.0 * π
q[6] = 0.0 * π
q[7] = 0.0 * π
q[8] = -0.15 * π # thigh waist bend
q[9] = -0.4 * π # thigh waist bend
q[10] = -0.5 * π # upper arm twist
q[11] = -0.0 * π # head
q[12] = 0.5 * π # upper arm twist
q[13] = 0.125 * π # knee bend
q[14] = 0.55 * π # knee bend
q[15] = -0.125 * π # upper arm bend
q[16] = 0.125 * π # upper arm bend
q[17] = 0.125 * π # ankle bend
q[18] = -0.2 * π # ankle bend
q[19] = 0.0 * π 
q[20] = 0.0 * π 
q[21] = 0.0 * π 
q[22] = 0.0 * π 
q[23] = 0.55 * π # elbow bend
q[24] = -0.55 * π # elbow bend
q[25] = 0.0 * π # 
q[26] = 0.0 * π # 
q[27] = 0.0 * π # 
q[28] = 0.0 * π # 
q[29] = 0.0 * π # 
q[30] = 0.5 * π # 

set_configuration!(vis, q)

settransform!(vis["world"], MeshCat.Translation(0.0, 0.0, 1.5))

# ## environment 
meshfile_cone = "traffic_cone/traffic_cone.obj"
obj_cone = MeshFileObject(meshfile_cone);

# ## bike 
meshfile_bike = "mtb/mtb.obj"
obj_bike = MeshFileObject(meshfile_bike);
const m_b = 5.0 
const m1 = 1.0 
const m2 = 1.0 
const h = 0.1 
const gravity = [0; -9.8]
const r_wheel_base = 2.0 

setobject!(vis["world"]["mountain_bike"]["base"], obj_bike)
settransform!(vis["world"]["mountain_bike"]["base"], MeshCat.Translation([0.5, 0.0, -0.55]) ∘ MeshCat.LinearMap(0.25 * RotZ(0.5 * π) * RotX(0.5 * π)))

setobject!(vis["traffic_cone"], obj_cone)
settransform!(vis["traffic_cone"], MeshCat.LinearMap(0.003  * RotX(0.5 * π)))

# ## animation 
x_offset = 9.20

# linearly interpolate such that we get more frames to make animations out of
interp_linear = LinearInterpolation(0.0:0.2:1.8, SVector{6}.(qs))
qs = interp_linear(0:.01:1.8)
anim = MeshCat.Animation(floor(Int, 1.0 / 0.01))

for k = 1:length(qs)
    atframe(anim, k) do
        bx = normalize(-qs[k][1:2] + qs[k][3:4])
        θ = atan(bx[2], bx[1])
        r = 0.5 * ([qs[k][1] - x_offset; 0.0 ; qs[k][2] + 2.5] + [qs[k][3] - x_offset; 0.0; qs[k][4]])
        Q1 = MeshCat.LinearMap(RotY(-θ))
        settransform!(vis.visualizer["world"], MeshCat.Translation(r) ∘ Q1)
    end
end
setanimation!(vis.visualizer, anim)

camvis = vis["/Cameras/default/rotated/<object>"]
setprop!(camvis, "zoom", 5.0)
MeshCat.settransform!(camvis,
    MeshCat.compose(
        MeshCat.LinearMap(Rotations.RotX(-pi / 2.0)),
        MeshCat.Translation(0.0, -50.0, 0.0),
    ))
MeshCat.setvisible!(vis.visualizer["/Grid"], true)
MeshCat.setvisible!(vis.visualizer["/Axes"], false)

top_color=RGBA(1,1,1.0)
bottom_color=RGBA(1,1,1.0)

MeshCat.setprop!(vis.visualizer["/Background"], "top_color", top_color)
MeshCat.setprop!(vis.visualizer["/Background"], "bottom_color", bottom_color)

# reset directory
cd(@__DIR__)

