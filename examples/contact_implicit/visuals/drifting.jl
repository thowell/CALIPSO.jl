function set_background!(vis::Visualizer; top_color=RoboDojo.RGBA(1,1,1.0), bottom_color=RoboDojo.RGBA(1,1,1.0))
    RoboDojo.MeshCat.setprop!(vis["/Background"], "top_color", top_color)
    RoboDojo.MeshCat.setprop!(vis["/Background"], "bottom_color", bottom_color)
end

function visualize!(vis, model::CYBERTRUCK, q;
    scale=0.1,
    Δt = 0.1)

    set_background!(vis)

    path_meshes = joinpath(@__DIR__, "..", "..", "..", "robot_meshes")
    meshfile = joinpath(path_meshes, "cybertruck", "cybertruck.obj")

    obj = RoboDojo.MeshCat.MeshFileObject(meshfile);
    
    RoboDojo.MeshCat.setobject!(vis["cybertruck"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    RoboDojo.MeshCat.setobject!(vis["cybertruck1"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck1"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    RoboDojo.MeshCat.setobject!(vis["cybertruck2"]["mesh"], obj)
    RoboDojo.MeshCat.settransform!(vis["cybertruck2"]["mesh"], RoboDojo.MeshCat.LinearMap(scale * RoboDojo.Rotations.RotZ(1.0 * pi) * RoboDojo.Rotations.RotX(pi / 2.0)))

    anim = RoboDojo.MeshCat.Animation(convert(Int,floor(1.0 / Δt)))

    for t = 1:length(q)
        RoboDojo.MeshCat.atframe(anim, t) do
            RoboDojo.MeshCat.settransform!(vis["cybertruck"], RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(q[t][1:2]..., 0.0), RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(q[t][3]))))
        end
    end

    RoboDojo.MeshCat.settransform!(vis["/Cameras/default"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation(2.0, 0.0, 1.0),RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.0))))
    
    # add parked cars
    
    RoboDojo.MeshCat.settransform!(vis["cybertruck1"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car1...; 0.0]),
        RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.5 * π) * RoboDojo.Rotations.RotX(0))))

    RoboDojo.MeshCat.settransform!(vis["cybertruck2"],
        RoboDojo.MeshCat.compose(RoboDojo.MeshCat.Translation([p_car2...; 0.0]),
        RoboDojo.MeshCat.LinearMap(RoboDojo.Rotations.RotZ(0.5 * π) * RoboDojo.Rotations.RotX(0))))

    RoboDojo.MeshCat.setanimation!(vis, anim)

    # set scene
    RoboDojo.settransform!(vis["/Cameras/default"],
	RoboDojo.compose(RoboDojo.Translation(0.0, 0.0, 10.0), RoboDojo.LinearMap(RoboDojo.RotY(-pi/2.5))))
    RoboDojo.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 3)
end
