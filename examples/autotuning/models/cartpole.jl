# ## dynamics
function cartpole_continuous(x, u)
    mc = 1.0 
    mp = 0.2 
    l = 0.5 
    g = 9.81 

    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc + mp mp * l * c; mp * l * c mp * l^2]
    Hinv = 1.0 / (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) * [H[2, 2] -H[1, 2]; -H[2, 1] H[1, 1]]
    
    C = [0 -mp * qd[2] * l * s; 0 0]
    G = [0, mp * g * l * s]
    B = [1, 0]

    qdd = -Hinv * (C * qd + G - B * u[1])

    return [qd; qdd]
end

function cartpole_discrete(x, u)
    h = timestep # timestep 
    x + h * cartpole_continuous(x + 0.5 * h * cartpole_continuous(x, u), u)
end

function cartpole_discrete(y, x, u)
    y - cartpole_discrete(x, u)
end

using RoboDojo
function cable_transform(y, z)
    v1 = [0.0, 0.0, 1.0]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1, v2)
    ang = acos(v1'*v2)
    R = RoboDojo.AngleAxis(ang, ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    RoboDojo.compose(RoboDojo.RoboDojo.Translation(z), RoboDojo.RoboDojo.LinearMap(R))
end

function default_background!(vis)
    RoboDojo.RoboDojo.setvisible!(vis["/Background"], true)
    RoboDojo.setprop!(vis["/Background"], "top_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setprop!(vis["/Background"], "bottom_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.RoboDojo.setvisible!(vis["/Axes"], false)
end

function _create_cartpole!(vis, model;
    i = 0,
    tl = 1.0,
    color = RoboDojo.Colors.RGBA(0, 0, 0, tl))

    l2 = RoboDojo.Cylinder(RoboDojo.Point3f0(-0.5 * 10.0, 0.0, 0.0),
        RoboDojo.Point3f0(0.5 * 10.0, 0.0, 0.0),
        convert(Float32, 0.0125))

    RoboDojo.setobject!(vis["slider_$i"], l2, RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))

    l1 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0, 0.0, 0.0),
        RoboDojo.Point3f0(0.0, 0.0, 0.5),
        convert(Float32, 0.025))

    RoboDojo.setobject!(vis["arm_$i"], l1,
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))

    RoboDojo.setobject!(vis["base_$i"], RoboDojo.HyperSphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.1)),
        RoboDojo.MeshPhongMaterial(color = color))

    RoboDojo.setobject!(vis["ee_$i"], RoboDojo.HyperSphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = color))
end

function _set_cartpole!(vis, model, x;
    i = 0)

    px = x[1] + 0.5 * sin(x[2])
    pz = -0.5 * cos(x[2])
    RoboDojo.settransform!(vis["arm_$i"], cable_transform([x[1]; 0;0], [px; 0.0; pz]))
    RoboDojo.settransform!(vis["base_$i"], RoboDojo.Translation([x[1]; 0.0; 0.0]))
    RoboDojo.settransform!(vis["ee_$i"], RoboDojo.Translation([px; 0.0; pz]))
end

function visualize_cartpole!(vis, model, q;
    i = 0,
    tl = 1.0,
    Δt = 0.1,
    color = RoboDojo.Colors.RGBA(0,0,0,1.0))

    default_background!(vis)
    _create_cartpole!(vis, model, i = i, color = color, tl = tl)

    anim = RoboDojo.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    for t = 1:length(q)
        RoboDojo.MeshCat.atframe(anim,t) do
            _set_cartpole!(vis, model, q[t], i = i)
        end
    end

    RoboDojo.settransform!(vis["/Cameras/default"],
        RoboDojo.compose(RoboDojo.Translation(0.0, 0.0, -1.0), RoboDojo.LinearMap(RoboDojo.RotZ(- pi / 2))))
    RoboDojo.RoboDojo.setvisible!(vis["/Grid"], false)

    RoboDojo.MeshCat.setanimation!(vis,anim)
end