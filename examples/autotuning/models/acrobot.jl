# ## dynamics
function acrobot_continuous(x, u)
    mass1 = 1.0  
    inertia1 = 0.33  
    length1 = 1.0 
    lengthcom1 = 0.5 

    mass2 = 1.0  
    inertia2 = 0.33  
    length2 = 1.0 
    lengthcom2 = 0.5 

    gravity = 9.81 
    friction1 = 0.25
    friction2 = 0.25

    function M(x)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

        return [a b; b c]
    end

    function Minv(x)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

        return 1.0 / (a * c - b * b) * [c -b; -b a]
    end

    function τ(x)
        a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
            - mass2 * gravity * (length1 * sin(x[1])
            + lengthcom2 * sin(x[1] + x[2])))

        b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

        return [a; b]
    end

    function C(x)
        a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
        d = 0.0

        return [a b; c d]
    end

    function B(x)
        [0.0; 1.0]
    end

    q = x[1:2]
    v = x[3:4]

    qdd = Minv(q) * (-1.0 * C(x) * v
            + τ(q) + B(q) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

function acrobot_discrete(x, u)
    h = timestep # timestep 
    x + h * acrobot_continuous(x + 0.5 * h * acrobot_continuous(x, u), u)
end

function acrobot_discrete(y, x, u)
    y - acrobot_discrete(x, u)
end

# visualsusing RoboDojo
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

    RoboDojo.compose(RoboDojo.Translation(z), RoboDojo.LinearMap(R))
end

function default_background!(vis)
    RoboDojo.setvisible!(vis["/Background"], true)
    RoboDojo.setprop!(vis["/Background"], "top_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setprop!(vis["/Background"], "bottom_color", RoboDojo.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    RoboDojo.setvisible!(vis["/Axes"], false)
end

# visualization
function _create_acrobot!(vis, model;
    tl = 1.0,
    limit_color = RoboDojo.Colors.RGBA(0.0, 1.0, 0.0, tl),
    color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl),
    i = 0,
    r = 0.1)

    l1 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0, 0.0, 0.0), RoboDojo.Point3f0(0.0, 0.0, 1.0),
        convert(Float32, 0.025))
    RoboDojo.setobject!(vis["l1_$i"], l1, RoboDojo.MeshPhongMaterial(color = color))
    l2 = RoboDojo.Cylinder(RoboDojo.Point3f0(0.0,0.0,0.0), RoboDojo.Point3f0(0.0, 0.0, 1.0),
        convert(Float32, 0.025))
    RoboDojo.setobject!(vis["l2_$i"], l2, RoboDojo.MeshPhongMaterial(color = color))

    RoboDojo.setobject!(vis["elbow_nominal_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))
    RoboDojo.setobject!(vis["elbow_limit_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = limit_color))
    RoboDojo.setobject!(vis["ee_$i"], RoboDojo.Sphere(RoboDojo.Point3f0(0.0),
        convert(Float32, 0.05)),
        RoboDojo.MeshPhongMaterial(color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, tl)))
end

function kinematics(model, x)
    [1.0 * sin(x[1]) + 1.0 * sin(x[1] + x[2]),
     -1.0 * 1.0 * cos(x[1]) - 1.0 * cos(x[1] + x[2])]
end

function _set_acrobot!(vis, model, x;
    i = 0, ϵ = 1.0e-1)

    p_mid = [kinematics_elbow(model, x)[1], 0.0, kinematics_elbow(model, x)[2]]
    p_ee = [kinematics(model, x)[1], 0.0, kinematics(model, x)[2]]

    RoboDojo.settransform!(vis["l1_$i"], cable_transform(zeros(3), p_mid))
    RoboDojo.settransform!(vis["l2_$i"], cable_transform(p_mid, p_ee))

    RoboDojo.settransform!(vis["elbow_nominal_$i"], RoboDojo.Translation(p_mid))
    RoboDojo.settransform!(vis["elbow_limit_$i"], RoboDojo.Translation(p_mid))
    RoboDojo.settransform!(vis["ee_$i"], RoboDojo.Translation(p_ee))

    if x[2] <= -0.5 * π + ϵ || x[2] >= 0.5 * π - ϵ
        RoboDojo.setvisible!(vis["elbow_nominal_$i"], false)
        RoboDojo.setvisible!(vis["elbow_limit_$i"], true)
    else
        RoboDojo.setvisible!(vis["elbow_nominal_$i"], true)
        RoboDojo.setvisible!(vis["elbow_limit_$i"], false)
    end
end

function kinematics_elbow(model, x)
    [1.0 * sin(x[1]),
     -1.0 * 1.0 * cos(x[1])]
end

# visualization
function visualize_elbow!(vis, model, x;
    tl = 1.0,
    i = 0,
    color = RoboDojo.Colors.RGBA(0.0, 0.0, 0.0, 1.0),
    limit_color = RoboDojo.Colors.RGBA(0.0, 1.0, 0.0, tl),
    r = 0.1, Δt = 0.1,
    ϵ = 1.0e-1)

    default_background!(vis)
    _create_acrobot!(vis, model,
        tl = tl,
        color = color,
        limit_color = limit_color,
        i = i,
        r = r)

    anim = RoboDojo.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(x)
    for t = 1:T
        RoboDojo.MeshCat.atframe(anim,t) do
            _set_acrobot!(vis, model, x[t], i = i, ϵ = ϵ)
        end
    end

    RoboDojo.settransform!(vis["/Cameras/default"],
       RoboDojo.compose(RoboDojo.Translation(0.0 , 0.0 , 0.0), RoboDojo.LinearMap(RoboDojo.RotZ(pi / 2.0))))

    RoboDojo.MeshCat.setanimation!(vis, anim)
end
