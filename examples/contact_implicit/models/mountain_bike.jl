# mountain bike dynamics
const m_b = 10.0

const m1 = 1.0
const m2 = 1.0
const h = 0.2
const gravity = [0;-9.8]

const r_wheel_base = 2.0

function control_forces(r1,r2,u)
    # forces between the rider mass and the bike
    r12 = (-r1 + r2)/norm(-r1 + r2)

    u*r12, -u*r12
end

function trans_del(m,r1,r2,r3)
    # translational discrete euler-lagrange
    (1/h)*m*(r2-r1) - (1/h)*m*(r3-r2)
end

function c(q)
    # distance equality constraint between wheels of the bike
    r1 = q[1:2]
    r2 = q[3:4]
    dot(r1-r2,r1-r2) - r_wheel_base^2
end
function Dc(q)
    # jacobian of c
    r1 = q[1:2]
    r2 = q[3:4]
    t0 = (r1-r2)
    g = 2*[t0;-t0;0;0]
    Matrix(transpose(g))
end
function d(q)
    # wheels must be above ground, rider must be above .3 m
    [q[2];q[4];q[6]-0.3]
end
function Dd(q)
    # jacobian of d
    [0 1 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 1]
end

function DEL(q1,q2,q3,λ,u₋,u₊)
    # discrete euler lagrange equation

    # pull out states
    r1₋ = q1[1:2];  r2₋ = q1[3:4];  r3₋ = q1[5:6]
    r1  = q2[1:2];  r2  = q2[3:4];  r3  = q2[5:6]
    r1₊ = q3[1:2];  r2₊ = q3[3:4];  r3₊ = q3[5:6]

    # gravity forces
    F_g = [
            m1*gravity;
            m2*gravity;
            m_b*gravity
            ]



    # - time step
    f1₋, f3₋1 = control_forces(.5*(r1₋ + r1),.5*(r3₋ + r3),u₋[1]) # 1 and 3
    f2₋, f3₋2 = control_forces(.5*(r2₋ + r2),.5*(r3₋ + r3),u₋[2]) # 2 and 3

    # + time step
    f1₊, f3₊1 = control_forces(.5*(r1₊ + r1),.5*(r3₊ + r3),u₊[1]) # 1 and 3
    f2₊, f3₊2 = control_forces(.5*(r2₊ + r2),.5*(r3₊ + r3),u₊[2]) # 2 and 3


    F₋ = F_g + [f1₋;f2₋;f3₋1 + f3₋2]
    F₊ = F_g + [f1₊;f2₊;f3₊1 + f3₊2]

    [
        trans_del(m1,r1₋,r1,r1₊);
        trans_del(m2,r2₋,r2,r2₊);
        trans_del(m_b,r3₋,r3,r3₊)
    ] + (h/2.0)*F₋ + (h/2.0)*F₊
end

function mountain_bike_dynamics(q1,q2,q3,λ,η,u1,u2,κ)
    # KKT conditions for NCP between sets of 3 timesteps

    [#  DEL                               LINK               CONTACT
        DEL(q1,q2,q3,λ,u1,u2) + h*transpose(Dc(q2))*λ + h*transpose(Dd(q2))*η; # 6
        c(q3);            # LINK CONSTRAINT     # 1
        η .* d(q3) .- κ;  # CONTACT CONSTRAINT  # 3
    ]
end
