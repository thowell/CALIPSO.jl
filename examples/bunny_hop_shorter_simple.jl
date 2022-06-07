# CALIPSO
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO

# Examples
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra


# bunny hop dynamics
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

function contact_kkt(q1,q2,q3,λ,η,u1,u2,κ)
    # KKT conditions for NCP between sets of 3 timesteps

    [#  DEL                               LINK               CONTACT
        DEL(q1,q2,q3,λ,u1,u2) + h*transpose(Dc(q2))*λ + h*transpose(Dd(q2))*η; # 6
        c(q3);            # LINK CONSTRAINT     # 1
        η .* d(q3) .- κ;  # CONTACT CONSTRAINT  # 3
    ]
end

# problem indexing for NLP
N = 10

# primal variables indexing
bidx_q = [(i-1)*6 .+ (1:6) for i = 1:N]
bidx_u = [(bidx_q[end][end] + (i-1)*2) .+ (1:2) for i = 1:N-1]
bidx_λ = [(bidx_u[end][end] + (i-1)*1) .+ (1:1) for i = 1:N-2]
bidx_η = [(bidx_λ[end][end] + (i-1)*3) .+ (1:3) for i = 1:N-2]
nz = bidx_η[end][end]

# equality constraint indexing
bidx_c = [(i-1)*10 .+ (1:10) for i = 1:N-2] # dynamics constraints
push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q1 constraint
push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q2 constraint
push!(bidx_c, bidx_c[end][end] .+ (1:2)) # jump_constraint constraint
nc = bidx_c[end][end]

# inequality constraint indexing
bidx_h = [(i-1)*3 .+ (1:3) for i = 1:N-2] # here is η > 0
bidx_h2 = [(bidx_h[end][end] + (i-1)*3) .+ (1:3) for i = 1:N] # Jc(q) >0
nh = bidx_h2[end][end]

# initial configuration
q0 = [
        -r_wheel_base/2,
        0,
        r_wheel_base/2,
        0,
        0,
        sqrt(1.5^2-(r_wheel_base/2)^2)
]

# next configuration
q1 = 1*q0 + 11*h*[1,0,1,0,1,0]

# reference control and state histories
usref = [ -0.95*sqrt(2)*.5*m_b*9.8*ones(2) for i = 1:N-1]
qsref = [q0 + 11*h*(i-1)*[1,0,1,0,1,0] for i = 1:N]

# cost matrices
Q1 = Diagonal([0, 1, 0, 1, 0, .1])
Q2 = Diagonal([0, 1, 0, 1, 0, 1e4])
R = .01*Diagonal(ones(2))


function equality_constraint(z)
    # dynamics constraints
    c = zeros(eltype(z),nc)
    for i = 1:N-2

        q₋ = z[bidx_q[i]]
        qm = z[bidx_q[i+1]]
        q₊= z[bidx_q[i+2]]
        λ = z[bidx_λ[i]]
        η = z[bidx_η[i]]
        u1 = z[bidx_u[i]]
        u2 = z[bidx_u[i+1]]
        c[bidx_c[i]] = contact_kkt(q₋,qm,q₊,λ,η,u1,u2,0)
    end

    # initial conditions + jump constraint
    c[bidx_c[N-1]] = z[bidx_q[1]] - q0
    c[bidx_c[N]] = z[bidx_q[2]] - q1
    c[bidx_c[N+1]] = z[bidx_q[5]][[2,4]] - [1.1;1.15]
    return c
end


function inequality_constraints(z)
    # inequality constraints
    ineq = zeros(eltype(z),nh)
    for i = 1:N-2
        ineq[bidx_h[i]] = z[bidx_η[i]] # 4
    end
    for i = 1:N
        qq = z[bidx_q[i]]
        ineq[bidx_h2[i]] = d(qq)  # 4
    end
    return ineq
end

function cost_function(z)
    J = zero(eltype(z))
    for i = 1:N
        dq = z[bidx_q[i]] - qsref[i]
        if i > 6
            J += .5*transpose(dq)*Q2*dq
        else
            J += .5*transpose(dq)*Q1*dq
        end

        qq = z[bidx_q[i]]
        x1 = qq[1]
        x2 = qq[3]
        xb = qq[5]
        J += 10*(xb - 0.5*(x1 + x2))^2
    end
    for i = 1:N-1
        # cost on velocity
        du = z[bidx_u[i]] - usref[i]
        J += .5*transpose(du)*R*du
        qq1 = z[bidx_q[i]]
        qq2 = z[bidx_q[i+1]]
        J += .005*transpose(qq1-qq2)*(qq1-qq2)
    end
    for i = 1:N-2
        # cost on acceleration
        qq1 = z[bidx_q[i]][[1,3,5]]
        qq2 = z[bidx_q[i+1]][[1,3,5]]
        qq3 = z[bidx_q[i+2]][[1,3,5]]
        a = (qq3 - 2*qq2 + qq1)
        J += 10*transpose(a)*a
    end
    return J
end

# initial guess
initial_z = 10*ones(nz)
for i = 1:N
    initial_z[bidx_q[i]] .= qsref[i] + .001*abs.(randn(6)) #+ .5*[0,1,0,1,0,1]
end
for i = 1:N-1
    initial_z[bidx_u[i]] .= usref[i] + .001*randn(2)
end
for i = 1:N-2
    initial_z[bidx_η[i]] = 9.8*[m1;m2;m_b]
end

solver = Solver(cost_function, equality_constraint, inequality_constraints, nz; nonnegative_indices = collect(1:nh))

using Random
Random.seed!(1234)
initialize!(solver,initial_z + .001*randn(nz))
solver.options.penalty_initial = 1e2
solve!(solver)

# qs = [solver.solution.variables[bidx_q[i]] for i = 1:N]
# us = [solver.solution.variables[bidx_u[i]] for i = 1:N-1]
#
# Qm = hcat(qs...)
# Um = hcat(us...)
# Qreffm = hcat(qsref...)
# #
# using JLD2
# jldsave("bunny_hop_simple_v30_v2.jld2";qs)
#
# using MATLAB
# mat"
# figure
# hold on
# plot($Qm([2,4,6],:)')
# legend('m1','m2','mb')
# hold off
# "
#
# mat"
# figure
# hold on
# plot($Um')
# hold off
# "
#
# mat"
# figure
# hold on
# plot($Qreffm([2,4,6],:)')
# legend('m1','m2','mb')
# hold off
# "
#
#
# mat"
# figure
# hold on
# plot($Qm([1,3,5],:)')
# legend('m1','m2','mb')
# hold off
# "
# # using MeshCat, GeometryBasics, CoordinateTransformations, Rotations
# #
# # # /Users/kevintracy/devel/TrustRegionSQP/bunny_hop.jld2
# # using Colors
# # vis = Visualizer()
# # black_color = MeshPhongMaterial(color=RGBA(0, 0, 0, 0.5))
# # red_color = MeshPhongMaterial(color=RGBA(1, 0, 0, 0.5))
# # green_color = MeshPhongMaterial(color=RGBA(0, 1, 0, 0.5))
# # sphere2 = Sphere(Point3(0.0), .3)
# # setobject!(vis["m3"],sphere2)
# #
# # cyl = Cylinder(Point3(0,-.1,0.0),Point3(0,.1,0),0.5f0)
# # setobject!(vis["m1"],cyl,black_color)
# # setobject!(vis["m2"],cyl,black_color)
# # cyl2 = Cylinder(Point3f(-r_wheel_base/2,0,0.0),Point3f(r_wheel_base/2,0,0),0.1f0)
# # setobject!(vis["bike"],cyl2,red_color)
# #
# # anim = MeshCat.Animation(floor(Int,1/0.2))
