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
const m_b = 2.0

const r_eq = 1.5
const kp = 10.0
const kb = 1

const m1 = 1.0
const m2 = 1.0
const h = 0.1
const gravity = [0;-9.8]

const joint_min = 1.3
const joint_max = 4

const r_wheel_base = 2.0

function control_forces(r1,r2,u)
    # r12 = normalize(-r1 + r2)
    r12 = (-r1 + r2)/norm(-r1 + r2)

    u*r12, -u*r12
end

function trans_del(m,r1,r2,r3)
    (1/h)*m*(r2-r1) - (1/h)*m*(r3-r2)
end

function c(q)
    r1 = q[1:2]
    r2 = q[3:4]
    norm(r1-r2) - r_wheel_base
end
function Dc(q)
    r1 = q[1:2]
    r2 = q[3:4]
    t0 = (r1-r2)
    g = (1/norm(t0))*[t0;-t0;0;0]
    Matrix(transpose(g))
    # transpose(g)
end
function d(q)
    [q[2];q[4];q[6]]
end
function JC(q)
    r1 = q[1:2]
    r2 = q[3:4]
    r3 = q[5:6]
    [
        norm(r1-r3)  - joint_min;
        norm(r2-r3)  - joint_min;
        -norm(r1-r3) + joint_max;
        -norm(r2-r3) + joint_max
    ]
end

function DJC(q)
    r1 = q[1:2]
    r2 = q[3:4]
    r3 = q[5:6]
    t0 = transpose(r1-r3)
    t1 = transpose(r2-r3)
    [
        t0/norm(t0) zeros(1,2) -t0/norm(t0);
        zeros(1,2)  t1/norm(t1) -t1/norm(t1);
        -t0/norm(t0) zeros(1,2) t0/norm(t0);
        zeros(1,2)  -t1/norm(t1) t1/norm(t1);
    ]
end

function Dd(q)
    [0 1 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 1]
end

function DEL(q1,q2,q3,λ,u₋,u₊)
    r1₋ = q1[1:2];  r2₋ = q1[3:4];  r3₋ = q1[5:6]
    r1  = q2[1:2];  r2  = q2[3:4];  r3  = q2[5:6]
    r1₊ = q3[1:2];  r2₊ = q3[3:4];  r3₊ = q3[5:6]

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

function contact_kkt(q1,q2,q3,λ,η,y,u1,u2,κ)
    [#  DEL               LINK          CONTACT
        DEL(q1,q2,q3,λ,u1,u2) + h*transpose(Dc(q2))*λ + h*transpose(Dd(q2))*η + h*transpose(DJC(q2))*y; # 6
        c(q3);            # LINK CONSTRAINT     # 1
        η .* d(q3) .- κ;  # CONTACT CONSTRAINT  # 2
        y .* JC(q3) .- κ  # joint limits comp
    ]
end

# problem indexing
N = 30  # 30 time steps

"""
30 time steps
29 states since each state is [x_i;x_{i+1}]
28 dynamics constraints
28 control inputs
"""

# bidx_q = [(i-1)*6 .+ (1:6) for i = 1:N]
# bidx_u = [(bidx_q[end][end] + (i-1)*2) .+ (1:2) for i = 1:N-1]
# bidx_λ = [(bidx_u[end][end] + (i-1)*1) .+ (1:1) for i = 1:N-2]
# bidx_y = [(bidx_λ[end][end] + (i-1)*4) .+ (1:4) for i = 1:N-2]
# bidx_η = [(bidx_y[end][end] + (i-1)*2) .+ (1:2) for i = 1:N-2]
# nz = bidx_η[end][end]
# bidx_c = [(i-1)*13 .+ (1:13) for i = 1:N-2] # dynamics constraints
# push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q1 constraint
# push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q2 constraint
# nc = bidx_c[end][end]
#
# bidx_h = [(i-1)*6 .+ (1:6) for i = 1:N-2] # here is y > 0 and η > 0
# bidx_h2 = [(bidx_h[end][end] + (i-1)*6) .+ (1:6) for i = 1:N] # [d(q);Jc(q)] >0
# nh = bidx_h2[end][end]

q0 = [
        -r_wheel_base/2,
        0,
        r_wheel_base/2,
        0,
        0,
        sqrt(r_eq^2-(r_wheel_base/2)^2)
] #- [5,0,5,0,5,0]

q1 = 1*q0 + 3*h*[1,0,1,0,1,0]


usref = [ -0.95*sqrt(2)*.5*m_b*9.8*ones(2) for i = 1:N-1]
qsref = [q0 + 3*h*(i-1)*[1,0,1,0,1,0] for i = 1:N]

jump_length = 6
scale_up = 2
for i = 11:(11 + jump_length - 1)
    qsref[i] = qsref[i] + (scale_up)*[0,1,0,1,0,1]/jump_length
    global scale_up += 1
end
scale_down = 1
for i = (11+jump_length):(11 + 2*jump_length - 1)
    qsref[i] = qsref[i] + (jump_length - scale_down)*[0,1,0,1,0,1]/jump_length
    global scale_down += 1
end

Q = [0, 1, 0, 1, 0, 1]
R = .01*[1,1]


function midpoint_implicit(x2, x, u)
    q1  = x[1:6]
    q2₋ = x[7:12]

    λ = x[13:13]
    η = x[14:16]
    y = x[17:20]

    q2₊ = x2[1:6]
    q3  = x2[7:12]

    r = contact_kkt(q1,q2₋,q3,λ,η,y,u,u,0)

    [
        r;
        q2₊ - q2₋
    ]
end

# ## dimensions
num_states = [20 for t = 1:N-1]
num_actions = [2 for t = 1:N-2]

dynamics = [midpoint_implicit for t = 1:N-2]

function cost_function(x,u,q1ref,q2ref,uref)
    q1  = x[1:6]
    q2₋ = x[7:12]
    1*dot(q1,Q .* q1ref) + 1*dot(q2₋, Q .* q2ref) + .01*dot(u-uref,u-uref)
end
function term_cost_function(x,q1ref,q2ref)
    q1  = x[1:6]
    q2₋ = x[7:12]
    1*dot(q1,Q .* q1ref) + 1*dot(q2₋, Q .* q2ref)
end
@show cost_function(randn(19),randn(2),qsref[3],qsref[4],usref[3])

objective = [[(x,u) -> cost_function(x,u,qsref[i],qsref[i+1],usref[i]) for i = 1:N-2]..., (x,u) -> term_cost_function(x,qsref[N-1],qsref[N])]

function ineq_constraint(x)
    q1  = x[1:6]
    q2₋ = x[7:12]

    λ = x[13:13]
    η = x[14:16]
    y = x[17:20]

    [
        y;
        η;
        d(q1);
        # d(q2₋);
        JC(q1)
        # q1[6];
        # JC(q2₋);
    ]
end
ineq = [(x,u)-> ineq_constraint(x) for t = 1:N-1]

eq = [(x,u)-> (x[1:12] - [q0;q1]), [empty_constraint for t = 1:N-2]...]


solver = Solver(objective, dynamics, num_states, num_actions;equality = eq, nonnegative = ineq,options=Options(
            verbose=true,
            constraint_tensor=true,
            update_factorization=false,
    ))
# solver = Solver(objective,dynamics,num_states,num_actions)
x_interpolation = [[qsref[i];qsref[i+1];1*ones(8)] + 0*(ones(20)) for i = 1:N-1]
initialize_states!(solver, x_interpolation)
u_guess = [(usref[i] + 0*randn(2)) for i = 1:N-2]
initialize_controls!(solver, u_guess)
solver.options.max_residual_iterations = 200
solver.options.penalty_initial = 1e0
solver.options.update_factorization = false
solver.options.linear_solver = :LU
solve!(solver)


x_sol, u_sol  = get_trajectory(solver)

Xm = hcat(x_sol...)
Um = hcat(u_sol...)
#
using MATLAB

mat"
figure
hold on
plot($Xm([2,4,6],:)')
legend('m1','m2','mb')
hold off
"
