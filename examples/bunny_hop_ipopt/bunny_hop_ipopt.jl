cd("/Users/kevintracy/devel/TrustRegionSQP/PDAL")
Pkg.activate(".")
using LinearAlgebra, ForwardDiff
using StaticArrays
const FD = ForwardDiff
using Printf
using MATLAB
using Ipopt
using MathOptInterface
const MOI = MathOptInterface;
using Random

# bunny hop dynamics
const m_b = 10.0

const m1 = 1.0
const m2 = 1.0
const h = 0.2
const gravity = [0;-9.8]

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
    dot(r1-r2,r1-r2) - r_wheel_base^2
end
function Dc(q)
    r1 = q[1:2]
    r2 = q[3:4]
    t0 = (r1-r2)
    g = 2*[t0;-t0;0;0]
    Matrix(transpose(g))
    # transpose(g)
end
function d(q)
    [q[2];q[4];q[6]-0.3]
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

function contact_kkt(q1,q2,q3,λ,η,u1,u2,κ)
    [#  DEL               LINK          CONTACT
        DEL(q1,q2,q3,λ,u1,u2) + h*transpose(Dc(q2))*λ + h*transpose(Dd(q2))*η; # 6
        c(q3);            # LINK CONSTRAINT     # 1
        η .* d(q3) .- κ;  # CONTACT CONSTRAINT  # 3
    ]
end

# problem indexing
N = 10



bidx_q = [(i-1)*6 .+ (1:6) for i = 1:N]
bidx_u = [(bidx_q[end][end] + (i-1)*2) .+ (1:2) for i = 1:N-1]
bidx_λ = [(bidx_u[end][end] + (i-1)*1) .+ (1:1) for i = 1:N-2]
bidx_η = [(bidx_λ[end][end] + (i-1)*3) .+ (1:3) for i = 1:N-2]
nz = bidx_η[end][end]

bidx_c = [(i-1)*10 .+ (1:10) for i = 1:N-2] # dynamics constraints
push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q1 constraint
push!(bidx_c, bidx_c[end][end] .+ (1:6)) # q2 constraint
push!(bidx_c, bidx_c[end][end] .+ (1:2)) # jump_constraint constraint
# push!(bidx_c, bidx_c[end][end] .+ (1:1)) # jump_constraint constraint
nc = bidx_c[end][end]

bidx_h = [(i-1)*3 .+ (1:3) for i = 1:N-2] # here is η > 0
bidx_h2 = [(bidx_h[end][end] + (i-1)*3) .+ (1:3) for i = 1:N] # Jc(q) >0
# bidx_h3 = bidx_h2[end][end] .+ (1:N) # Jc(q) >0
nh = bidx_h2[end][end]

q0 = [
        -r_wheel_base/2,
        0,
        r_wheel_base/2,
        0,
        0,
        sqrt(1.5^2-(r_wheel_base/2)^2)
] #- [5,0,5,0,5,0]

q1 = 1*q0 + 11*h*[1,0,1,0,1,0]


usref = [ -0.95*sqrt(2)*.5*m_b*9.8*ones(2) for i = 1:N-1]
qsref = [q0 + 11*h*(i-1)*[1,0,1,0,1,0] for i = 1:N]


Q1 = Diagonal([0, 1, 0, 1, 0, .1])
Q2 = Diagonal([0, 1, 0, 1, 0, 1e4])
R = .01*Diagonal(ones(2))



function equality_constraint(z)
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

    c[bidx_c[N-1]] = z[bidx_q[1]] - q0
    c[bidx_c[N]] = z[bidx_q[2]] - q1
    c[bidx_c[N+1]] = z[bidx_q[5]][[2,4]] - [1.1;1.15]
    # c[bidx_c[N+1]] = [z[bidx_q[6][2]];z[bidx_q[5][4]]] - [.7;.7]
    # c[bidx_c[N+2]] = z[bidx_q[N]][[6]] - [1.0]
    return c
end


function inequality_constraints(z)
    ineq = zeros(eltype(z),nh)
    for i = 1:N-2
        ineq[bidx_h[i]] = z[bidx_η[i]] # 4
    end
    for i = 1:N
        qq = z[bidx_q[i]]
        ineq[bidx_h2[i]] = d(qq)  # 4

        # ineq[bidx_h3[i]] = qq[3] - qq[1]
    end

    # qn = z[bidx_q[N]]
    # ineq[bidx_h3] = [qn[5] - qn[1]; -qn[5] + qn[3]]
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
        du = z[bidx_u[i]] - usref[i]
        J += .5*transpose(du)*R*du
        qq1 = z[bidx_q[i]]
        qq2 = z[bidx_q[i+1]]
        J += .005*transpose(qq1-qq2)*(qq1-qq2)
    end
    for i = 1:N-2
        qq1 = z[bidx_q[i]][[1,3,5]]
        qq2 = z[bidx_q[i+1]][[1,3,5]]
        qq3 = z[bidx_q[i+2]][[1,3,5]]
        a = (qq3 - 2*qq2 + qq1)
        J += 10*transpose(a)*a
    end
    return J
end


initial_z = 10*ones(nz)
for i = 1:N
    initial_z[bidx_q[i]] .= qsref[i] + .001*abs.(randn(6)) #+ .5*[0,1,0,1,0,1]
end
for i = 1:N-1
    initial_z[bidx_u[i]] .= usref[i] + .001*randn(2)
end
for i = 1:N-2
    # initial_z[bidx_y[i]] .= .001
    initial_z[bidx_η[i]] = 9.8*[m1;m2;m_b]
end
#
# using JLD2
# using FileIO
#
# @load "/Users/kevintracy/devel/TrustRegionSQP/bunny_hop/z0.jld2"
# #
# # z0 = randn(nz)
# @show norm(equality_constraint(z0))
# @show minimum(inequality_constraints(z0))



using Random
Random.seed!(1234)
initial_z += .001*randn(nz)



function con!(cv,z)
    cv[1:nc] .= equality_constraint(z)
    cv[(nc + 1):(nc + nh)] .= inequality_constraints(z)
end
struct ProblemMOI <: MOI.AbstractNLPEvaluator
    n_nlp::Int
    m_nlp::Int
    idx_ineq
    obj_grad::Bool
    con_jac::Bool
    sparsity_jac
    sparsity_hess
    primal_bounds
    constraint_bounds
    hessian_lagrangian::Bool
end

function ProblemMOI(n_nlp,m_nlp;
        idx_ineq=(1:0),
        obj_grad=true,
        con_jac=true,
        sparsity_jac=sparsity_jacobian(n_nlp,m_nlp),
        sparsity_hess=sparsity_hessian(n_nlp,m_nlp),
        primal_bounds=primal_bounds(n_nlp),
        constraint_bounds=constraint_bounds(m_nlp,idx_ineq=idx_ineq),
        hessian_lagrangian=false)

    ProblemMOI(n_nlp,m_nlp,
        idx_ineq,
        obj_grad,
        con_jac,
        sparsity_jac,
        sparsity_hess,
        primal_bounds,
        constraint_bounds,
        hessian_lagrangian)
end

function primal_bounds(n)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    return x_l, x_u
end

function constraint_bounds(m; idx_ineq=((nc + 1):(nc + nh)))
    c_l = zeros(m)
    c_l[(nc + 1):(nc + nh)] .= 0

    c_u = zeros(m)
    c_u[(nc + 1):(nc + nh)] .= Inf
    return c_l, c_u
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function sparsity_jacobian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function sparsity_hessian(n,m)

    row = []
    col = []

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    cost_function(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ForwardDiff.gradient!(grad_f,cost_function,x)
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    con!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    ForwardDiff.jacobian!(reshape(jac,prob.m_nlp,prob.n_nlp), con!, zeros(prob.m_nlp), x)
    return nothing
end

function MOI.features_available(prob::MOI.AbstractNLPEvaluator)
    return [:Grad, :Jac]
end

MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.sparsity_jac

function solve(x0,prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-6,c_tol=1.0e-6,max_iter=10000)
    x_l, x_u = prob.primal_bounds
    c_l, c_u = prob.constraint_bounds

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res
end

function solve_ipopt()


    n_nlp = nz
    m_nlp = nc + nh
    prob = ProblemMOI(n_nlp,m_nlp)
    z = solve(initial_z,prob)
    return z
end


zopt = solve_ipopt()

qs = [zopt[bidx_q[i]] for i = 1:N]
us = [zopt[bidx_u[i]] for i = 1:N-1]

Qm = hcat(qs...)
Um = hcat(us...)
Qreffm = hcat(qsref...)
#

using MATLAB
mat"
figure
hold on
plot($Qm([2,4,6],:)')
legend('m1','m2','mb')
hold off
"

mat"
figure
hold on
plot($Um')
hold off
"

mat"
figure
hold on
plot($Qreffm([2,4,6],:)')
legend('m1','m2','mb')
hold off
"


mat"
figure
hold on
plot($Qm([1,3,5],:)')
legend('m1','m2','mb')
hold off
"
