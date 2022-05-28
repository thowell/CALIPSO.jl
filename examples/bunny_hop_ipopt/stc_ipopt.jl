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

const stc_a = -0.5
const stc_b = 3.0
const stc_c = 0.3
const stc_d = 3.0

const Fx_min = -10.0#-8.0
const Fx_max = 10.0#8.0
const Fy_min = -10.0#-8.0
const Fy_max = 10.0#8.0
const Fz_min = 0.0#6.0
const Fz_max = 20.0#12.5

function rocket(x, u)
    mass = 1.0
    gravity = -9.81

    p = x[1:3]
    v = x[4:6]

    f = u[1:3]

    [
        v;
        [0.0; 0.0; gravity] + 1.0 / mass * f;
    ]
end

function midpoint_implicit(y, x, u)
    h = 0.05 # timestep
    y - (x + h * rocket(0.5 * (x + y), u))
end

nx = 6 # add 1 for slack
nu = 11
N = 51
idx_x = [((i-1)*(nx + nu) .+ (1:(nx))) for i = 1:N]
idx_u = [((i-1)*(nx + nu) .+ ((nx + 1):(nx+nu))) for i = 1:N-1]
idx_c = [((i-1)*(nx) .+ (1:nx)) for i = 1:N+1] # N-1 dynacon + IC + TC
idx_c2 = [(idx_c[end][end] + (i-1)*6) .+ (1:6) for i = 1:N-1]
nz = (N)*nx + (N-1)*nu
nc = idx_c2[end][end]
nh = 14*(N-1)
x0 = [-5.0;0;5;0;0;0]
xg = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
function cost(z)
    J = zero(eltype(z))
    for i = 1:N
        J += 1.0*norm(z[idx_x[i]][1:3])^2
        J += 0.1*norm(z[idx_x[i]][4:6])^2
    end
    for i = 1:N-1
        J += 0.1*norm(z[idx_u[i]][1:3])^2
    end
    return J
end
function stc_constraint(x,u)
    tx,ty,tz,g1p,g1m,c1p,c1m,g2p,g2m,c2p,c2m = u

    g1 = -x[1] + stc_a # ≧ 0
    c1 = x[3] - stc_b  # ≧ 0

    g2 = x[1] - stc_c
    c2 = x[3] - stc_d

    [g1p - g1m - g1;
     c1p - c1m - c1;
     g1p*c1m;
     g2p - g2m - g2;
     c2p - c2m - c2;
     g2p*c2m]
end
function c(z)
    cv = zeros(eltype(z),nc)
    for i = 1:N-1 # dynamics constraints
        x = z[idx_x[i]]
        u = z[idx_u[i]]
        x2 = z[idx_x[i+1]]
        # @show cv[idx_c[i]]
        # @show midpoint_implicit(x2, x, u)
        cv[idx_c[i]] = midpoint_implicit(x2, x, u)
        cv[idx_c2[i]] = stc_constraint(x,u)
    end
    # @info "done with loop"
    cv[idx_c[N]]   = z[idx_x[1]] - x0  # IC
    cv[idx_c[N+1]] = z[idx_x[N]] - xg  # TC
    return cv
end

function h(z)
    hv = zeros(eltype(z),nh)
    for i = 1:N-1
        idx = 14*(i-1) .+ (1:14)
        # x = z[idx_x[i]]
        u = z[idx_u[i]]
        hv[idx] = [u[1] - Fx_min; Fx_max - u[1];
        u[2] - Fy_min; Fy_max - u[2];
        u[3] - Fz_min; Fz_max - u[3];
        u[4:7];
        u[8:11] ]
    end
    return hv
end
function con!(cv,z)
    cv[1:nc] .= c(z)
    cv[(nc + 1):(nc + nh)] .= h(z)
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
    c_l[(nc + 1):(nc + nh)] .= -Inf

    c_u = zeros(m)
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
    cost(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ForwardDiff.gradient!(grad_f,cost,x)
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

function solve_ipopt(z0)

    # z0 = zeros(nz)
    # Random.seed!(1)
    # for i = 1:N
    #     z0[idx_x[i]] = x0 - 0.1*abs.(randn(nx))
    # end
    n_nlp = nz
    m_nlp = nc + nh
    prob = ProblemMOI(n_nlp,m_nlp)
    z = solve(z0,prob)
    Xm = zeros(nx,N)
    for i = 1:N
        Xm[:,i] = z[idx_x[i]]
    end
    Um = zeros(nu,N)
    for i = 1:N-1
        Um[:,i] = z[idx_u[i]]
    end
    return Xm, Um
end

# initial guess
x_interpolation = LinRange(x0, xg, N)
for i = 1:N
    # h = 0.05
    x_interpolation[i][4:6] = (x0[1:3]-xg[1:3])/(0.05*N)
end
u_guess = [zeros(nu) for t = 1:N-1]
for i = 1:N-1
    u_guess[i][1:3] = [0;0;9.8]
    g1 = -x_interpolation[i][1] + stc_a
    if g1 >= 0
        u_guess[i][4] = g1
        u_guess[i][5] = 0
    else
        u_guess[i][4] = 0
        u_guess[i][5] = -g1
    end
    c1 =  x_interpolation[i][3] - stc_b
    if c1 >= 0
        u_guess[i][6] = c1
        u_guess[i][7] = 0
    else
        u_guess[i][6] = 0
        u_guess[i][7] = -c1
    end

    g2 = x_interpolation[i][1] - stc_c
    if g2 >= 0
        u_guess[i][4+4] = g2
        u_guess[i][5+4] = 0
    else
        u_guess[i][4+4] = 0
        u_guess[i][5+4] = -g2
    end
    c2 =  x_interpolation[i][3] - stc_d
    if c2 >= 0
        u_guess[i][6+4] = c2
        u_guess[i][7+4] = 0
    else
        u_guess[i][6+4] = 0
        u_guess[i][7+4] = -c2
    end
end
init_z = zeros(nz)
for i = 1:N
    init_z[idx_x[i]] = x_interpolation[i]
end
for i = 1:N-1
    init_z[idx_u[i]] = u_guess[i]
end
init_z += 1*randn(nz)

# solve_wachter()
Xm, Um = solve_ipopt(init_z)
#
mat"
figure
title('IPOPT')
hold on
plot($Xm(1,:),$Xm(3,:),'*')
hold off
"

mat"
figure
hold on
a1 = patch([-5 $stc_a-0.1 $stc_a-0.1 -5],[0 0 $stc_b-0.1 $stc_b-0.1],'k');
a2 = patch([$stc_c 5 5 $stc_c],[0 0 $stc_d-0.1 $stc_d-0.1],'k');
a1.FaceAlpha = 0.2;
a2.FaceAlpha = 0.2;
p1 = plot($Xm(1,:),$Xm(3,:),'color','k','linewidth',2)
s = 10;
for i = 1:$N-1
    aa = quiver($Xm(1,i),$Xm(3,i),$Um(1,i)/s,$Um(3,i)/s,'color','#D95319','linewidth',1.5);
end
axis equal
xlim([-5.2,0.7])
ylim([0,5.5])
yticks([0 1 2 3 4 5])
xlabel('X')
ylabel('Y')
legend([a2 p1 aa],'keepout zone','constrained','thrust vector','location','northeast')
hold off
"
