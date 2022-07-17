# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using RoboDojo
using LinearAlgebra
include("models/mountain_bike.jl")

# ## problem indexing for NLP
N = 10

# ## primal variables indexing
idx_q = [(i-1)*6 .+ (1:6) for i = 1:N]
idx_u = [(idx_q[end][end] + (i-1)*2) .+ (1:2) for i = 1:N-1]
idx_λ = [(idx_u[end][end] + (i-1)*1) .+ (1:1) for i = 1:N-2]
idx_η = [(idx_λ[end][end] + (i-1)*3) .+ (1:3) for i = 1:N-2]
nz = idx_η[end][end]

# ## equality constraint indexing
idx_c = [(i-1)*10 .+ (1:10) for i = 1:N-2] # dynamics constraints
push!(idx_c, idx_c[end][end] .+ (1:6)) # q1 constraint
push!(idx_c, idx_c[end][end] .+ (1:6)) # q2 constraint
push!(idx_c, idx_c[end][end] .+ (1:2)) # jump_constraint constraint
nc = idx_c[end][end]

# ## inequality constraint indexing
idx_h_η = [(i-1)*3 .+ (1:3) for i = 1:N-2] # here is η > 0
idx_h_d = [(idx_h_η[end][end] + (i-1)*3) .+ (1:3) for i = 1:N] # Jc(q) >0
nh = idx_h_d[end][end]

# ## initial configuration
q0 = [
        -r_wheel_base/2,
        0,
        r_wheel_base/2,
        0,
        0,
        sqrt(1.5^2-(r_wheel_base/2)^2)
]

# ## next configuration
q1 = 1*q0 + 11*h*[1,0,1,0,1,0]

# ## reference control and state histories
usref = [ -0.95*sqrt(2)*.5*m_b*9.8*ones(2) for i = 1:N-1]
qsref = [q0 + 11*h*(i-1)*[1,0,1,0,1,0] for i = 1:N]

# ## cost matrices
Q1 = Diagonal([0, 1, 0, 1, 0, .1])
Q2 = Diagonal([0, 1, 0, 1, 0, 1e4])
R = .01*Diagonal(ones(2))


# ## equality constraints
function equality_constraint(z)
    # dynamics constraints
    c = zeros(eltype(z),nc)
    for i = 1:N-2

        q₋ = z[idx_q[i]]
        qm = z[idx_q[i+1]]
        q₊= z[idx_q[i+2]]
        λ = z[idx_λ[i]]
        η = z[idx_η[i]]
        u1 = z[idx_u[i]]
        u2 = z[idx_u[i+1]]
        c[idx_c[i]] = mountain_bike_dynamics(q₋,qm,q₊,λ,η,u1,u2,0)
    end

    # initial conditions + jump constraint
    c[idx_c[N-1]] = z[idx_q[1]] - q0
    c[idx_c[N]] = z[idx_q[2]] - q1
    c[idx_c[N+1]] = z[idx_q[5]][[2,4]] - [1.1;1.15]
    return c
end

# ## inequality constraints
function inequality_constraints(z)
    # inequality constraints
    ineq = zeros(eltype(z),nh)
    for i = 1:N-2
        ineq[idx_h_η[i]] = z[idx_η[i]] # 4
    end
    for i = 1:N
        qq = z[idx_q[i]]
        ineq[idx_h_d[i]] = d(qq)  # 4
    end
    return ineq
end

# ## cost function
function cost_function(z)
    J = zero(eltype(z))
    for i = 1:N
        dq = z[idx_q[i]] - qsref[i]
        if i > 6
            J += .5*transpose(dq)*Q2*dq
        else
            J += .5*transpose(dq)*Q1*dq
        end

        qq = z[idx_q[i]]
        x1 = qq[1]
        x2 = qq[3]
        xb = qq[5]
        J += 10*(xb - 0.5*(x1 + x2))^2
    end
    for i = 1:N-1
        # cost on velocity
        du = z[idx_u[i]] - usref[i]
        J += .5*transpose(du)*R*du
        qq1 = z[idx_q[i]]
        qq2 = z[idx_q[i+1]]
        J += .005*transpose(qq1-qq2)*(qq1-qq2)
    end
    for i = 1:N-2
        # cost on acceleration
        qq1 = z[idx_q[i]][[1,3,5]]
        qq2 = z[idx_q[i+1]][[1,3,5]]
        qq3 = z[idx_q[i+2]][[1,3,5]]
        a = (qq3 - 2*qq2 + qq1)
        J += 10*transpose(a)*a
    end
    return J
end

# ## initialize
using Random
Random.seed!(1234)

initial_z = 10*ones(nz)
for i = 1:N
    initial_z[idx_q[i]] .= qsref[i] + .001*abs.(randn(6)) #+ .5*[0,1,0,1,0,1]
end
for i = 1:N-1
    initial_z[idx_u[i]] .= usref[i] + .001*randn(2)
end
for i = 1:N-2
    initial_z[idx_η[i]] = 9.8*[m1;m2;m_b]
end

# ## solve
solver = Solver(cost_function, equality_constraint, inequality_constraints, nz; nonnegative_indices = collect(1:nh))
initialize!(solver,initial_z + .001*randn(nz))
solver.options.penalty_initial = 1e2
solve!(solver)

# ## recover solution
qs = [solver.solution.variables[idx_q[i]] for i = 1:N]
us = [solver.solution.variables[idx_u[i]] for i = 1:N-1]

# ## animation 
@load "visuals/bunny_hop.jld2"
include("visuals/bunnyhop.jl")

cd(@__DIR__)


