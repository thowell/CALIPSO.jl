# PREAMBLE

# PKG_SETUP

# ## Setup

# ## 
eval_hess_lag = true 

# ## horizon 
T = 11 

# ## acrobot 
num_state = 2
num_action = 1 
num_parameter = 0 

function pendulum(x, u, w)
    mass = 1.0
    length_com = 0.5
    gravity = 9.81
    damping = 0.1

    [
        x[2],
        (u[1] / ((mass * length_com * length_com))
            - gravity * sin(x[1]) / length_com
            - damping * x[2] / (mass * length_com * length_com))
    ]
end

function midpoint_implicit(y, x, u, w)
    h = 0.05 # timestep 
    y - (x + h * pendulum(0.5 * (x + y), u, w))
end

# ## model
dt = Dynamics(
    midpoint_implicit, 
    num_state, 
    num_state, 
    num_action, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
dynamics = [dt for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0] 
xT = [π; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2])
ct = Cost(ot, num_state, num_action, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
cT = Cost(oT, num_state, 0, 
    num_parameter=num_parameter,
    evaluate_hessian=eval_hess_lag)
objective = [[ct for t = 1:T-1]..., cT]

# ## constraints
bnd1 = Bound(num_state, num_action)
bndt = Bound(num_state, num_action)
bndT = Bound(num_state, 0)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

# ## initial 
con1 = Constraint((x, u, w) -> x - x1, num_state, num_action,
    evaluate_hessian=eval_hess_lag)
conT = Constraint((x, u, w) -> x - xT, num_state, num_action,
    evaluate_hessian=eval_hess_lag)
constraints = [con1, [Constraint() for t = 2:T-1]..., conT]

# ## problem 
solver = Solver(dynamics, objective, constraints, bounds, 
    evaluate_hessian=eval_hess_lag,
    options=Options{Float64}())

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

initialize_states!(solver, x_interpolation)
initialize_controls!(solver, u_guess)

# # ## solve
# @time solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)
# @show x_sol[1]
# @show x_sol[T]

# # ## state
# plot(hcat(x_sol...)')

# # ## control
# plot(hcat(u_sol[1:end-1]..., u_sol[end-1])', 
#     linetype=:steppost)

# DirectTrajectoryOptimization.jl environment 

# ## tested w/ pendulum example

# ## assemble full system 

# dimensions
nz = solver.nlp.num_variables
ny = solver.nlp.num_constraint
nj = solver.nlp.num_jacobian
nh = solver.nlp.num_hessian_lagrangian

# variables
# z = rand(nz)
# y = rand(ny)

# data
obj_grad = zeros(nz)
con = zeros(ny)
con_jac = zeros(nj) 
hess_lag = zeros(nh)

# objective 
MOI.eval_objective(solver.nlp, z)
MOI.eval_objective_gradient(solver.nlp, obj_grad, z)

# constraints 
MOI.eval_constraint(solver.nlp, con, z)
MOI.eval_constraint_jacobian(solver.nlp, con_jac, z)
solver.nlp.hessian_lagrangian && MOI.eval_hessian_lagrangian(solver.nlp, hess_lag, z, 1.0, y)

"""
KKT system

H = [
        ∇²L C' 
        C   0
    ]

h = [
        ∇L 
        c
    ]
"""
C = spzeros(ny, nz) 
H = spzeros(nz + ny, nz + ny)
h = zeros(nz + ny) 

# C 
for (i, idx) in enumerate(solver.nlp.jacobian_sparsity)
    C[idx...] = con_jac[i]
end

# ∇L 
for i = 1:nz
    # objective gradient
    h[i] += obj_grad[i] 

    # C' y 
    cy = 0.0 
    for j = 1:ny 
        # cy += C[j, i] * y[j] 
        if (j, i) in solver.nlp.jacobian_sparsity
            cy += C[j, i] * y[j]
        end
    end
    h[i] += cy
end

# c 
for j = 1:ny 
    h[nz + j] += con[j]
end

#TODO: only fill in half?
# ∇²L
for (i, idx) in enumerate(solver.nlp.hessian_lagrangian_sparsity)
    H[idx...] = hess_lag[i] 
end

for (i, idx) in enumerate(solver.nlp.jacobian_sparsity)
    # C 
    H[nz + idx[1], idx[2]] = con_jac[i] 
    # C' 
    H[idx[2], nz + idx[1]] = con_jac[i]
end

# regularization 
primal_reg = 1.0e-5 
for i = 1:nz 
    H[i, i] += primal_reg 
end

dual_reg = 1.0e-5
for j = 1:ny 
    H[nz + j, nz + j] -= dual_reg 
end

H_dense = Array(H)
eigen(H_dense)
cond(H_dense)
rank(H_dense)

using QDLDL

F = qdldl(H)
sol = zeros(nz + ny)
sol = copy(h)
QDLDL.solve!(F, sol)
norm(sol - (H \ h), Inf)

