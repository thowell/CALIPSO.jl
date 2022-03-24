# DirectTrajectoryOptimization.jl environment 

# ## tested w/ pendulum example

# ## assemble full system 

# dimensions
nz = solver.nlp.num_variables
ny = solver.nlp.num_constraint
nw = nz + ny
nj = solver.nlp.num_jacobian
nh = solver.nlp.num_hessian_lagrangian
nθ = 0

# variables
w = zeros(nw)
θ = zeros(nθ)

# states
for (t, idx) in enumerate(solver.nlp.indices.states)
    w[idx] = solver.nlp.trajopt.states[t]
end

# actions
for (t, idx) in enumerate(solver.nlp.indices.actions)
    w[idx] = solver.nlp.trajopt.actions[t]
end

# duals
w[nz .+ (1:ny)] .= 0.0

z = w[1:nz]
y = w[nz .+ (1:ny)]

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

rank(Array(C))

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

function r_func(r, w, θ, κ)
    r .= 0.0 

    # data
    obj_grad = zeros(nz)
    con = zeros(ny)
    con_jac = zeros(nj) 
    hess_lag = zeros(nh)
    C = spzeros(ny, nz) 

    z = w[1:nz] 
    y = w[nz .+ (1:ny)]

    # objective 
    # MOI.eval_objective(solver.nlp, z)
    MOI.eval_objective_gradient(solver.nlp, obj_grad, z)

    # constraints 
    MOI.eval_constraint(solver.nlp, con, z)
    MOI.eval_constraint_jacobian(solver.nlp, con_jac, z)
    solver.nlp.hessian_lagrangian && MOI.eval_hessian_lagrangian(solver.nlp, hess_lag, z, 1.0, y)

    # assemble
    # C 
    for (i, idx) in enumerate(solver.nlp.jacobian_sparsity)
        C[idx...] = con_jac[i]
    end

    # ∇L 
    for i = 1:nz
        # objective gradient
        r[i] += obj_grad[i] 

        # C' y 
        cy = 0.0 
        for j = 1:ny 
            # cy += C[j, i] * y[j] 
            if (j, i) in solver.nlp.jacobian_sparsity
                cy += C[j, i] * y[j]
            end
        end
        r[i] += cy
    end

    # c 
    for j = 1:ny 
        r[nz + j] += con[j]
    end

    return nothing
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

function rw_func(rw, z, θ) 
    rw .= 0.0
    # data
    obj_grad = zeros(nz)
    con = zeros(ny)
    con_jac = zeros(nj) 
    hess_lag = zeros(nh)

    z = w[1:nz] 
    y = w[nz .+ (1:ny)]

    # objective 
    # MOI.eval_objective(solver.nlp, z)
    # MOI.eval_objective_gradient(solver.nlp, obj_grad, z)

    # constraints 
    # MOI.eval_constraint(solver.nlp, con, z)
    MOI.eval_constraint_jacobian(solver.nlp, con_jac, z)
    solver.nlp.hessian_lagrangian && MOI.eval_hessian_lagrangian(solver.nlp, hess_lag, z, 1.0, y)

    # assemble
    for (i, idx) in enumerate(solver.nlp.hessian_lagrangian_sparsity)
        rw[idx...] = hess_lag[i] 
    end
    
    for (i, idx) in enumerate(solver.nlp.jacobian_sparsity)
        # C 
        rw[nz + idx[1], idx[2]] = con_jac[i] 
        # C' 
        rw[idx[2], nz + idx[1]] = con_jac[i]
    end
    
    # regularization 
    primal_reg = 1.0e-5 
    for i = 1:nz 
        rw[i, i] += primal_reg 
    end
    
    dual_reg = 1.0e-5
    for j = 1:ny 
        rw[nz + j, nz + j] -= dual_reg 
    end

    return nothing
end

H_dense = Array(H)
eigen(H_dense)
cond(H_dense)
rank(H_dense)

r = zeros(nw)
rw = spzeros(nw, nw)

r_func(r, w, zeros(nθ), [1.0])
norm(r - h)
rw_func(rw, w, zeros(nθ))
norm(rw - H)

# using QDLDL

F = CALIPSO.qdldl(H)
sol = zeros(nz + ny)
sol = copy(h)
CALIPSO.solve!(F, sol)
norm(sol - (H \ h), Inf)
I, J, V = findnz(H)

update_A!(F, H)


## solver 
idx = IndicesOptimization(
    nw, nw,
    [collect(1:0), collect(1:0)], [collect(1:0), collect(1:0)],
    Vector{Vector{Vector{Int}}}(), Vector{Vector{Vector{Int}}}(),
    collect(1:nw),
    collect(1:0),
    Vector{Int}(),
    Vector{Vector{Int}}(),
    collect(1:0),
)

ip = interior_point(w, θ;
    s = Euclidean(length(w)),
    idx = idx,
    r! = r_func, 
    rz! = rw_func, 
    # rθ! = rθ_func,
    r  = zeros(idx.nΔ),
    rz = zeros(idx.nΔ, idx.nΔ),
    rθ = zeros(idx.nΔ, nθ),
    opts = InteriorPointOptions(
            undercut=10.0,
            max_iter=500,
            max_ls=50,
            γ_reg=0.0,
            r_tol=1e-5,
            κ_tol=1e-5,  
            ϵ_min=0.0,
            solver=:ldl_solver,
            diff_sol=false,
            verbose=true))

interior_point_solve!(ip)

x_sol = [ip.z[idx] for idx in solver.nlp.indices.states]
u_sol = [ip.z[idx] for idx in solver.nlp.indices.actions]
x_sol[end]