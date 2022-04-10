using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

"""
    minimize   v' b
    subject to ||b|| <= μn
"""

n = 3 
m = 1 
p_nn = 0
p_soc = [3] 
p = p_nn + sum(p_soc)
idx_ineq = collect(1:p_nn)
idx_soc = [collect(p_nn + sum(p_soc[1:(i-1)]) .+ (1:d)) for (i, d) in enumerate(p_soc)]

v = [0.0; 1.0; 1.0] 
μ = 1.0 
γ = 1.0 

obj(x) = transpose(v) * x #+ 1.0e-5 * dot(x, x)
eq(x) = [x[1] - μ * γ]
cone(x) = x

# solver
methods = ProblemMethods(n, obj, eq, cone)
solver = Solver(methods, n, m, p;
    nonnegative_indices=idx_ineq,
    second_order_indices=idx_soc)

x0 = ones(n)
initialize!(solver, x0)

# solve 
solve!(solver)
@test norm(solver.data.residual, Inf) < 1.0e-5