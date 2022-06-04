using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

"""
    minimize   v' b
    subject to ||b|| <= μn
"""

num_variables = 3 
num_equality = 1 
num_cone = 3
idx_ineq = Int[]
idx_soc = [collect(1:3)]

v = [0.0; 0.0; 0.0] 
μ = 0.0 
γ = 0.0 

obj(x, θ) = transpose(v) * x
eq(x, θ) = [x[1] - μ * γ]
cone(x, θ) = x

# solver
methods = ProblemMethods(n, obj, eq, cone)
solver = Solver(methods, n, m, p;
    nonnegative_indices=idx_ineq,
    second_order_indices=idx_soc,)

x0 = ones(n)
initialize!(solver, x0)

# solve 
solve!(solver)
