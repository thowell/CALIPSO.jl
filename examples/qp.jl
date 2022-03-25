using Convex, SCS, ECOS
using LinearAlgebra
using Symbolics 

# random QP 
n = 10 
m = 5 

_P = rand(n, n)
P = _P' * _P
p = randn(n)

x0 = randn(n)
A = rand(m, n)
u = A * x0 .+ 1.0
l = A * x0 .- 1.0

# Create a (column vector) variable of size n x 1.
z = Convex.Variable(n + m)

# The problem is to minimize ||Ax - b||^2 subject to x >= 0
# This can be done by: minimize(objective, constraints)
problem = minimize(quadform(z[1:n], P) + p' * z[1:n], [z[n .+ (1:m)] == A * z[1:n], z[n .+ (1:m)] <= u, l <= z[n .+ (1:m)]])

# Solve the problem by calling solve!
solve!(problem, ECOS.Optimizer; silent_solver = false)

# Check the status of the problem
problem.status # :Optimal, :Infeasible, :Unbounded etc.

# Get the optimum value
problem.optval

# Solution 
all(A * z.value[1:n] .<= u)
all(l .<= A * z.value[1:n])
norm(z.value[n .+ (1:m)] - A * z.value[1:n]) < 1.0e-8


f(x) = transpose(x) * P * x + p' * x
h(x) = [
            u - A * x;
            A * x - l;
       ]