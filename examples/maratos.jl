# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra

num_variables = 2
num_parameters = 0
num_equality = 1
num_inequality = 0
x0 = [2.0; 1.0]

obj(x, θ) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
eq(x, θ) = [x[1]^2 + x[2]^2 - 1.0]
ineq(x, θ) = zeros(0)

# solver
methods = ProblemMethods(num_variables, num_parameters, obj, eq, ineq)
solver = Solver(methods, num_variables, num_parameters, num_equality, num_inequality)
initialize!(solver, x0)

# solve 
solve!(solver)
norm(solver.data.residual, Inf) < 1.0e-5