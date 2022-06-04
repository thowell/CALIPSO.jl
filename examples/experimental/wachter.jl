# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra

num_variables = 3
num_equality = 2
num_inequality = 2
num_parameters = 0
x0 = [-2.0, 3.0, 1.0]

obj(x, θ) = x[1]
eq(x, θ) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
ineq(x, θ) = x[2:3]

# solver
methods = ProblemMethods(num_variables, num_parameters,  obj, eq, ineq)
solver = Solver(methods, num_variables, num_parameters, num_equality, num_inequality)
initialize!(solver, x0)

# solve 
solve!(solver)

norm(solver.data.residual, Inf) < 1.0e-3
norm(solver.solution.variables[1:3] - [1.0; 0.0; 0.5], Inf) < 1.0e-3