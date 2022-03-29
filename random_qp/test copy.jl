using Pkg 
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using SparseArrays
using BenchmarkTools
using QDLDL
using Test

include("generate.jl")
include("indices.jl")
include("data.jl")
include("cones.jl")
include("dimensions.jl") 
include(joinpath("linear_solvers", "abstract.jl"))
include(joinpath("linear_solvers", "inertia.jl"))
include(joinpath(@__DIR__, "..", "src/solver/ldl.jl"))
include("problem.jl")
include("options.jl")
include("solver.jl")
include("initialize.jl")
include("solve.jl")
include("iterative_refinement.jl")

# test0
# num_variables = 50
# num_equality = 30
# num_inequality = 3

# x0 = ones(num_variables)

# obj(x) = transpose(x) * x
# eq(x) = x[1:30].^2 .- 1.2
# ineq(x) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

# test6 
# num_variables = 2
# num_equality = 0
# num_inequality = 2

# obj(x) = -x[1]*x[2] + 2/(3*sqrt(3))
# eq(x) = zeros(0)
# ineq(x) = [-x[1] - x[2]^2 + 1.0;
#              x[1] + x[2]]

# test7 
# num_variables = 2
# num_equality = 0
# num_inequality = 2

# obj(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
# eq(x) = zeros(0)
# ineq(x) = [-(x[1] -1)^3 + x[2] - 1;
#             -x[1] - x[2] + 2]

# test8
# num_variables = 3
# num_equality = 0
# num_inequality = 1

# obj(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
# eq(x) = zeros(0)
# ineq(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

# knitro
# num_variables = 8
# num_equality = 7
# num_inequality = 8

# obj(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
# eq(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
#             3*x[1] - x[2] - 3.0 - x[6];
#             -x[1] + 0.5*x[2] + 4.0 - x[7];
#             -x[1] - x[2] + 7.0 - x[8];
#             x[3]*x[6];
#             x[4]*x[7];
#             x[5]*x[8];]
# ineq(x) = x

# maratos
# num_variables = 2
# num_equality = 1
# num_inequality = 0

# obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
# eq(x) = [x[1]^2 + x[2]^2 - 1.0]
# ineq(x) = zeros(0)

# wachter
num_variables = 3
num_equality = 2
num_inequality = 2

obj(x) = x[1]
eq(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
ineq(x) = x[2:3]


# solver
m = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(m, num_variables, num_equality, num_inequality)

solver.variables .= 1.0
solver.penalty .= 1.0 
solver.dual .= 1.0 
solver.central_path .= 1.0
problem!(solver.problem, solver.methods, solver.indices, solver.variables;
    gradient=true,
    constraint=true,
    jacobian=true,
    hessian=true)

residual!(solver.data, solver.problem, solver.indices, solver.variables, solver.central_path, solver.penalty, solver.dual)
matrix!(solver.data, solver.problem, solver.indices, solver.variables, solver.central_path, solver.penalty, solver.dual, 
    solver.primal_regularization, solver.dual_regularization)

norm(solver.data.matrix)
norm(solver.data.residual)

norm(solver.data.matrix[1:119, 1:119])
norm(solver.data.residual[1:119])
norm(solver.data.residual[1:50])
norm(solver.data.residual[50 .+ (1:30)])
norm(solver.data.residual[80 .+ (1:3)])
norm(solver.data.residual[83 .+ (1:30)])
norm(solver.data.residual[113 .+ (1:3)])
norm(solver.data.residual[116 .+ (1:3)])



# solve 
solve!(solver)

compute_inertia!(solver.linear_solver)
inertia(solver)

solver.linear_solver.inertia
solver.dimensions.variables
solver.dimensions.primal 
solver.dimensions.total
solver.dimensions.dual
