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
include(joinpath(@__DIR__, "..", "src/solver/ldl.jl"))
include("problem.jl")
include("options.jl")
include("solver.jl")
include("initialize.jl")
include("solve.jl")

# dimensions 
num_variables = 3 
num_equality = 2
num_inequality = 2

obj(x) = x[1] + 1.0e-5 * dot(x, x)
eq(x) = [x[1]^2.0 - x[2] - 1.0; x[1] - x[3] - 0.5] 
ineq(x) = [x[2]; x[3]]

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(methods, num_variables, num_equality, num_inequality,
    options=Options(penalty_initial=1.0e32))

# initialize 
x = [-2.0, 3.0, 1.0]
# x = [1.0, 0.0, 0.5]
initialize!(solver, x)

# solve 
solve!(solver)

solver.variables[1:3]
