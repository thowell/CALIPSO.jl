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
num_variables = 10 
num_equality = 0 
num_inequality = 5

# methods
objective, equality, inequality, flag = generate_random_qp(num_variables, num_equality, num_inequality);

# solver
methods = ProblemMethods(num_variables, objective, equality, inequality)
solver = Solver(methods, num_variables, num_equality, num_inequality)

# initialize 
x = randn(num_variables)
initialize!(solver, x)

# solve 
solve!(solver)
