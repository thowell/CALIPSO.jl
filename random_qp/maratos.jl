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
num_variables = 2
num_equality = 1
num_inequality = 0

obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
eq(x) = [x[1]^2 + x[2]^2 - 1.0]
ineq(x) = zeros(0)

# solver
methods = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(methods, num_variables, num_equality, num_inequality,
    options=Options(penalty_initial=1.0))

# initialize 
θ = 1.0
x = [cos(θ); sin(θ)]
x = [1.0; 0.0]
initialize!(solver, x)

# solve 
solve!(solver)
solver.variables[1:2]