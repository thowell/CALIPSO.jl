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

num_variables = 50
num_equality = 30
num_inequality = 0#3

x0 = ones(num_variables)

obj(x) = transpose(x) * x
eq(x) = x[1:30].^2 .- 1.2
ineq(x) = zeros(0)#[x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

# solver
m = ProblemMethods(num_variables, obj, eq, ineq)
solver = Solver(m, num_variables, num_equality, num_inequality)

# solve 
solve!(solver)

compute_inertia!(solver.linear_solver)
inertia(solver)

solver.linear_solver.inertia
solver.dimensions.variables
solver.dimensions.primal 
solver.dimensions.total
solver.dimensions.dual
