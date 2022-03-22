using Test
using Symbolics
using ForwardDiff
using LinearAlgebra
using SparseArrays
using DirectTrajectoryOptimization
const CALIPSO = DirectTrajectoryOptimization

include("objective.jl")
include("dynamics.jl")
include("constraints.jl")
include("hessian_lagrangian.jl")
include("solve.jl")