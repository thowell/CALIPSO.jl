using Test
using ForwardDiff 
using SparseArrays
using Symbolics
using LinearAlgebra
using Random
using BenchmarkTools
using RoboDojo
using CALIPSO

# Solver
include("solver/problem.jl")
include("solver/test1.jl")
include("solver/test2.jl")
include("solver/test3.jl")
include("solver/test4.jl")
include("solver/maratos.jl")
include("solver/knitro.jl")
include("solver/wachter.jl")

# # Trajectory Optimization 
include("trajectory_optimization/objective.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/hessian_lagrangian.jl")

# # Examples
include("examples/pendulum.jl")
include("examples/cartpole.jl")
include("examples/acrobot.jl")
# include("examples/hopper_gait.jl")




