using Test
using FiniteDiff
using SparseArrays
using Symbolics
using LinearAlgebra
using RoboDojo
using CALIPSO

# Solver
include("solver/problem.jl")
include("solver/qp_equality.jl")
include("solver/qp_nonnegative.jl")
include("solver/test1.jl")
include("solver/test2.jl")
include("solver/test3.jl")
include("solver/test4.jl")
include("solver/maratos.jl")
include("solver/knitro.jl")
include("solver/wachter.jl")
include("solver/friction_cone.jl")
include("solver/portfolio.jl")

# Trajectory Optimization
include("trajectory_optimization/objective.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/hessian_lagrangian.jl")

# Examples
include("examples/pendulum.jl")
include("examples/cartpole.jl")
include("examples/acrobot.jl")
include("examples/rocket_landing.jl")
include("examples/double_integrator.jl")
# include("examples/box_move.jl")
# include("examples/ball_in_cup.jl")
# include("examples/cyberdrift.jl")
# include("examples/hopper_gait.jl")

# include("examples/quadruped_drop.jl")
# include("examples/quadruped_gait.jl")
