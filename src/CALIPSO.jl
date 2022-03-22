module CALIPSO

using LinearAlgebra
using Symbolics
using SparseArrays
using JLD2
using Scratch
using QDLDL
using MathOptInterface 
const MOI = MathOptInterface

# Utilities 
include("utilities.jl")

# Solver
include("solver/lu.jl")
include("solver/ldl.jl")
include("solver/cones.jl")
include("solver/indices.jl")
include("solver/interior_point.jl")
# include("solver/qdldl.jl")

# Trajectory Optimization 
include("trajectory_optimization/costs.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/bounds.jl")
include("trajectory_optimization/general_constraint.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/options.jl")
include("trajectory_optimization/data.jl")
include("trajectory_optimization/solver.jl")
include("trajectory_optimization/moi.jl")
include("trajectory_optimization/utilities.jl")

# objective 
export Cost

# constraints 
export Bound, Bounds, Constraint, Constraints, GeneralConstraint

# dynamics 
export Dynamics

# solver 
export Solver, Options, initialize_states!, initialize_controls!, solve!, get_trajectory

# utils 
export linear_interpolation

end # module