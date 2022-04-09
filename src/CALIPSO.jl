module CALIPSO

using LinearAlgebra
using Symbolics
using SparseArrays
using Scratch
using QDLDL

# Solver
include("solver/generate_gradients.jl")
include("solver/indices.jl")
include("solver/methods.jl")
include("solver/problem_data.jl")
include("solver/solver_data.jl")
include("solver/cones/nonnegative.jl")
include("solver/cones/second_order.jl")
include("solver/cones/cone.jl")
include("solver/dimensions.jl") 
include("solver/inertia.jl")
include("solver/qdldl.jl")
include("solver/options.jl")
include("solver/residual.jl")
include("solver/residual_matrix.jl")
include("solver/search_direction.jl")
include("solver/solver.jl")
include("solver/merit.jl")
include("solver/constraint_violation.jl")
include("solver/initialize.jl")
include("solver/solve.jl")
include("solver/iterative_refinement.jl")

export 
    ProblemMethods, Solver, solve!, initialize!, Options

# Trajectory Optimization 
include("trajectory_optimization/costs.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/data.jl")
include("trajectory_optimization/indices.jl")
include("trajectory_optimization/problem.jl")
include("trajectory_optimization/evaluate.jl")
include("trajectory_optimization/utilities.jl")

# Interface 
include("trajectory_optimization/methods.jl")

# objective 
export Cost

# constraints 
export Bound, Bounds, Constraint, Constraints

# dynamics 
export Dynamics

# solver 
export initialize_states!, initialize_controls!, get_trajectory

# utils 
export linear_interpolation

end # module