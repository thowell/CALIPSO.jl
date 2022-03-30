module CALIPSO

using LinearAlgebra
using Symbolics
using SparseArrays
using Scratch
using QDLDL
using MathOptInterface 
const MOI = MathOptInterface

# Solver
include("solver/indices.jl")
include("solver/problem_data.jl")
include("solver/solver_data.jl")
include("solver/cones.jl")
include("solver/dimensions.jl") 
include("solver/inertia.jl")
include("solver/qdldl.jl")
include("solver/options.jl")
include("solver/residual.jl")
include("solver/residual_matrix.jl")
include("solver/search_direction.jl")
include("solver/solver.jl")
include("solver/initialize.jl")
include("solver/solve.jl")
include("solver/iterative_refinement.jl")
include("solver/generate_gradients.jl")

export 
    ProblemMethods, Solver, solve!, initialize!, Options

# Trajectory Optimization 
include("trajectory_optimization/costs.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/bounds.jl")
include("trajectory_optimization/general_constraint.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/data.jl")
include("trajectory_optimization/problem.jl")
include("trajectory_optimization/moi.jl")
include("trajectory_optimization/utilities.jl")

# objective 
export Cost

# constraints 
export Bound, Bounds, Constraint, Constraints, GeneralConstraint

# dynamics 
export Dynamics

# solver 
export initialize_states!, initialize_controls!, get_trajectory

# utils 
export linear_interpolation

end # module