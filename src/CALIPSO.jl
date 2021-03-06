module CALIPSO

using LinearAlgebra
using Symbolics
using SparseArrays
using Scratch
# using QDLDL
using AMD
using ILUZero

using Printf
using Crayons

# Solver
include("solver/codegen.jl")
include("solver/indices.jl")
include("solver/methods.jl")
include("solver/dimensions.jl")
include("solver/point.jl")
include("solver/filter.jl")
include("solver/problem_data.jl")
include("solver/evaluate.jl")
include("solver/solver_data.jl")
include("solver/cones/nonnegative.jl")
include("solver/cones/second_order.jl")
include("solver/cones/methods.jl")
include("solver/cones/cone.jl")
include("solver/cones/codegen.jl")
include("solver/inertia.jl")
include("solver/qdldl.jl")
include("solver/linear_solver.jl")
include("solver/options.jl")
include("solver/print.jl")
include("solver/residual.jl")
include("solver/residual_jacobian_variables.jl")
include("solver/residual_jacobian_parameters.jl")
include("solver/search_direction.jl")
include("solver/line_search.jl")
include("solver/optimality_error.jl")
include("solver/solver.jl")
include("solver/merit.jl")
include("solver/constraint_violation.jl")
include("solver/initialize.jl")
include("solver/differentiate.jl")
include("solver/solve.jl")
include("solver/iterative_refinement.jl")

export
    Solver, solve!, initialize!, Options,
    empty_constraint, callback_inner, callback_outer

# Trajectory Optimization
include("trajectory_optimization/costs.jl")
include("trajectory_optimization/constraints.jl")
include("trajectory_optimization/constraints_vector.jl")
include("trajectory_optimization/equality_general.jl")
include("trajectory_optimization/dynamics.jl")
include("trajectory_optimization/data.jl")
include("trajectory_optimization/indices.jl")
include("trajectory_optimization/sparsity.jl")
include("trajectory_optimization/dimensions.jl")
include("trajectory_optimization/problem.jl")
include("trajectory_optimization/evaluate.jl")
include("trajectory_optimization/solver.jl")
include("trajectory_optimization/utilities.jl")

# Interface
include("trajectory_optimization/methods.jl")

# objective
export Cost

# constraints
export Constraint, Constraints

# dynamics
export Dynamics

# solver
export initialize_states!, initialize_actions!, get_trajectory

# utils
export linear_interpolation

end # module
