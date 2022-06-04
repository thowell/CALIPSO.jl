# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO

# ## problem
objective(x) = x[1]
equality(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
cone(x) = x[2:3]

# ## variables 
num_variables = 3

# ## solver
solver = Solver(objective, equality, cone, num_variables);

# ## initialize
x0 = [-2.0, 3.0, 1.0]
initialize!(solver, x0)

# ## solve 
solve!(solver)

# ## solution 
@show solver.solution.variables # x* = [1.0, 0.0, 0.5]

    