# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using CALIPSO
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# ## problem 
objective(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
equality(x) = [x[1]^2 + x[2]^2 - 1.0]

# ## variables 
num_variables = 2

# ## solver
solver = Solver(objective, equality, empty_constraint, num_variables);

# ## initialize 
x0 = [2.0; 1.0]
initialize!(solver, x0)

# ## solve 
solve!(solver)

# ## solution 
@show solver.solution.variables # x* = [1.0, 0.0]


