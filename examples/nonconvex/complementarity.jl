# ## dependencies
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO

# ## problem
objective(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
equality(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
            3*x[1] - x[2] - 3.0 - x[6];
            -x[1] + 0.5*x[2] + 4.0 - x[7];
            -x[1] - x[2] + 7.0 - x[8];
            x[3]*x[6];
            x[4]*x[7];
            x[5]*x[8];]
cone(x) = x #.- 1.0e-5

# ## variables 
num_variables = 8

# ## solver
solver = Solver(objective, equality, cone, num_variables);

# ## initialize
x0 = zeros(num_variables)
initialize!(solver, x0)

# ## solve 
solve!(solver)

# ## solution 
@show solver.solution.variables # x^* = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 6.0]

  