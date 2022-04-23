using BenchmarkTools 
using InteractiveUtils

num_variables = 3
num_parameters = 0
num_equality = 2
num_cone = 2
x0 = [-2.0, 3.0, 1.0]

obj(x, θ) = x[1]
eq(x, θ) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
cone(x, θ) = x[2:3]

# solver
method = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
solver = Solver(method, num_variables, num_parameters, num_equality, num_cone)
initialize!(solver, x0)

problem = solver.problem 
method = solver.methods 
idx = solver.indices 
solution = solver.solution
parameters = solver.parameters

@code_warntype problem!(
        problem, 
        method, 
        idx, 
        solution, 
        parameters)

@benchmark problem!(
        $problem, 
        $method, 
        $idx, 
        $solution, 
        $parameters,
        objective=true,
        objective_gradient_variables=true,
        objective_gradient_parameters=true,
        objective_jacobian_variables_variables=true,
        objective_jacobian_variables_parameters=true,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        equality_dual=true,
        equality_dual_jacobian_variables=true,
        equality_dual_jacobian_variables_variables=true,
        equality_dual_jacobian_variables_parameters=true,
        cone_constraint=true,
        cone_jacobian_variables=true,
        cone_jacobian_parameters=true,
        cone_dual=true,
        cone_dual_jacobian_variables=true,
        cone_dual_jacobian_variables_variables=true,
        cone_dual_jacobian_variables_parameters=true,
    )

# solve 
@benchmark solve!($solver)

@code_warntype initialize_slacks!(solver)
solve!(solver)
