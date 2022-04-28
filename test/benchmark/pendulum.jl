using BenchmarkTools 
using InteractiveUtils

# @testset "Examples: Pendulum" begin 
# ## horizon 
T = 11 

# ## acrobot 
num_state = 2
num_action = 1 
num_parameter = 0 

function pendulum(x, u, w)
    mass = 1.0
    length_com = 0.5
    gravity = 9.81
    damping = 0.1

    [
        x[2],
        (u[1] / ((mass * length_com * length_com))
            - gravity * sin(x[1]) / length_com
            - damping * x[2] / (mass * length_com * length_com))
    ]
end

function midpoint_implicit(y, x, u, w)
    h = 0.05 # timestep 
    y - (x + h * pendulum(0.5 * (x + y), u, w))
end

# ## model
dyn = [
        Dynamics(
            midpoint_implicit, 
            num_state, 
            num_state, 
            num_action, 
            num_parameter=num_parameter) for t = 1:T-1
]

# ## initialization
x1 = [0.0; 0.0] 
xT = [π; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2])
ct = Cost(ot, num_state, num_action)
cT = Cost(oT, num_state, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints 
eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
eq = [eq1, [Constraint() for t = 2:T-1]..., eqT]

ineq = [Constraint() for t = 1:T]

soc = [[Constraint()] for t = 1:T]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

methods = ProblemMethods(trajopt)
idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# ## solver
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    nonnegative_indices=idx_nn, 
    second_order_indices=idx_soc,
    custom=trajopt,
    options=Options(verbose=false))
initialize_states!(solver, trajopt, x_interpolation) 
initialize_controls!(solver, trajopt, u_guess)

problem = solver.problem 
method = solver.methods 
idx = solver.indices 
solution = solver.solution
parameters = solver.parameters

@code_warntype evaluate!(
        problem, 
        method, 
        idx, 
        solution, 
        parameters)

@benchmark evaluate!(
        $problem, 
        $method, 
        $idx, 
        $solution, 
        $parameters,
        objective=true,
        # objective_gradient_variables=true,
        # objective_gradient_parameters=true,
        # objective_jacobian_variables_variables=true,
        # objective_jacobian_variables_parameters=true,
        # equality_constraint=true,
        # equality_jacobian_variables=true,
        # equality_jacobian_parameters=true,
        # equality_dual=true,
        # equality_dual_jacobian_variables=true,
        # equality_dual_jacobian_variables_variables=true,
        # equality_dual_jacobian_variables_parameters=true,
        # cone_constraint=true,
        # cone_jacobian_variables=true,
        # cone_jacobian_parameters=true,
        # cone_dual=true,
        # cone_dual_jacobian_variables=true,
        # cone_dual_jacobian_variables_variables=true,
        # cone_dual_jacobian_variables_parameters=true,
    )

# ## solve 
solve!(solver)
solver.solution.variables

@benchmark solve!($solver)


typeof(obj[1].cost)
obj[1].cost isa Function
c = zeros(1)
cost(c, trajopt.data.objective, trajopt.data.states, trajopt.data.actions, trajopt.data.parameters)
@code_warntype cost(c, trajopt.data.objective, trajopt.data.states, trajopt.data.actions, trajopt.data.parameters)

@benchmark cost($c, $trajopt.data.objective, $trajopt.data.states, $trajopt.data.actions, $trajopt.data.parameters)
trajopt.data.objective[1].cost_cache
typeof(trajopt.data.states[1])
@code_warntype solve!(solver)


@variables x[1:num_state], u[1:num_action], w[1:num_parameter]
    
c = ot(x, u, w)
gz = Symbolics.gradient(c, [x; u])
gw = Symbolics.gradient(c, w)

num_gradient_variables = num_state + num_action
num_gradient_parameters = num_parameter

cost_func = Symbolics.build_function([c], x, u, w, expression=Val{false})[2]
gradient_variables_func = Symbolics.build_function(gz, x, u, w, expression=Val{false})[2]
gradient_parameters_func = Symbolics.build_function(gw, x, u, w, expression=Val{false})[2]

cc = zeros(1)
@code_warntype cost_func(cc, trajopt.data.states[1], trajopt.data.actions[1], trajopt.data.parameters[1])

@benchmark cost_func($cc, $(trajopt.data.states[1]), $(trajopt.data.actions[1]), $(trajopt.data.parameters[1]))
Base.invokelatest(cost_func)(cc, trajopt.data.states[1], trajopt.data.actions[1], trajopt.data.parameters[1])
cost_func
typeof(cost_func)# isa Symbol
# # test solution
# @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

# slack_norm = max(
#                 norm(solver.data.residual.equality_dual, Inf),
#                 norm(solver.data.residual.cone_dual, Inf),
# )
# @test slack_norm < solver.options.slack_tolerance

# @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
# @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# end