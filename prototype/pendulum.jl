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
dt = Dynamics(
    midpoint_implicit, 
    num_state, 
    num_state, 
    num_action, 
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1] 

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

u_min = -15.0
u_max = 15.0
ineqt = Constraint((x, u, w) -> [u_max * ones(num_action) - u; u - u_min * ones(num_action)], num_state, num_action)
ineqT = Constraint()
ineq = [[ineqt for t = 1:T-1]..., ineqT]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq)
trajopt.dimensions.cone == (T-1) * num_action
# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

methods = ProblemMethods(trajopt)

f = methods.objective 
function fx(x) 
    grad = zeros(trajopt.dimensions.total_variables)
    methods.objective_gradient(grad, x) 
    return grad 
end
function fxx(x) 
    hess = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)
    methods.objective_jacobian_variables_variables(hess, x) 
    return hess
end
function g(x) 
    con = zeros(trajopt.dimensions.equality)
    methods.equality(con, x) 
    return con 
end
function gx(x) 
    jac = zeros(trajopt.dimensions.equality, trajopt.dimensions.total_variables)
    methods.equality_jacobian_variables(jac, x) 
    return jac
end
function gyxx(x, y) 
    hess = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)
    methods.equality_dual_jacobian_variables_variables(hess, x, y) 
    return hess
end

function h(x) 
    con = zeros(trajopt.dimensions.cone)
    methods.inequality(con, x) 
    return con 
end
function hx(x) 
    jac = zeros(trajopt.dimensions.cone, trajopt.dimensions.total_variables)
    methods.inequality_jacobian(jac, x) 
    return jac
end
function hyxx(x, y) 
    hess = zeros(trajopt.dimensions.total_variables, trajopt.dimensions.total_variables)
    methods.inequality_dual_jacobian_variables_variables(hess, x, y) 
    return hess
end

x̄ = rand(trajopt.dimensions.total_variables)
ȳ = rand(trajopt.dimensions.equality)
z̄ = rand(trajopt.dimensions.cone)
f(x̄)
fx(x̄)
fxx(x̄)
g(x̄)
gx(x̄)
gyxx(x̄, ȳ)
h(x̄)
hx(x̄)
hyxx(x̄, ȳ)

# initialize
x = zeros(trajopt.dimensions.total_variables)
 
for (t, idx) in enumerate(trajopt.indices.states)
    x[idx] = x_interpolation[t]
end

for (t, idx) in enumerate(trajopt.indices.actions)
    x[idx] = u_guess[t]
end

