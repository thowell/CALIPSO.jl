# ## horizon
T = 51 

# ## acrobot 
num_state = 4 
num_action = 1 
num_parameter = 0 

function acrobot(x, u, w)
    mass1 = 1.0  
    inertia1 = 0.33  
    length1 = 1.0 
    lengthcom1 = 0.5 

    mass2 = 1.0  
    inertia2 = 0.33  
    length2 = 1.0 
    lengthcom2 = 0.5 

    gravity = 9.81 
    friction1 = 0.1 
    friction2 = 0.1

    function M(x, w)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

    return [a b; b c]
    end

    function Minv(x, w)
        a = (inertia1 + inertia2 + mass2 * length1 * length1
            + 2.0 * mass2 * length1 * lengthcom2 * cos(x[2]))

        b = inertia2 + mass2 * length1 * lengthcom2 * cos(x[2])

        c = inertia2

        return 1.0 / (a * c - b * b) * [c -b; -b a]
    end

    function τ(x, w)
        a = (-1.0 * mass1 * gravity * lengthcom1 * sin(x[1])
            - mass2 * gravity * (length1 * sin(x[1])
            + lengthcom2 * sin(x[1] + x[2])))

        b = -1.0 * mass2 * gravity * lengthcom2 * sin(x[1] + x[2])

        return [a; b]
    end

    function C(x, w)
        a = -2.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        b = -1.0 * mass2 * length1 * lengthcom2 * sin(x[2]) * x[4]
        c = mass2 * length1 * lengthcom2 * sin(x[2]) * x[3]
        d = 0.0

        return [a b; c d]
    end

    function B(x, w)
        [0.0; 1.0]
    end

    q = view(x, 1:2)
    v = view(x, 3:4)

    qdd = Minv(q, w) * (-1.0 * C(x, w) * v
            + τ(q, w) + B(q, w) * u[1] - [friction1; friction2] .* v)

    return [x[3]; x[4]; qdd[1]; qdd[2]]
end

function midpoint_explicit(x, u, w)
    h = 0.05 # timestep 
    x + h * acrobot(x + 0.5 * h * acrobot(x, u, w), u, w)
end

function midpoint_implicit(y, x, u, w)
    y - midpoint_explicit(x, u, w)
end

# ## model
dt = Dynamics(midpoint_implicit, num_state, num_state, num_action, 
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0] 
xT = [0.0; π; 0.0; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x[3:4], x[3:4])
ct = Cost(ot, num_state, num_action, 
    num_parameter=num_parameter)
cT = Cost(oT, num_state, 0, 
    num_parameter=num_parameter)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
eq = [
        Constraint((x, u, w) -> x - x1, num_state, num_action), 
        [Constraint((x, u, w) -> zeros(0), num_state, num_action) for t = 2:T-1]..., 
        Constraint((x, u, w) -> x - xT, num_state, 0)
    ]

u_min = -1.0
u_max = 5.0
ineqt = Constraint((x, u, w) -> [u_max * ones(num_action) - u; u - u_min * ones(num_action)], num_state, num_action)
ineqT = Constraint()
ineq = [[ineqt for t = 1:T-1]..., ineqT]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq)

# ## initialize
u_guess = [0.1 * rand(num_action) for t = 1:T-1]
# x_rollout = [x1] 
# for t = 1:T-1 
#     push!(x_rollout, midpoint_explicit(x_rollout[end], u_guess[t], zeros(num_parameter)))
# end

x_interpolation = linear_interpolation(x1, xT, T)

# solver
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

# x̄ = rand(trajopt.dimensions.total_variables)
# ȳ = rand(trajopt.dimensions.equality)
# f(x̄)
# fx(x̄)
# fxx(x̄)
# g(x̄)
# gx(x̄)
# gyxx(x̄, ȳ)

# initialize
x = zeros(trajopt.dimensions.total_variables)
 
for (t, idx) in enumerate(trajopt.indices.states)
    x[idx] = x_interpolation[t]
    # x[idx] = x_rollout[t]
end

for (t, idx) in enumerate(trajopt.indices.actions)
    x[idx] = u_guess[t]
end
