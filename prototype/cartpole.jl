# ## horizon 
T = 26

# ## cartpole 
num_state = 4 
num_action = 1 
num_parameter = 0 

function cartpole(x, u, w)
    mc = 1.0 
    mp = 0.2 
    l = 0.5 
    g = 9.81 

    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    Hinv = 1.0 / (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) * [H[2, 2] -H[1, 2]; -H[2, 1] H[1, 1]]
    
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp*g*l*s]
    B = [1, 0]

    qdd = -Hinv * (C*qd + G - B*u[1])


    return [qd; qdd]
end

function midpoint_explicit(x, u, w)
    h = 0.1 # timestep 
    x + h * cartpole(x + 0.5 * h * cartpole(x, u, w), u, w)
end

function midpoint_implicit(y, x, u, w)
    y - midpoint_explicit(x, u, w)
end

# ## model
dt = Dynamics(midpoint_implicit, num_state, num_state, num_action)
dyn = [dt for t = 1:T-1] 

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0] 
xT = [0.0; π; 0.0; 0.0] 

# ## objective 
Q = 1.0 
R = 0.1 
Qf = 10.0

ot = (x, u, w) -> 0.5 * Q * dot(x - xT, x - xT) + 0.5 * R * dot(u, u)
oT = (x, u, w) -> 0.5 * Qf * dot(x - xT, x - xT)
ct = Cost(ot, num_state, num_action)
cT = Cost(oT, num_state, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
eq = [
            Constraint((x, u, w) -> x - x1, num_state, num_action,
                evaluate_hessian=true), 
            [Constraint() for t = 2:T-1]..., 
            Constraint((x, u, w) -> x - xT, num_state, 0,
                evaluate_hessian=true)
    ]

u_min = -10.0
u_max = 10.0
ineqt = Constraint((x, u, w) -> [u_max * ones(num_action) - u; u - u_min * ones(num_action)], num_state, num_action)
ineqT = Constraint()
ineq = [[ineqt for t = 1:T-1]..., ineqT]

# ## initialize
u_guess = [0.1 * randn(num_action) for t = 1:T-1]

# x_rollout = [x1] 
# for t = 1:T-1 
#     push!(x_rollout, midpoint_explicit(x_rollout[end], u_guess[t], zeros(num_parameter)))
# end

x_interpolation = linear_interpolation(x1, xT, T)

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq)
methods = ProblemMethods(trajopt)

f = methods.objective 
function fx(x) 
    grad = zeros(trajopt.num_variables)
    methods.objective_gradient(grad, x) 
    return grad 
end
function fxx(x) 
    hess = zeros(trajopt.num_variables, trajopt.num_variables)
    methods.objective_hessian(hess, x) 
    return hess
end
function g(x) 
    con = zeros(trajopt.num_equality)
    methods.equality(con, x) 
    return con 
end
function gx(x) 
    jac = zeros(trajopt.num_equality, trajopt.num_variables)
    methods.equality_jacobian(jac, x) 
    return jac
end
function gyxx(x, y) 
    hess = zeros(trajopt.num_variables, trajopt.num_variables)
    methods.equality_hessian(hess, x, y) 
    return hess
end

function h(x) 
    con = zeros(trajopt.num_cone)
    methods.inequality(con, x) 
    return con 
end
function hx(x) 
    jac = zeros(trajopt.num_cone, trajopt.num_variables)
    methods.inequality_jacobian(jac, x) 
    return jac
end
function hyxx(x, y) 
    hess = zeros(trajopt.num_variables, trajopt.num_variables)
    methods.inequality_hessian(hess, x, y) 
    return hess
end

x̄ = rand(trajopt.num_variables)
ȳ = rand(trajopt.num_equality)
z̄ = rand(trajopt.num_cone)
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
x = zeros(trajopt.num_variables)
 
for (t, idx) in enumerate(trajopt.indices.states)
    x[idx] = x_interpolation[t]
end

for (t, idx) in enumerate(trajopt.indices.actions)
    x[idx] = u_guess[t]
end

