# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra

# ## horizon 
T = 51

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
    h = 0.05 # timestep 
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
Q = 1.0e-2 
R = 1.0e-1 
Qf = 1.0e2 

ot = (x, u, w) -> 0.5 * Q * dot(x - xT, x - xT) + 0.5 * R * dot(u, u)
oT = (x, u, w) -> 0.5 * Qf * dot(x - xT, x - xT)
ct = Cost(ot, num_state, num_action)
cT = Cost(oT, num_state, 0)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
eq = [
            Constraint((x, u, w) -> x - x1, num_state, num_action,
                constraint_tensor=true), 
            [Constraint() for t = 2:T-1]..., 
            Constraint((x, u, w) -> x - xT, num_state, 0,
                constraint_tensor=true)
    ]

ineq = [Constraint() for t = 1:T]

so = [[Constraint()] for t = 1:T]

# ## initialize
u_guess = [0.01 * ones(num_action) for t = 1:T-1]

# x_rollout = [x1] 
# for t = 1:T-1 
#     push!(x_rollout, midpoint_explicit(x_rollout[end], u_guess[t], zeros(num_parameter)))
# end
# initialize_states!(solver, x_rollout)

x_interpolation = linear_interpolation(x1, xT, T)

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so)
methods = ProblemMethods(trajopt)

# solver
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    options=Options(verbose=true))
initialize_states!(solver, trajopt, x_interpolation)
initialize_actions!(solver, trajopt, u_guess)


# solve 
solve!(solver)
norm(solver.data.residual, Inf) < 1.0e-5