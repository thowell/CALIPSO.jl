using CALIPSO 
using LinearAlgebra 
horizon = 45

# ## cartpole 
num_states = [4 for t = 1:horizon]
num_actions = [1 for t = 1:horizon-1] 
num_parameters = [0 for t = 1:horizon] 

# ## initialization
state_initial = [0.0; 0.0; 0.0; 0.0] 
state_goal = [0.0; π; 0.0; 0.0] 

# ## objective 
Q = 1.0e-2 
R = 1.0e-1 
Qf = 1.0e2 

objective = [
    [(x, u, w) -> 0.5 * Q * dot(x - state_goal, x - state_goal) + 0.5 * R * dot(u, u) for t = 1:horizon-1]..., 
    (x, u, w) -> 0.5 * Qf * dot(x - state_goal, x - state_goal),
]

# ## dynamics
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
    timestep = 0.05 
    x + timestep * cartpole(x + 0.5 * timestep * cartpole(x, u, w), u, w)
end

function midpoint_cartpole(y, x, u, w)
    y - midpoint_explicit(x, u, w)
end

dynamics = [midpoint_cartpole for t = 1:horizon-1] 

# ## constraints
equality = [
        (x, u, w) -> x - state_initial,
        [empty_constraint for t = 2:horizon-1]..., 
        (x, u, w) -> x - state_goal,
    ]

nonnegative = [empty_constraint for t = 1:horizon]
second_order = [[empty_constraint] for t = 1:horizon]

# dimensions
nz = sum(num_states) + sum(num_actions)
np = sum(num_parameters)
T = length(num_states) 
@assert length(num_actions) == T - 1

# indices 
x_idx, u_idx = state_action_indices(num_states, num_actions)
p_idx = parameter_indices(num_parameters) 

# variables

z = Symbolics.variables(:z, 1:nz)# z[nz] p[sum(np)]
p = Symbolics.variables(:p, 1:sum(np))
x = [z[idx] for idx in x_idx]
u = [z[idx] for idx in u_idx]
θ = [p[idx] for idx in p_idx]

# symbolic expressions
o = [objective[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
d = [dynamics[t](x[t+1], x[t], u[t], θ[t]) for t = 1:T-1]
e = [equality[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
nn = [nonnegative[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
s = [[so(x[t], t == T ? zeros(0) : u[t], θ[t]) for so in second_order[t]] for t = 1:T]

# constraint dimensions 
dynamics_dimensions = [length(dt) for dt in d]
equality_dimensions = [length(et) for et in e]
nonnegative_dimensions = [length(nnt) for nnt in nn]
second_order_dimensions = [[length(so) for so in st] for st in s]

# assemble 
os = sum(o)
es = vcat(d..., e...)
cs = vcat(nn..., (s...)...)

## derivatives 

# objective
oz = Symbolics.gradient(os, z)
oθ = Symbolics.gradient(os, θ)
ozz = Symbolics.sparsejacobian(oz, z)
ozθ = Symbolics.sparsejacobian(oz, θ)

# equality
ez = Symbolics.sparsejacobian(es, z)
eθ = Symbolics.sparsejacobian(es, θ)

# cone
cz = Symbolics.sparsejacobian(cs, z)
cθ = Symbolics.sparsejacobian(cs, θ)

## product derivatives
ye = Symbolics.variables(:ye, 1:length(es))
yc = Symbolics.variables(:yc, 1:length(cs))

# equality
ey = (isempty(es) || isempty(ye)) ? 0.0 : dot(es, ye)
eyz = Symbolics.gradient(ey, z)
eyzz = Symbolics.sparsejacobian(eyz, z)
eyzθ = Symbolics.sparsejacobian(eyz, θ)

# cone
cy = (isempty(cs) || isempty(yc)) ? 0.0 : dot(cs, yc)
cyz = Symbolics.gradient(cy, z)
cyzz = Symbolics.sparsejacobian(cyz, z)
cyzθ = Symbolics.sparsejacobian(cyz, θ)

# sparsity
ozz_sparsity = collect(zip([findnz(ozz)[1:2]...]...))
ozθ_sparsity = collect(zip([findnz(ozθ)[1:2]...]...))

ez_sparsity = collect(zip([findnz(ez)[1:2]...]...))
eθ_sparsity = collect(zip([findnz(eθ)[1:2]...]...))

cz_sparsity = collect(zip([findnz(cz)[1:2]...]...))
cθ_sparsity = collect(zip([findnz(cθ)[1:2]...]...))

eyzz_sparsity = collect(zip([findnz(eyzz)[1:2]...]...))
eyzθ_sparsity = collect(zip([findnz(eyzθ)[1:2]...]...))

cyzz_sparsity = collect(zip([findnz(cyzz)[1:2]...]...))
cyzθ_sparsity = collect(zip([findnz(cyzθ)[1:2]...]...))

## build functions
threads = false 
checkbounds = true
# objective
o_func = Symbolics.build_function([os], z, p, 
    parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
oz_func = Symbolics.build_function(oz, z, p, 
    parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
oθ_func = Symbolics.build_function(oθ, z, p, 
    parallel=((threads && length(oθ) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
ozz_func = Symbolics.build_function(ozz.nzval, z, p, 
    parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
ozθ_func = Symbolics.build_function(ozθ.nzval, z, p, 
    parallel=((threads && length(ozθ.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]

# equality
e_func = Symbolics.build_function(es, z, p, 
    parallel=((threads && length(es) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
ez_func = Symbolics.build_function(ez.nzval, z, p, 
    parallel=((threads && length(ez.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]

e_func(zeros(length(es)), rand(nz), rand(np))

ez_func(zeros(length(ez.nzval)), rand(nz), rand(np))

eθ_func = Symbolics.build_function(eθ.nzval, z, p, 
    parallel=((threads && length(eθ.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]

ey_func = Symbolics.build_function([ey], z, p, ye, 
    parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
eyz_func = Symbolics.build_function(eyz, z, p, ye, 
    parallel=((threads && length(eyz) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
eyzz_func = Symbolics.build_function(eyzz.nzval, z, p, ye, 
    parallel=((threads && length(eyzz.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
eyzθ_func = Symbolics.build_function(eyzθ.nzval, z, p, ye, 
    parallel=((threads && length(eyzθ.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]

# cone 
c_func = Symbolics.build_function(cs, z, p, 
    parallel=((threads && length(cs) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
cz_func = Symbolics.build_function(cz.nzval, z, p, 
    parallel=((threads && length(cz.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
cθ_func = Symbolics.build_function(cθ.nzval, z, p, 
    parallel=((threads && length(cθ.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]

cy_func = Symbolics.build_function([cy], z, p, yc, 
    parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
cyz_func = Symbolics.build_function(cyz, z, p, yc, 
    parallel=((threads && length(cyz) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
cyzz_func = Symbolics.build_function(cyzz.nzval, z, p, yc, 
    parallel=((threads && length(cyzz.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]
cyzθ_func = Symbolics.build_function(cyzθ.nzval, z, p, yc, 
    parallel=((threads && length(cyzθ.nzval) > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
    expression=Val{false},
    checkbounds=checkbounds)[2]





options = Options(verbose=true)
o, oz, oθ, ozz, ozθ, OZZ, OZθ,ozz_s, ozθ_s, e, ez, eθ, EZ, Eθ, ez_s, eθ_s, ey, eyz, eyzz, eyzθ, EYZZ, EYZθ, eyzz_s, eyzθ_s, c, cz, cθ, CZ, Cθ, cz_s, cθ_s, cy, cyz, cyzz, cyzθ, CYZZ, CYZθ, cyzz_s, cyzθ_s, nn_idx, so_idx, nz, np, ne, nc = generate_trajectory_optimization(objective, dynamics, equality, nonnegative, second_order, num_states, num_actions, num_parameters; threads=options.codegen_threads, checkbounds=options.codegen_checkbounds, verbose=options.verbose);
    
m = ProblemMethods(
    o, oz, oθ, ozz, ozθ, OZZ, OZθ,ozz_s, ozθ_s, e, ez, eθ, EZ, Eθ, ez_s, eθ_s, ey, eyz, eyzz, eyzθ, EYZZ, EYZθ, eyzz_s, eyzθ_s, c, cz, cθ, CZ, Cθ, cz_s, cθ_s, cy, cyz, cyzz, cyzθ, CYZZ, CYZθ, cyzz_s, cyzθ_s,
);




solver = Solver(m, nz, np, ne, nc,
        # parameters=vcat(parameters...),
        # nonnegative_indices=nn_idx, 
        # second_order_indices=so_idx,
        options=options);

function eval_methods!(s::Solver, m::ProblemMethods) 
    nothing 
end

eval_methods!(solver, m)


problem = solver.problem; 
method = solver.methods; 
idx = solver.indices;
solution = solver.solution;
parameters = solver.parameters;



method.equality_constraint(problem.equality_constraint, solver.solution.variables, solver.parameters)
problem.equality_constraint
method.equality_jacobian_variables(method.equality_jacobian_variables_cache, solver.solution.variables, solver.parameters)
method.equality_jacobian_variables_cache



problem!(
        problem, 
        method, 
        idx, 
        solution, 
        parameters,
        objective=true,
        objective_gradient_variables=true,
        objective_gradient_parameters=true,
        objective_jacobian_variables_variables=true,
        objective_jacobian_variables_parameters=true,
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

a = 1

















# # ## solver
# solver = Solver(
#     objective, 
#     dynamics, 
#     equality, 
#     nonnegative, 
#     second_order, 
#     num_states, 
#     num_actions, 
#     num_parameters;
#     options=Options(
#         verbose=true))

# # ## initialize
# states_guess = linear_interpolation(state_initial, state_goal, horizon)
# actions_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
# x = trajectory(states_guess, actions_guess) # assemble into a single vector
# initialize!(solver, x)

# # ## solve 
# solve!(solver)

# # @benchmark solve!($solver)

