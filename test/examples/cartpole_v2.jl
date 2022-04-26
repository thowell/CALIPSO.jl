# @testset "Examples: Cart-pole" begin 
# ## horizon 
horizon = 51

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

# # codegen
# nx = num_states 
# nu = num_actions
# np = num_parameters
# nz = sum(nx) + sum(nu)

# x_idx, u_idx = state_action_indices(nx, nu)
# p_idx = parameter_indices(np) 

# @variables z[1:nz] p[1:sum(np)] 
# x = [z[idx] for idx in x_idx]
# u = [z[idx] for idx in u_idx]
# θ = [p[idx] for idx in p_idx]

# # symbolic expressions
# T = horizon
# o = [objective[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
# d = [dynamics[t](x[t+1], x[t], u[t], θ[t]) for t = 1:T-1]
# e = [equality[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
# i = [nonnegative[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
# s = [[so(x[t], t == T ? zeros(0) : u[t], θ[t]) for so in second_order[t]] for t = 1:T]

# # constraint dimensions 
# d_dim = [length(dt) for dt in d]
# e_dim = [length(et) for et in e]
# i_dim = [length(it) for it in i]
# s_dim = [[length(so) for so in st] for st in s]

# # assemble 
# os = sum(o)
# es = vcat(d..., e...)
# cs = vcat(i..., (s...)...)

# # derivatives 
# oz = Symbolics.gradient(os, z)
# oθ = Symbolics.gradient(os, θ)
# ozz = Symbolics.sparsejacobian(oz, z)
# ozθ = Symbolics.sparsejacobian(oz, θ)

# ozz_sparsity = collect(zip([findnz(ozz)[1:2]...]...))
# ozθ_sparsity = collect(zip([findnz(ozθ)[1:2]...]...))

# ez = Symbolics.sparsejacobian(es, z)
# eθ = Symbolics.sparsejacobian(es, θ)

# ez_sparsity = collect(zip([findnz(ez)[1:2]...]...))
# eθ_sparsity = collect(zip([findnz(eθ)[1:2]...]...))

# cz = Symbolics.sparsejacobian(cs, z)
# cθ = Symbolics.sparsejacobian(cs, θ)

# cz_sparsity = collect(zip([findnz(cz)[1:2]...]...))
# cθ_sparsity = collect(zip([findnz(cθ)[1:2]...]...))

# # product derivatives
# @variables ye[1:length(es)] yc[1:length(cs)]

# ey = dot(es, ye)
# cy = dot(cs, yc)

# eyz = Symbolics.gradient(ey, z)
# eyzz = Symbolics.sparsejacobian(eyz, z)
# eyzθ = Symbolics.sparsejacobian(eyz, θ)

# eyzz_sparsity = collect(zip([findnz(eyzz)[1:2]...]...))
# eyzθ_sparsity = collect(zip([findnz(eyzθ)[1:2]...]...))

# cyz = Symbolics.gradient(cy, z)
# cyzz = Symbolics.sparsejacobian(cyz, z)
# cyzθ = Symbolics.sparsejacobian(cyz, θ)

# cyzz_sparsity = collect(zip([findnz(cyzz)[1:2]...]...))
# cyzθ_sparsity = collect(zip([findnz(cyzθ)[1:2]...]...))

# # build functions
# o_func = Symbolics.build_function([os], z, p, checkbounds=true, expression=Val{false})[2]
# o_func_alt = eval(Symbolics.build_function([os], z, p, checkbounds=true)[2])
# typeof(o_func) <: Function
# typeof(o_func_alt) <: Function

# struct FuncStruct{F<:Function}
#     x::F
# end

# FuncStruct(o_func)
# FuncStruct(o_func_alt)


# oz_func = Symbolics.build_function(oz, z, p, checkbounds=true, expression=Val{false})[2]
# oθ_func = Symbolics.build_function(oθ, z, p, checkbounds=true, expression=Val{false})[2]
# ozz_func = Symbolics.build_function(ozz.nzval, z, p, checkbounds=true, expression=Val{false})[2]
# ozθ_func = Symbolics.build_function(ozθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

# e_func = Symbolics.build_function(es, z, p, checkbounds=true, expression=Val{false})[2]
# ez_func = Symbolics.build_function(ez.nzval, z, p, checkbounds=true, expression=Val{false})[2]
# eθ_func = Symbolics.build_function(eθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

# ey_func = Symbolics.build_function([ey], z, p, ye, checkbounds=true, expression=Val{false})[2]
# eyz_func = Symbolics.build_function(eyz, z, p, ye, checkbounds=true, expression=Val{false})[2]
# eyzz_func = Symbolics.build_function(eyzz.nzval, z, p, ye, checkbounds=true, expression=Val{false})[2]
# eyzθ_func = Symbolics.build_function(eyzθ.nzval, z, p, ye, checkbounds=true, expression=Val{false})[2]

# c_func = Symbolics.build_function(cs, z, p, checkbounds=true, expression=Val{false})[2]
# cz_func = Symbolics.build_function(cz.nzval, z, p, checkbounds=true, expression=Val{false})[2]
# cθ_func = Symbolics.build_function(cθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

# cy_func = Symbolics.build_function([cy], z, p, yc, checkbounds=true, expression=Val{false})[2]
# cyz_func = Symbolics.build_function(cyz, z, p, yc, checkbounds=true, expression=Val{false})[2]
# cyzz_func = Symbolics.build_function(cyzz.nzval, z, p, yc, checkbounds=true, expression=Val{false})[2]
# cyzθ_func = Symbolics.build_function(cyzθ.nzval, z, p, yc, checkbounds=true, expression=Val{false})[2]


# ## solver
# m, nn_idx, so_idx, nz, np, ne, nc = Solver(
#     objective, 
#     dynamics, 
#     equality, 
#     nonnegative, 
#     second_order, 
#     num_states, 
#     num_actions, 
#     num_parameters;
#     options=Options());

# solver = Solver(m, nz, np, ne, nc);

solver = Solver(
    objective, 
    dynamics, 
    equality, 
    nonnegative, 
    second_order, 
    num_states, 
    num_actions, 
    num_parameters;
    options=Options());

# problem!(p_data, methods, idx, solution, parameters,
#     objective=true,
#     objective_gradient_variables=true,
#     objective_gradient_parameters=true,
#     objective_jacobian_variables_variables=true,
#     objective_jacobian_variables_parameters=true,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     equality_jacobian_parameters=true,
#     equality_dual=true,
#     equality_dual_jacobian_variables=true,
#     equality_dual_jacobian_variables_variables=true,
#     equality_dual_jacobian_variables_parameters=true,
#     cone_constraint=true,
#     cone_jacobian_variables=true,
#     cone_jacobian_parameters=true,
#     cone_dual=true,
#     cone_dual_jacobian_variables=true,
#     cone_dual_jacobian_variables_variables=true,
#     cone_dual_jacobian_variables_parameters=true,
# )

# # ## initialize
states_guess = linear_interpolation(state_initial, state_goal, horizon)
actions_guess = [0.01 * ones(num_actions[t]) for t = 1:horizon-1]
x = trajectory(states_guess, actions_guess) # assemble into a single vector
initialize!(solver, x)
solver.options.verbose=false
# solver

# solver = Solver(m, nz, np, ne, nc,
#         parameters=vcat(parameters...),
#         nonnegative_indices=nn_idx, 
#         second_order_indices=so_idx,
#         options=options)

# # ## problem 
# trajopt = CALIPSO.TrajectoryOptimizationProblem(dynamics, objective, equality, nonnegative, second_order)
# methods = ProblemMethods(trajopt)
# idx_nn, idx_soc = CALIPSO.cone_indices(trajopt)

# # solve 
solve!(solver)

solver.solution.variables[end-2:end]

solver.dimensions.variables
sum(num_states)
sum(num_actions)

state_action_indices(num_states, num_actions)[1]
# # test solution
# @test norm(solver.data.residual.all, solver.options.residual_norm) / solver.dimensions.total < solver.options.residual_tolerance

# slack_norm = max(
#                 norm(solver.data.residual.equality_dual, Inf),
#                 norm(solver.data.residual.cone_dual, Inf),
# )
# @test slack_norm < solver.options.slack_tolerance

# @test norm(solver.problem.equality_constraint, Inf) <= solver.options.equality_tolerance 
# @test norm(solver.problem.cone_product, Inf) <= solver.options.complementarity_tolerance 
# # end

