using BenchmarkTools 
using InteractiveUtils

# @testset "Examples: Pendulum" begin 
# ## horizon 
T = 11 

# ## dimensions 
nx = [2 for t = 1:T]
nu = [1 for t = 1:T-1] 
np = [0 for t = 1:T] 

# ## indices
x_idx = [sum(nx[1:(t-1)]) + sum(nu[1:(t-1)]) .+ collect(1:nx[t]) for t = 1:T]
u_idx = [sum(nx[1:t]) + sum(nu[1:(t-1)]) .+ collect(1:nu[t]) for t = 1:T-1]
p_idx = [sum(np[1:(t-1)]) .+ collect(1:np[t]) for t = 1:T]

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
dyn = [midpoint_implicit for t = 1:T-1]

# ## initialization
x1 = [0.0; 0.0] 
xT = [π; 0.0] 

# ## objective 
ot = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 0.1 * dot(x[1:2], x[1:2])
obj = [[ot for t = 1:T-1]..., oT]

# ## constraints 
eq1(x, u, w) = x - x1
eqt(x, u, w) = zeros(0)
eqT(x, u, w) = x - xT
eq = [eq1, [eqt for t = 2:T-1]..., eqT]

ineqt(x, u, w) = zeros(0)
ineq = [ineqt for t = 1:T]

soct(x, u, w) = zeros(0)
soc = [[soct] for t = 1:T]

# codegen
nz = sum(nx) + sum(nu)

@variables z[1:nz] p[1:sum(np)] 
x = [z[idx] for idx in x_idx]
u = [z[idx] for idx in u_idx]
θ = [p[idx] for idx in p_idx]

# symbolic expressions
o = [obj[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
d = [dyn[t](x[t+1], x[t], u[t], θ[t]) for t = 1:T-1]
e = [eq[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
i = [ineq[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
s = [[so(x[t], t == T ? zeros(0) : u[t], θ[t]) for so in soc[t]] for t = 1:T]

# constraint dimensions 
d_dim = [length(dt) for dt in d]
e_dim = [length(et) for et in e]
i_dim = [length(it) for it in i]
s_dim = [[length(so) for so in st] for st in s]

# assemble 
os = sum(o)
es = vcat(d..., e...)
cs = vcat(i..., (s...)...)

# derivatives 
oz = Symbolics.gradient(os, z)
oθ = Symbolics.gradient(os, θ)
ozz = Symbolics.sparsejacobian(oz, z)
ozθ = Symbolics.sparsejacobian(oz, θ)

ozz_sparsity = collect(zip([findnz(ozz)[1:2]...]...))
ozθ_sparsity = collect(zip([findnz(ozθ)[1:2]...]...))

ez = Symbolics.sparsejacobian(es, z)
eθ = Symbolics.sparsejacobian(es, θ)

ez_sparsity = collect(zip([findnz(ez)[1:2]...]...))
eθ_sparsity = collect(zip([findnz(eθ)[1:2]...]...))

cz = Symbolics.sparsejacobian(cs, z)
cθ = Symbolics.sparsejacobian(cs, θ)

cz_sparsity = collect(zip([findnz(cz)[1:2]...]...))
cθ_sparsity = collect(zip([findnz(cθ)[1:2]...]...))

# product derivatives
@variables ye[1:length(es)] yc[1:length(cs)]

ey = dot(es, ye)
cy = dot(cs, yc)

eyz = Symbolics.gradient(ey, z)
eyzz = Symbolics.sparsejacobian(eyz, z)
eyzθ = Symbolics.sparsejacobian(eyz, θ)

eyzz_sparsity = collect(zip([findnz(eyzz)[1:2]...]...))
eyzθ_sparsity = collect(zip([findnz(eyzθ)[1:2]...]...))

cyz = Symbolics.gradient(cy, z)
cyzz = Symbolics.sparsejacobian(cyz, z)
cyzθ = Symbolics.sparsejacobian(cyz, θ)

cyzz_sparsity = collect(zip([findnz(cyzz)[1:2]...]...))
cyzθ_sparsity = collect(zip([findnz(cyzθ)[1:2]...]...))

# build functions
o_func = Symbolics.build_function([os], z, p, checkbounds=true, expression=Val{false})[2]
oz_func = Symbolics.build_function(oz, z, p, checkbounds=true, expression=Val{false})[2]
oθ_func = Symbolics.build_function(oθ, z, p, checkbounds=true, expression=Val{false})[2]
ozz_func = Symbolics.build_function(ozz.nzval, z, p, checkbounds=true, expression=Val{false})[2]
ozθ_func = Symbolics.build_function(ozθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

e_func = Symbolics.build_function(es, z, p, checkbounds=true, expression=Val{false})[2]
ez_func = Symbolics.build_function(ez.nzval, z, p, checkbounds=true, expression=Val{false})[2]
eθ_func = Symbolics.build_function(eθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

ey_func = Symbolics.build_function([ey], z, p, ye, checkbounds=true, expression=Val{false})[2]
eyz_func = Symbolics.build_function(eyz, z, p, ye, checkbounds=true, expression=Val{false})[2]
eyzz_func = Symbolics.build_function(eyzz.nzval, z, p, ye, checkbounds=true, expression=Val{false})[2]
eyzθ_func = Symbolics.build_function(eyzθ.nzval, z, p, ye, checkbounds=true, expression=Val{false})[2]

c_func = Symbolics.build_function(cs, z, p, checkbounds=true, expression=Val{false})[2]
cz_func = Symbolics.build_function(cz.nzval, z, p, checkbounds=true, expression=Val{false})[2]
cθ_func = Symbolics.build_function(cθ.nzval, z, p, checkbounds=true, expression=Val{false})[2]

cy_func = Symbolics.build_function([cy], z, p, yc, checkbounds=true, expression=Val{false})[2]
cyz_func = Symbolics.build_function(cyz, z, p, yc, checkbounds=true, expression=Val{false})[2]
cyzz_func = Symbolics.build_function(cyzz.nzval, z, p, yc, checkbounds=true, expression=Val{false})[2]
cyzθ_func = Symbolics.build_function(cyzθ.nzval, z, p, yc, checkbounds=true, expression=Val{false})[2]

methods = ProblemMethods(
    o_func,
    oz_func,
    oθ_func, 
    ozz_func, 
    ozθ_func, 
    zeros(length(ozz_sparsity)), zeros(length(ozθ_sparsity)),
    ozz_sparsity, ozθ_sparsity,
    e_func,
    ez_func, 
    eθ_func, 
    zeros(length(ez_sparsity)), zeros(length(eθ_sparsity)),
    ez_sparsity, eθ_sparsity,
    ey_func, 
    eyz_func, 
    eyzz_func, 
    eyzθ_func, 
    zeros(length(eyzz_sparsity)), zeros(length(eyzθ_sparsity)),
    eyzz_sparsity, eyzθ_sparsity,
    c_func, 
    cz_func, 
    cθ_func, 
    zeros(length(cz_sparsity)), zeros(length(cθ_sparsity)),
    cz_sparsity, cθ_sparsity,
    cy_func, 
    cyz_func,
    cyzz_func, 
    cyzθ_func, 
    zeros(length(cyzz_sparsity)), zeros(length(cyzθ_sparsity)),
    cyzz_sparsity, cyzθ_sparsity,
)



# ## problem 
# trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, soc)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(nu[t]) for t = 1:T-1]

z_guess = zeros(nz)
for t = 1:T 
    z_guess[x_idx[t]] = x_interpolation[t] 
    t == T && continue 
    z_guess[u_idx[t]] = u_guess[t] 
end

function constraint_indices(dims::Vector{Int}; 
    shift=0) where T

    indices = Vector{Int}[]

    for d in dims
        indices = [indices..., collect(shift .+ (1:d)),]
        shift += d
    end

    return indices
end 

function constraint_indices(dims::Vector{Vector{Int}}; 
    shift=0) where T

    INDICES = Vector{Vector{Int}}[]
    for dim in dims
        indices = Vector{Int}[]
        for d in dim
            indices = [indices..., collect(shift .+ (1:d)),]
            shift += d
        end
        push!(INDICES, indices)
    end

    return INDICES
end 

# nonnegative duals
idx_nn = constraint_indices(i_dim)

# second-order duals
idx_so = constraint_indices(s_dim, 
    shift=sum(i_dim))


# ## solver
nz
sum(np)
sum(d_dim)
sum(e_dim)
sum(i_dim)

sum(i_dim) + sum(sum.(s_dim))


solver = Solver(methods, nz, sum(np), sum(d_dim) + sum(e_dim), sum(i_dim) + sum(sum.(s_dim)),
    # nonnegative_indices=idx_nn, 
    # second_order_indices=idx_so,
    options=Options(verbose=true))
# initialize_states!(solver, trajopt, x_interpolation) 
# initialize_controls!(solver, trajopt, u_guess)
initialize!(solver, z_guess)

problem = solver.problem 
method = solver.methods 
cone_methods = solver.cone_methods
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
solver.dimensions.parameters
problem.objective_gradient_parameters
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

cone!(problem, cone_methods, idx, solution;
    barrier=true, 
    barrier_gradient=true,
    product=true,
    jacobian=true,
    target=true,
)

solver.problem.objective
solver.problem.objective_gradient_variables
solver.problem.objective_jacobian_variables_variables

solver.solution.variables

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