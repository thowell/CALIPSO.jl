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
T = 5

# ## acrobot 
num_state = 2
num_action = 1 
num_parameters = [2 * 2 + 2 + 2 + 1 + 2, [2 * 2 + 2 + 2 + 1 for t = 2:T-1]..., 2 + 2]

# ## parameters
x1 = [0.0; 0.0] 
xT = [1.0; 0.0] 

A = [1.0 1.0; 0.0 1.0]
B = [0.0; 1.0]
Qt = [1.0 0.0; 0.0 1.0] 
Rt = [0.1]
QT = [10.0 0.0; 0.0 10.0] 
θ1 = [vec(A); B; diag(Qt); Rt; x1]
θt = [vec(A); B; diag(Qt); Rt]  
θT = [diag(QT); xT] 
parameters = [θ1, [θt for t = 2:T-1]..., θT]

function double_integrator(y, x, u, w)
    A = reshape(w[1:4], 2, 2) 
    B = w[4 .+ (1:2)] 

    return y - (A * x + B * u[1])
end

# ## model
dyn = [
        Dynamics(
        double_integrator, 
        num_state, 
        num_state, 
        num_action, 
        num_parameter=num_parameters[t]) for t = 1:T-1
] 

# ## objective 
function o1(x, u, w) 
    Q1 = Diagonal(w[6 .+ (1:2)])
    R1 = w[8 + 1]
    return 0.5 * transpose(x) * Q1 * x + 0.5 * R1 * transpose(u) * u
end

function ot(x, u, w) 
    Qt = Diagonal(w[6 .+ (1:2)])
    Rt = w[8 + 1]
    return 0.5 * transpose(x) * Qt * x + 0.5 * Rt * transpose(u) * u
end

function oT(x, u, w) 
    QT = Diagonal(w[0 .+ (1:2)])
    return 0.5 * transpose(x) * QT * x
end

obj = [
        Cost(o1, num_state, num_action, 
            num_parameter=num_parameters[1]),
        [Cost(ot, num_state, num_action, 
            num_parameter=num_parameters[t]) for t = 2:T-1]...,
        Cost(oT, num_state, 0,
            num_parameter=num_parameters[T]),
]

# ## constraints 
eq = [
    Constraint((x, u, w) -> 1 * (x - w[9 .+ (1:2)]), num_state, num_action,
        num_parameter=num_parameters[1]),
    [Constraint() for t = 2:T-1]...,
    Constraint((x, u, w) -> 1 * (x - w[2 .+ (1:2)]), num_state, 0,
        num_parameter=num_parameters[T]),
]

ineq = [Constraint() for t = 1:T]
so = [[Constraint()] for t = 1:T]

# ## problem 
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so;
    parameters=parameters)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
u_guess = [1.0 * randn(num_action) for t = 1:T-1]

methods = ProblemMethods(trajopt)

# ## solver
solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    parameters=vcat(parameters...),
    options=Options(
        residual_tolerance=1.0e-12, 
        equality_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-8,
        differentiate=true))

initialize_states!(solver, trajopt, x_interpolation) 
initialize_actions!(solver, trajopt, u_guess)

# ## solve 
solve!(solver)

# ## tests 
using Test
@test solver.dimensions.parameters - sum(num_parameters) == 0
@test norm(solver.parameters - vcat([θ1, [θt for t = 2:T-1]..., θT]...), Inf) < 1.0e-5
@test norm(solver.data.residual, Inf) < 1.0e-5

nz = T * num_state + (T - 1) * num_action
nz += num_state + num_state # t=1
for t = 2:T-1 
    nz += num_state 
end
nz += num_state

nθ = sum(num_parameters)

function lagrangian(z, θ)
    x = [z[(t - 1) * (num_state + num_action) .+ (1:num_state)] for t = 1:T]
    u = [z[(t - 1) * (num_state + num_action) + num_state .+ (1:num_action)] for t = 1:T-1]
    λdyn = [z[(T - 1) * (num_state + num_action) + num_state + (t - 1) * num_state .+ (1:num_state)] for t = 1:T-1] 
    λx1 = z[(T - 1) * (num_state + num_action) + num_state + (T - 1) * num_state .+ (1:num_state)] 
    λxT = z[(T - 1) * (num_state + num_action) + num_state + (T - 1) * num_state + num_state .+ (1:num_state)]

    w = [θ[sum(num_parameters[1:(t-1)]) .+ (1:num_parameters[t])] for t = 1:T]

    L = 0.0 

    for t = 1:T 
        if t == 1
            L += o1(x[1], u[1], w[1])
            L += transpose(λdyn[1]) * double_integrator(x[2], x[1], u[1], w[1]);
            L += transpose(λx1) * (x[1] - w[1][9 .+ (1:2)])
        elseif t == T 
            L += oT(x[T], zeros(0), w[T]) 
            L += transpose(λxT) * (x[T] - w[T][2 .+ (1:2)])
        else 
            L += ot(x[t], u[t], w[t]) 
            L += transpose(λdyn[t]) * double_integrator(x[t+1], x[t], u[t], w[t])
        end 
    end

    return L
end

@variables zv[1:nz] θv[1:nθ]

L = lagrangian(zv, θv)
Lz = Symbolics.gradient(L, zv)
Lzz = Symbolics.jacobian(Lz, zv)
Lzθ = Symbolics.jacobian(Lz, θv)

Lzz_func = eval(Symbolics.build_function(Lzz, zv, θv)[2])
Lzθ_func = eval(Symbolics.build_function(Lzθ, zv, θv)[2])

Lzz0 = zeros(nz, nz)
Lzθ0 = zeros(nz, nθ)

Lzz_func(Lzz0, [solver.solution.variables; solver.solution.equality_dual], solver.parameters)
Lzθ_func(Lzθ0, [solver.solution.variables; solver.solution.equality_dual], solver.parameters)

sensitivity = -1.0 * Lzz0 \ Lzθ0

@test norm(Lzθ0[solver.indices.variables, :] - solver.problem.objective_jacobian_variables_parameters - solver.problem.equality_dual_jacobian_variables_parameters, Inf) < 1.0e-5
@test norm(Lzθ0[solver.dimensions.variables .+ (1:solver.dimensions.equality_dual), :] - solver.problem.equality_jacobian_parameters, Inf) < 1.0e-5
@test norm(sensitivity[solver.indices.variables, :] - solver.data.solution_sensitivity[solver.indices.variables, :], Inf) < 1.0e-3
