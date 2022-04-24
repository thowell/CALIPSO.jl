using BenchmarkTools 
using InteractiveUtils

num_variables = 10
num_equality = 5
num_cone = num_variables
num_parameters = num_variables + num_variables + num_equality * num_variables + num_equality

x̂ = max.(0.0, randn(num_variables))
Q = rand(num_variables, num_variables) 
P = Diagonal(diag(Q' * Q))
p = randn(num_variables)
A = rand(num_equality, num_variables)
b = A * x̂

parameters = [diag(P); p; vec(A); b]

function obj(x, θ) 
    Pθ = Diagonal(θ[1:num_variables])
    pθ = θ[num_variables .+ (1:num_variables)]
    
    J = 0.0
    J += 0.5 * transpose(x) * Pθ * x 
    J += transpose(pθ) * x

    return J 
end

function eq(x, θ)
    Aθ = reshape(θ[num_variables + num_variables .+ (1:(num_equality * num_variables))], num_equality, num_variables)
    bθ = θ[num_variables + num_variables + num_equality * num_variables .+ (1:num_equality)]
    return Aθ * x - bθ
end

cone(x, θ) = x

# solver
x0 = randn(num_variables)

# num_variables = 3
# num_parameters = 0
# num_equality = 2
# num_cone = 2
# x0 = [-2.0, 3.0, 1.0]

# obj(x, θ) = x[1]
# eq(x, θ) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
# cone(x, θ) = x[2:3]

# solver
method = ProblemMethods(num_variables, num_parameters, obj, eq, cone)
solver = Solver(method, num_variables, num_parameters, num_equality, num_cone; 
    options=Options(
            verbose=false, 
            differentiate=true,
    ))
initialize!(solver, x0)
solve!(solver)
@benchmark solve!($solver)
@code_warntype solve!(solver)

# @code_warntype iterative_refinement!(solver.data.step, solver)

# @benchmark iterative_refinement!($solver.data.step, $solver)

linear_solve!(solver.linear_solver, solver.data.residual_symmetric.all, solver.data.jacobian_variables_symmetric, solver.data.residual_symmetric.all)
@code_warntype linear_solve!(solver.linear_solver, solver.data.residual_symmetric.all, solver.data.jacobian_variables_symmetric, solver.data.residual_symmetric.all)
@benchmark linear_solve!($solver.linear_solver, $solver.data.residual_symmetric.all, $solver.data.jacobian_variables_symmetric, $solver.data.residual_symmetric.all)
@code_warntype solve!(solver)

a = 1
# problem = solver.problem 
# method = solver.methods 
# idx = solver.indices 
# solution = solver.solution
# parameters = solver.parameters

# @code_warntype problem!(
#         problem, 
#         method, 
#         idx, 
#         solution, 
#         parameters)

# @benchmark problem!(
#         $problem, 
#         $method, 
#         $idx, 
#         $solution, 
#         $parameters,
#         objective=true,
#         objective_gradient_variables=true,
#         objective_gradient_parameters=true,
#         objective_jacobian_variables_variables=true,
#         objective_jacobian_variables_parameters=true,
#         equality_constraint=true,
#         equality_jacobian_variables=true,
#         equality_jacobian_parameters=true,
#         equality_dual=true,
#         equality_dual_jacobian_variables=true,
#         equality_dual_jacobian_variables_variables=true,
#         equality_dual_jacobian_variables_parameters=true,
#         cone_constraint=true,
#         cone_jacobian_variables=true,
#         cone_jacobian_parameters=true,
#         cone_dual=true,

#         cone_dual_jacobian_variables=true,
#         cone_dual_jacobian_variables_variables=true,
#         cone_dual_jacobian_variables_parameters=true,
#     )

# # solve 


# # @benchmark compute_inertia!($solver.linear_solver)


# a = 1
# # # create random KKT system
# # nz = 100
# # nc = 70
# # H = sprand(nz,nz,0.05);
# # H = H'*H + I
# # b = randn(nz + nc)
# # A1 = sprand(nc,nz,0.8)
# # K1 = [H A1';A1 -1e-3*I]
# # K2 = copy(K1)
# # @benchmark triu!($triuK1, $triuK1)
# # @benchmark
# # # get triu of K1 and create QDLDL struct
# # @benchmark $triuK1 .= triu($K1)
# # F = qdldl(triuK1)
# # @benchmark SparseArrays.fkeep!(A, (i, j, x) -> j >= i + 0) setup=(A=K1) evals=1
# # # compare with backslash
# # @test norm(K1\b - F\b,Inf) < 1e-12
# # @benchmark SparseArrays.triu!(A) setup=(A=K1) evals=1
# # # create a new KKT system with the same sparsity pattern
# # A2 = copy(A1)
# # A2.nzval .= randn(length(A2.nzval))
# # K2 = [H A2';A2 -1e-7*I]
# # triuK2 = triu(K2)

# # # update factorization of F in place (non allocating)
# # update_values!(F,1:length(triuK2.nzval),triuK2.nzval)
# # @benchmark update_values!($F,$(1:length(triuK2.nzval)),$(triuK2.nzval))

# # @benchmark refactor!($F)

# # A = sparse([1.0 2.0; 0.0 4.0])
# # A.nzval

# # fill!(A, 0.0)
# # A.nzval

