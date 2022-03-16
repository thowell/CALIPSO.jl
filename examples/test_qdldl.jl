using CALIPSO
using LinearAlgebra
using QDLDL
using SparseArrays
using Random 
using BenchmarkTools

rng = Random.MersenneTwister(0706)

function random_psd(n)

    A = sprandn(rng,n,n,0.2)
    A = A+A';
    A = A + Diagonal(((sum(abs.(A),dims=1)[:]))) #make diagonally dominant

end

m = 20
n = 30

A = random_psd(n)
B = sprandn(m,n,0.2)
C = -random_psd(m)
M = [A B' ; B C]
M.nzval
solver = ldl_solver(M)

solver.F
x = zeros(n + m)
b = rand(n + m)

sol_lu = M \ b
linear_solve!(solver, x, M, b)
x - sol_lu

@benchmark linear_solve!($solver, $x, $M, $b)
M_dense = Array(M)
@benchmark linear_solve!($solver, $x, $M_dense, $b)

n, m = size(M) 
M_sp = spzeros(n, m)
for i = 1:n 
    for j = 1:m 
        M_sp[i, j] = M[i, j]
    end
end

# M.nzval
# M.rowval

# update_A!(F, M)
# @benchmark update_A!($F, $M)




##### 
n = 10
m = 3

nw = n + m 
nθ = 2n + m * n + m

function obj(z, θ)
    x = z[1:n]

    P = Diagonal(θ[1:n])
    p = θ[n .+ (1:n)]
    return transpose(x) * P * x + transpose(p) * x 
end

function constraints(z, θ) 
    x = z[1:n] 
    A = reshape(θ[2n .+ (1:(m * n))], m, n)
    b = θ[2n + m * n .+ (1:m)]

    A * x - b
end

function lagrangian(w, θ)
    z = w[1:n]
    y = w[n .+ (1:m)]

    L = 0.0 
    L += obj(z, θ)
    L += dot(y, constraints(z, θ))

    return L
end

# residual
function _r(w, θ, κ)
    L = lagrangian(w, θ)
    Lz = Symbolics.gradient(L, w[1:n])
    c = constraints(w[1:n], θ)

    [
        Lz; 
        c; 
    ]
end

@variables z_sym[1:nw]
@variables θ_sym[1:nθ]
@variables κ_sym[1:1]

r_sym = _r(z_sym, θ_sym, κ_sym)
rf! = eval(Symbolics.build_function(r_sym, z_sym, θ_sym, κ_sym)[2])
rz_exp = Symbolics.jacobian(r_sym, z_sym)
rθ_exp = Symbolics.jacobian(r_sym, θ_sym)
rz_sp = similar(rz_exp, Float64)
rθ_sp = similar(rθ_exp, Float64)
rzf! = eval(Symbolics.build_function(rz_exp, z_sym, θ_sym)[2])
rθf! = eval(Symbolics.build_function(rθ_exp, z_sym, θ_sym)[2])

function rzf_reg!(rz, z, θ)
    rzf!(rz, z, θ)
    rz .+= Diagonal([1.0e-5 * ones(n); -1.0e-5 * ones(m)])
end

x0 = randn(n)
A = rand(m, n)
b = A * x0
z = [randn(n); zeros(m)] 
θ = [ones(n); zeros(n); vec(A); b]
κ = [1.0]
r = zeros(nw)
rz = zeros(nw, nw)
rθ = zeros(nw, nθ)
rf!(r, z, θ, κ)
rzf!(rz, z, θ)
rθf!(rθ, z, θ)
r 
rz 

rθ
# options
opts = CALIPSO.InteriorPointOptions(diff_sol = true, 
    undercut=10.0,
    max_ls=25, 
    max_iter=100, 
    verbose=true, 
    solver=:lu_solver)

idx = CALIPSO.IndicesOptimization(
        nw, nw, 
        [collect(1:0), collect(1:0)],
        [collect(1:0), collect(1:0)],
        Vector{Vector{Int}}[], 
        Vector{Vector{Int}}[], 
        collect(1:(n + m)), 
        Vector{Int}(), 
        Vector{Int}(),
        Vector{Int}[],
        Vector{Int}())

rzf_reg!(rz, z, θ)
rank(rz)
cond(rz)
eigen(rz).values

# solver
ip = CALIPSO.interior_point(z, θ,
    idx=idx,
    r! = rf!, rz! = rzf_reg!, rθ! = rθf!,
    rz = rz,
    rθ = rθ,
    opts = opts)

ip.rz \ ip.r
F = qdldl(sparse(ip.rz))
solve(F, ip.r)
# solve
status = CALIPSO.interior_point_solve!(ip)


