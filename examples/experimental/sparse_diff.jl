using BenchmarkTools 
using InteractiveUtils
using FiniteDiff 
using SparseDiffTools
using SparsityDetection 
using SparseArrays
using LinearAlgebra

function d!(d, x)
    n = length(x) 
    for i = 1:n 
        d[i] = cos(x[i])
    end
    return d
end

n = 100
d0 = zeros(n)
dx0 = zeros(n, n) 
x0 = randn(n)

d!(d0, x0)
@code_warntype d!(d0, x0)
@benchmark d!($d0, $x0)

dx_cache = FiniteDiff.JacobianCache(x0, x0)
FiniteDiff.finite_difference_jacobian!(dx0, (a, b) -> d!(a, b), x0, dx_cache)

dx! = (a, b) -> d!(a, b)
@code_warntype FiniteDiff.finite_difference_jacobian!(dx0, dx!, x0, dx_cache)
@benchmark FiniteDiff.finite_difference_jacobian!($dx0, $dx!, $x0, $dx_cache)

dx_sparsity_pattern = SparsityDetection.jacobian_sparsity(d!, d0, x0)
dx_jac = Float64.(sparse(dx_sparsity_pattern))

dx_colors = SparseDiffTools.matrix_colors(dx_jac)
dx_sparse_cache = FiniteDiff.JacobianCache(x0, x0, colorvec=dx_colors)

FiniteDiff.finite_difference_jacobian!(dx_jac, dx!, x0, dx_cache, colorvec=dx_colors)

@code_warntype FiniteDiff.finite_difference_jacobian!(dx_jac, dx!, x0, dx_cache, colorvec=dx_colors)
@benchmark FiniteDiff.finite_difference_jacobian!($dx_jac, $dx!, $x0, $dx_sparse_cache, colorvec=$dx_colors)

struct DCache{T} 
    d::Vector{T} 
end 

dcache = DCache(zeros(n))

function d(cache::DCache{T}, x::Vector{T}) where T
    fill!(cache.d, 0.0)
    d!(cache.d, x) 
    return cache.d 
end

d(dcache, x0)
@code_warntype d(dcache, x0)
@benchmark d($dcache, $x0)

function dy(cache::DCache{T}, x::Vector{T}, y::Vector{T})::T where T
    d(cache, x) 
    dot(cache.d, y) 
end

y0 = randn(n)

dy(dcache, x0, y0)
@code_warntype dy(dcache, x0, y0)
@benchmark dy($dcache, $x0, $y0)

dyx_cache = FiniteDiff.JacobianCache(x0, x0)

dyx_sparsity_pattern = SparsityDetection.jacobian_sparsity((a, b) -> begin d!(a, b); return dot(a, y0) end, x0)
dyx_jac = Float64.(sparse(dyx_sparsity_pattern))

dyx_colors = SparseDiffTools.matrix_colors(dyx_jac)
dx_sparse_cache = FiniteDiff.JacobianCache(x0, x0, colorvec=dx_colors)

FiniteDiff.finite_difference_jacobian!(dx_jac, dx!, x0, dx_cache, colorvec=dx_colors)

FiniteDiff.finite_difference_
