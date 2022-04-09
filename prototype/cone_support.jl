using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

# setup 
num_variables = 100 
num_nonnegative = 10 
num_soc = [5, 3, 2]
num_cone = num_nonnegative + sum(num_soc)

x = rand(num_variables) 
y = rand(num_variables) 

ineq(x) = x[20 .+ (1:num_nonnegative)] 

soc1(x) = x[1:num_soc[1]] 
soc2(x) = x[num_soc[1] .+ (1:num_soc[2])] 
soc3(x) = x[num_soc[1] + num_soc[2] .+ (1:num_soc[3])] 

# indices
idx_ineq = collect(1:num_nonnegative) 
idx_soc = [collect(num_nonnegative + sum(num_soc[1:(i-1)]) .+ (1:num_soc[i])) for i = 1:length(num_soc)]

# initialize
function initialize_nonnegative!(x, idx_ineq; 
    initial=1.0) 
    x[idx_ineq] .= initial
    return 
end

function initialize_second_order!(x, idx_soc; 
    first_initial=1.0, initial=0.1)
    for idx in idx_soc
        for (i, ii) in enumerate(idx)
            x[ii] = (i == 1 ? first_initial : initial)
        end 
    end 
    return 
end

function initialize_cone!(x, idx_ineq, idx_soc) 
    initialize_nonnegative!(x, idx_ineq) 
    initialize_second_order!(x, idx_soc) 
end

initialize_cone!(x, idx_ineq, idx_soc)
initialize_cone!(y, idx_ineq, idx_soc)

# barrier 
nonnegative_barrier(x) = sum(log.(x)) 
nonnegative_barrier_gradient(x) = 1.0 ./ x

second_order_barrier(x) = 0.5 * log(x[1]^2 - dot(x[2:end], x[2:end]))
second_order_barrier_gradient(x) = 1.0 / (x[1]^2 - dot(x[2:end], x[2:end])) * [x[1]; -x[2:end]]

function cone_barrier(x, idx_ineq, idx_soc)
    Φ = 0.0 
    
    # non-negative orthant
    x_ineq = @views x[idx_ineq]
    Φ -= nonnegative_barrier(x_ineq) 

    # soc 
    for idx in idx_soc 
        x_soc = @views x[idx] 
        Φ -= second_order_barrier(x_soc) 
    end

    return Φ 
end

function cone_barrier_gradient(x, idx_ineq, idx_soc)
    vcat(
        [nonnegative_barrier_gradient(x[idx_ineq]), 
        [second_order_barrier_gradient(x[idx]) for idx in idx_soc]...]...)
end

cone_barrier(x, idx_ineq, idx_soc)
cone_barrier_gradient(x, idx_ineq, idx_soc)

# product 
nonnegative_product(a, b) = a .* b 
second_order_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]

function cone_product(a, b, idx_ineq, idx_soc) 
    vcat(
        [nonnegative_product(a[idx_ineq], b[idx_ineq]), 
        [second_order_product(a[idx], b[idx]) for idx in idx_soc]...]...)
end

nonnegative_product(x[idx_ineq], y[idx_ineq])
second_order_product(x[idx_soc[1]], y[idx_soc[1]])
second_order_product(x[idx_soc[2]], y[idx_soc[2]])
cone_product(x, y, idx_ineq, idx_soc)

function nonnegative_product_jacobian(a, b) 
    Diagonal(b) 
end

function second_order_product_jacobian(a, b) 
    n = length(a)
    Diagonal(b[1] * ones(n)) + [0.0 b[2:n]'; b[2:n] zeros(n-1, n-1)]
end

function cone_product_jacobian(a, b, idx_ineq, idx_soc) 
    cat(
        nonnegative_product_jacobian(a[idx_ineq], b[idx_ineq]),
        [second_order_product_jacobian(a[idx], b[idx]) for idx in idx_soc]..., 
    dims=(1, 2))
end

function cone_product_jacobian_inverse(a, b, idx_ineq, idx_soc) 
    cat(
        inv(nonnegative_product_jacobian(a[idx_ineq], b[idx_ineq])),
        [inv(second_order_product_jacobian(a[idx], b[idx])) for idx in idx_soc]..., 
    dims=(1, 2))
end

nonnegative_product_jacobian(x[idx_ineq], y[idx_ineq])
second_order_product_jacobian(x[idx_soc[1]], y[idx_soc[1]])
cone_product_jacobian(x, y, idx_ineq, idx_soc)
cone_product_jacobian(x, y, idx_ineq, idx_soc)

norm(Array(cone_product_jacobian_inverse(x, y, idx_ineq, idx_soc)) - inv(Array(cone_product_jacobian(x, y, idx_ineq, idx_soc))))

# target 
nonnegative_target(idx) = ones(length(idx))
second_order_target(idx) = [1.0; zeros(length(idx) - 1)] 

function cone_target(idx_ineq, idx_soc) 
    vcat(
        [nonnegative_target(idx_ineq), 
        [second_order_target(idx) for idx in idx_soc]...]...)
end

nonnegative_target(idx_ineq)
second_order_target(idx_soc[1])
second_order_target(idx_soc[2]) 
cone_target(idx_ineq, idx_soc)

# violation 
function nonnegative_violation(x)
    for xi in x 
        xi <= 0.0 && (return true)
    end 
    return false 
end
second_order_violation(x) = x[1] <= norm(x[2:end])

function cone_violation(x, idx_ineq, idx_soc) 
    nonnegative_violation(x[idx_ineq]) && (return true)
    for idx in idx_soc 
        second_order_violation(x[idx]) && (return true)
    end
    return false
end

nonnegative_violation(x[idx_ineq])
second_order_violation(x[idx_soc[1]])
second_order_violation(ones(5))
cone_violation(x, idx_ineq, idx_soc)
cone_violation(ones(num_variables), idx_ineq, idx_soc)