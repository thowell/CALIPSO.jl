using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

# setup 
num_variables = 100 
num_inequality = 10 
num_soc = [5, 3, 2]
num_cone = num_inequality + sum(num_soc)

x = rand(num_variables) 
y = rand(num_variables) 

ineq(x) = x[20 .+ (1:num_inequality)] 

soc1(x) = x[1:num_soc[1]] 
soc2(x) = x[num_soc[1] .+ (1:num_soc[2])] 
soc3(x) = x[num_soc[1] + num_soc[2] .+ (1:num_soc[3])] 

# indices
idx_ineq = collect(1:num_inequality) 
idx_soc = [collect(num_inequality + sum(num_soc[1:(i-1)]) .+ (1:num_soc[i])) for i = 1:length(num_soc)]

# initialize
function initialize_inequality!(x, idx_ineq; 
    initial=1.0) 
    x[idx_ineq] .= initial
    return 
end

function initialize_soc!(x, idx_soc; 
    first_initial=1.0, initial=0.1)
    for idx in idx_soc
        for (i, ii) in enumerate(idx)
            x[ii] = (i == 1 ? first_initial : initial)
        end 
    end 
    return 
end

function initialize_cone!(x, idx_ineq, idx_soc) 
    initialize_inequality!(x, idx_ineq) 
    initialize_soc!(x, idx_soc) 
end

initialize_cone!(x, idx_ineq, idx_soc)
initialize_cone!(y, idx_ineq, idx_soc)

# barrier 
barrier_inequality(x) = sum(log.(x)) 
barrier_gradient_inequality(x) = 1.0 ./ x

barrier_soc(x) = log(x[1]^2 - dot(x[2:end], x[2:end]))
barrier_gradient_soc(x) = 1.0 / (x[1]^2 - dot(x[2:end], x[2:end])) * [x[1]; -x[2:end]]

function barrier_cone(x, idx_ineq, idx_soc)
    Φ = 0.0 
    
    # non-negative orthant
    x_ineq = @views x[idx_ineq]
    Φ -= barrier_inequality(x_ineq) 

    # soc 
    for idx in idx_soc 
        x_soc = @views x[idx] 
        Φ -= barrier_soc(x_soc) 
    end

    return Φ 
end

function barrier_gradient_cone(x, idx_ineq, idx_soc)
    vcat(
        [barrier_gradient_inequality(x[idx_ineq]), 
        [barrier_gradient_soc(x[idx]) for idx in idx_soc]...]...)
end

barrier_cone(x, idx_ineq, idx_soc)
barrier_gradient_cone(x, idx_ineq, idx_soc)

# product 
inequality_product(a, b) = a .* b 
soc_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]

function cone_product(a, b, idx_ineq, idx_soc) 
    vcat(
        [inequality_product(a[idx_ineq], b[idx_ineq]), 
        [soc_product(a[idx], b[idx]) for idx in idx_soc]...]...)
end

inequality_product(x[idx_ineq], y[idx_ineq])
soc_product(x[idx_soc[1]], y[idx_soc[1]])
soc_product(x[idx_soc[2]], y[idx_soc[2]])
cone_product(x, y, idx_ineq, idx_soc)

function inequality_product_jacobian(a, b) 
    Diagonal(b) 
end

function soc_product_jacobian(a, b) 
    n = length(a)
    Diagonal(b[1] * ones(n)) + [0.0 b[2:n]'; b[2:n] zeros(n-1, n-1)]
end

function cone_product_jacobian(a, b, idx_ineq, idx_soc) 
    cat(
        inequality_product_jacobian(a[idx_ineq], b[idx_ineq]),
        [soc_product_jacobian(a[idx], b[idx]) for idx in idx_soc]..., 
    dims=(1, 2))
end

function cone_product_jacobian_inverse(a, b, idx_ineq, idx_soc) 
    cat(
        inv(inequality_product_jacobian(a[idx_ineq], b[idx_ineq])),
        [inv(soc_product_jacobian(a[idx], b[idx])) for idx in idx_soc]..., 
    dims=(1, 2))
end

inequality_product_jacobian(x[idx_ineq], y[idx_ineq])
soc_product_jacobian(x[idx_soc[1]], y[idx_soc[1]])
cone_product_jacobian(x, y, idx_ineq, idx_soc)
cone_product_jacobian(x, y, idx_ineq, idx_soc)

norm(Array(cone_product_jacobian_inverse(x, y, idx_ineq, idx_soc)) - inv(Array(cone_product_jacobian(x, y, idx_ineq, idx_soc))))

# target 
target_inequality(x) = ones(length(x))
target_soc(x) = [1.0; zeros(length(x) - 1)] 

function target_cone(x, idx_ineq, idx_soc) 
    vcat(
        [target_inequality(x[idx_ineq]), 
        [target_soc(x[idx]) for idx in idx_soc]...]...)
end

target_inequality(x[idx_ineq])
target_soc(x[idx_soc[1]])
target_soc(x[idx_soc[2]]) 
target_cone(x, idx_ineq, idx_soc)

# check 
function inequality_violation(x)
    for xi in x 
        xi <= 0.0 && (return true)
    end 
    return false 
end
soc_violation(x) = x[1] <= norm(x[2:end])

function cone_violation(x, idx_ineq, idx_soc) 
    inequality_violation(x[idx_ineq]) && (return true)
    for idx in idx_soc 
        soc_violation(x[idx]) && (return true)
    end
    return false
end

inequality_violation(x[idx_ineq])
soc_violation(x[idx_soc[1]])
soc_violation(ones(5))
cone_violation(x, idx_ineq, idx_soc)
cone_violation(ones(num_variables), idx_ineq, idx_soc)