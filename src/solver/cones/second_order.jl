# initialize
function initialize_second_order!(x, idx_soc; 
    first_initial=1.0, initial=0.1)
    for idx in idx_soc
        for (i, ii) in enumerate(idx)
            x[ii] = (i == 1 ? first_initial : initial)
        end 
    end 
    return 
end

# barrier 
second_order_barrier(x) = 0.5 * log(x[1]^2 - dot(x[2:end], x[2:end]))
second_order_barrier_gradient(x) = 1.0 / (x[1]^2 - dot(x[2:end], x[2:end])) * [x[1]; -x[2:end]]

# product 
second_order_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]

function second_order_product_jacobian(a, b) 
    n = length(a)
    Diagonal(b[1] * ones(n)) + [0.0 b[2:n]'; b[2:n] zeros(n-1, n-1)]
end

function second_order_product_jacobian_inverse(a, u)
    n = length(a)
    α = -1 / u[1]^2 * dot(u[2:end], u[2:end])
    β = 1 / (1 + α)
    S1 = zeros(eltype(u), n, n)
    S1[end, 1:end-1] = u[end:-1:2] / u[1]
    S2 = zeros(eltype(u), n, n)
    S2[1:end-1, end] = u[end:-1:2] / u[1]
    P = zeros(eltype(u), n, n)
    for i = 1:n
        P[end-i+1, i] = 1
    end
    Vi = (I - S1) * (I - β * (S2 * (I - S1)))
    Ui = P * 1 / u[1] * Vi * P
    return Ui
end

# target 
second_order_target(x) = [1.0; zeros(length(x) - 1)] 

# violation 
function second_order_violation(x̂, x, τ, idx) 
    x̂[idx[1]] - (1.0 - τ) * x[idx[1]] <= norm(x̂[idx[2:end]] - (1.0 - τ) * x[idx[2:end]])
end


# function jacobian(u)
#     U = u[1] * Array(Diagonal(ones(n)))
#     U[2:end,1] = u[2:end]
#     U[1,2:end] = u[2:end]
#     return U
# end

# function inverse(u)
#     n = length(u)
#     α = -1 / u[1]^2 * dot(u[2:end], u[2:end])
#     β = 1 / (1 + α)
#     S1 = zeros(eltype(u), n, n)
#     S1[end, 1:end-1] = u[end:-1:2] / u[1]
#     S2 = zeros(eltype(u), n, n)
#     S2[1:end-1, end] = u[end:-1:2] / u[1]
#     P = zeros(eltype(u), n, n)
#     for i = 1:n
#         P[end-i+1, i] = 1
#     end
#     Vi = (I - S1) * (I - β * (S2 * (I - S1)))
#     Ui = P * 1 / u[1] * Vi * P
#     return Ui
# end

# @variables u[1:3] 
# jacobian(u)
# inverse(u)
# function inverse(u,x)
#     n = length(u)
#     α = -1/u[1]^2 * norm(u[2:end])^2
#     β = 1 / (1 + α)

#     x0 = x - [u[2:end]'*x[2:end]; zeros(n-1)]
#     x1 = x - β * [0; u[2:end] * x0[1]]
#     x2 = x1 - [u[2:end]'*x1[2:end]; zeros(n-1)]
#     x = 1/u[1] * x2
#     return xi
# end


# n = 3
# x = rand(n)
# u = rand(n)
# U = jacobian(u)
# U * inverse(u) * x - x
# U * inverse(u,x) - x