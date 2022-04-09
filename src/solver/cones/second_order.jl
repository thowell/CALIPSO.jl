# indices
idx_soc = [collect(num_nonnegative + sum(num_soc[1:(i-1)]) .+ (1:num_soc[i])) for i = 1:length(num_soc)]

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
second_order_barrier(x) = log(x[1]^2 - dot(x[2:end], x[2:end]))
second_order_barrier_gradient(x) = 1.0 / (x[1]^2 - dot(x[2:end], x[2:end])) * [x[1]; -x[2:end]]

# product 
second_order_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]

function second_order_product_jacobian(a, b) 
    n = length(a)
    Diagonal(b[1] * ones(n)) + [0.0 b[2:n]'; b[2:n] zeros(n-1, n-1)]
end

# target 
second_order_target(x) = [1.0; zeros(length(x) - 1)] 

# violation 
second_order_violation(x) = x[1] <= norm(x[2:end])