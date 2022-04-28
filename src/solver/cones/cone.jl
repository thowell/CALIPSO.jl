function initialize_cone!(x, idx_ineq, idx_soc) 
    initialize_nonnegative!(x, idx_ineq) 
    initialize_second_order!(x, idx_soc) 
end

# barrier 
function cone_barrier(x, idx_ineq, idx_soc)
    Φ = 0.0 
    
    # nonnegative orthant
    if length(idx_ineq) > 0
        x_ineq = @views x[idx_ineq]
        Φ += nonnegative_barrier(x_ineq) 
    end

    # soc 
    for idx in idx_soc 
        if length(idx) > 0
            x_soc = @views x[idx] 
            Φ += second_order_barrier(x_soc) 
        end
    end

    return Φ 
end

function cone_barrier_gradient(x, idx_ineq, idx_soc)
    vcat(
        [length(idx_ineq) > 0 ? nonnegative_barrier_gradient(x[idx_ineq]) : zeros(0), 
        [length(idx) > 0 ? second_order_barrier_gradient(x[idx]) : zeros(0) for idx in idx_soc]...]...)
end

# product 
function cone_product(a, b, idx_ineq, idx_soc) 
    vcat(
        [length(idx_ineq) > 0 ? nonnegative_product(a[idx_ineq], b[idx_ineq]) : zeros(0), 
        [length(idx) > 0 ? second_order_product(a[idx], b[idx]) : zeros(0) for idx in idx_soc]...]...)
end

function cone_product_jacobian(a, b, idx_ineq, idx_soc) 
    cat(
        length(idx_ineq) > 0 ? nonnegative_product_jacobian(a[idx_ineq], b[idx_ineq]) : zeros(0, 0),
        [length(idx) > 0 ? second_order_product_jacobian(a[idx], b[idx]) : zeros(0, 0) for idx in idx_soc]..., 
    dims=(1, 2))
end

function cone_product_jacobian_inverse(a, b, idx_ineq, idx_soc) 
    cat(
        length(idx_ineq) > 0 ? nonnegative_product_jacobian_inverse(a[idx_ineq], b[idx_ineq]) : zeros(0, 0),
        [length(idx) > 0 ? second_order_product_jacobian_inverse(a[idx], b[idx]) : zeros(0, 0) for idx in idx_soc]..., 
    dims=(1, 2))
end

# target 
function cone_target(idx_ineq, idx_soc) 
    vcat(
        [length(idx_ineq) > 0 ? nonnegative_target(idx_ineq) : zeros(0), 
        [length(idx) > 0 ? second_order_target(idx) : zeros(0) for idx in idx_soc]...]...)
end

# violation 
function cone_violation(x̂, x, τ, idx_ineq, idx_soc) 
    length(idx_ineq) > 0 && (nonnegative_violation(x̂, x, τ[1], idx_ineq) && (return true))
    for idx in idx_soc 
        length(idx) > 0 && (second_order_violation(x̂, x, τ[1], idx) && (return true))
    end
    return false
end

# evalute
function cone!(problem::ProblemData{T}, methods::ConeMethods, idx::Indices, solution::Point{T};
    barrier=false, 
    barrier_gradient=false,
    product=false,
    jacobian=false,
    target=false,
    ) where T

    s = solution.cone_slack
    t = solution.cone_slack_dual

    # dimension
    nc = length(s) 

    # barrier 
    barrier && methods.barrier(problem.barrier, s) 
    barrier_gradient && methods.barrier_gradient(problem.barrier_gradient, s)

    # cone
    (product && nc > 0) && methods.product(problem.cone_product, s, t)
    if (jacobian && nc > 0) 
        methods.product_jacobian(methods.product_jacobian_cache, s, t)
        for (i, idx) in enumerate(methods.product_jacobian_sparsity) 
            problem.cone_product_jacobian_primal[idx...] = methods.product_jacobian_cache[i] 
        end
    end
    if (jacobian && nc > 0) 
        methods.product_jacobian(methods.product_jacobian_cache, t, s)
        for (i, idx) in enumerate(methods.product_jacobian_sparsity)
            problem.cone_product_jacobian_dual[idx...] = methods.product_jacobian_cache[i] 
        end
    end
    (target && nc > 0) && methods.target(problem.cone_target, s, t)

    return
end