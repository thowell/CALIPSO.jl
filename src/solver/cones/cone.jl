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
        length(idx_ineq) > 0 ? inv(nonnegative_product_jacobian(a[idx_ineq], b[idx_ineq])) : zeros(0, 0),
        [length(idx) > 0 ? inv(second_order_product_jacobian(a[idx], b[idx])) : zeros(0, 0) for idx in idx_soc]..., 
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
    length(idx_ineq) > 0 && (nonnegative_violation(x̂[idx_ineq], x[idx_ineq], τ[1]) && (return true))
    for idx in idx_soc 
        length(idx) > 0 && (second_order_violation(x̂[idx], x[idx], τ[1]) && (return true))
    end
    return false
end

# evalute
function cone!(problem::ProblemData{T}, methods::ProblemMethods, idx::Indices, solution::Point{T};
    product=false,
    jacobian=false,
    target=false,
    ) where T

    s = solution.cone_slack
    t = solution.cone_slack_dual

    product && (problem.cone_product .= cone_product(s, t, idx.cone_nonnegative, idx.cone_second_order))
    jacobian && (problem.cone_product_jacobian_primal .= cone_product_jacobian(s, t, idx.cone_nonnegative, idx.cone_second_order))
    jacobian && (problem.cone_product_jacobian_dual .= cone_product_jacobian(t, s, idx.cone_nonnegative, idx.cone_second_order))
    target && (problem.cone_target .= cone_target(idx.cone_nonnegative, idx.cone_second_order))

    return
end