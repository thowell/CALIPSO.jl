function optimality_error(residual, indices)
    # Lagrangian gradient = 
    lag_grad = @views residual[indices.primals] 

    # constraint slack 
    equality = @views residual[indices.equality_dual] 
    cone     = @views residual[indices.cone_dual]

    # complementarity 
    comp = @views residual[indices.cone_slack_dual]

    return max(
                norm(lag_grad, Inf),
                norm(equality, Inf),
                norm(cone, Inf), 
                norm(comp, Inf),
    )
end

