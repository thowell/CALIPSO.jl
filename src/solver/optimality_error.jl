function optimality_error(variables, residual, indices)
    # variables 
    y = @views variables[indices.equality_dual] 
    z = @views variables[indices.cone_dual] 
    t = @views variables[indices.cone_slack_dual] 

    # scaling 
    sd = length(y) + length(z) > 0 ? max(100.0, (norm(y, 1) + norm(z, 1)) / (length(y) + length(z))) / 100.0 : 1.0
    sc = length(t) > 0 ? max(100.0, norm(t, 1) / length(t)) / 100.0 : 1.0

    # Lagrangian gradient 
    lag_grad = @views residual[indices.primals] 

    # constraint slack 
    equality = @views residual[indices.equality_dual] 
    cone     = @views residual[indices.cone_dual]

    # complementarity 
    comp = @views residual[indices.cone_slack_dual]

    return max(
                norm(lag_grad, Inf) / sd,
                norm(equality, Inf),
                norm(cone, Inf), 
                norm(comp, Inf) / sc,
    )
end

