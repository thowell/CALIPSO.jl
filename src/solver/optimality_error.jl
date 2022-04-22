function optimality_error(solution, residual, indices)
    # variables 
    y = solution.equality_dual 
    z = solution.cone_dual 
    t = solution.cone_slack_dual 

    # scaling 
    sd = length(y) + length(z) > 0 ? max(100.0, (norm(y, 1) + norm(z, 1)) / (length(y) + length(z))) / 100.0 : 1.0
    sc = length(t) > 0 ? max(100.0, norm(t, 1) / length(t)) / 100.0 : 1.0

    # Lagrangian gradient 
    lag_grad = residual.primals

    # constraint slack 
    equality = residual.equality_dual
    cone     = residual.cone_dual

    # complementarity 
    comp = residual.cone_slack_dual

    return max(
                norm(lag_grad, Inf) / sd,
                norm(equality, Inf),
                norm(cone, Inf), 
                norm(comp, Inf) / sc,
    )
end

