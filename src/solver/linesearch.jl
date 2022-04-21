#TODO: add reference
function switching_condition()
    Mx = s.merit_gradient
    α = s.step_size
    sM = s.options.exponent_merit
    δ = s.options.regularization
    θ = s.constraint_violation
    sθ = s.options.exponent_constraint_violation

    return (Mx' * d < 0.0 && α * (-Mx' * d)^sM > δ * θ^sθ)
end

# TODO: add reference
function sufficient_progress()
    θ_cand = s.constraint_violation_candidate
    θ = s.constraint_violation
    M_cand = s.merit_candidate
    M = s.merit
    γθ = s.options.constraint_violation_tolerance
    γM = s.options.merit_tolerance
    ϵ = s.options.machine_tolerance

    return (θ_cand - 10.0 * ϵ * abs(θ) <= (1.0 - γθ) * θ || M_cand - 10.0 * ϵ * abs(M) <= M - γM * θ)
end

# TODO: add reference
function armijo()
    M_cand = s.merit_candidate
    M = s.merit
    γa = s.options.armijo_tolerance
    α = s.step_size
    Mx = s.merit_gradient
    d = s.dx
    ϵ = s.options.machine_tolerance

    return (M_cand - M - 10.0 * ϵ * abs(M) <= γa * α * Mx' * d)
end
