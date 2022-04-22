
"""
    check_filter(constraint_violation, merit, f)

    Check if the constraint residual `constraint_violation` and the barrier objective `merit` are accepted by the filter.
    To be accepted, the pair must be acceptable to each pair stored in the filter.
"""
function check_filter(constraint_violation, merit, filter)
    for f in filter
        if !(constraint_violation < f[1] || merit < f[2])
            return false
        end
    end
    return true
end

"""
    augment_filter!(constraint_violation, merit, filter)

    Add the pair `(constraint_violation, merit)` to the filter `f` (a vector of pairs)
"""
function augment_filter!(constraint_violation, merit, filter)
    if isempty(filter)
        push!(filter, (constraint_violation, merit))
        return nothing
    # remove filter points dominated by new point
    elseif check_filter(constraint_violation, merit, filter)
        filter_copy = copy(filter)
        empty!(filter)
        push!(filter, (constraint_violation, merit))
        for f in filter_copy
            if !(f[1] >= constraint_violation && f[2] >= merit)
                push!(filter, f)
            end
        end
    end
    return
end

"""
    augment_filter!(solver)

    Check current step, and add to the filter if necessary, adding some padding to the points
    to ensure sufficient decrease (Eq. 18).
"""
function augment_filter!(solver, merit, merit_candidate, merit_gradient, violation, step_size, search_direction)
    if !switching_condition(step_size, search_direction, merit_gradient, solver.options.merit_exponent, violation, solver.options.violation_exponent, 1.0) || !armijo(merit, merit_candidate, merit_gradient, search_direction, step_size, solver.options.armijo_tolerance, solver.options.machine_tolerance)
        augment_filter!(
            (1.0 - solver.options.violation_tolerance) * violation, 
            merit - solver.options.merit_tolerance * violation, solver.data.filter
        )
    end
    return nothing
end

