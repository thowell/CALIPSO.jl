
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
function augment_filter!(solver)
    # if !switching_condition(s.dx, s) || !armijo(s)
    #     augment_filter!(
    #         (1.0 - s.options.constraint_violation_tolerance) * s.constraint_violation, 
    #         s.merit - s.options.merit_tolerance*s.constraint_violation, s.filter
    #     )
    # end
    return nothing
end

