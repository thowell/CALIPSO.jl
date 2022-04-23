
struct Filter{T}
    pairs::Vector{Tuple{T,T}}
    cache::Vector{Tuple{T,T}}
    index::Vector{Int}
end

function Filter(; max_length=1000) 
    Filter(
        [(1.0e8, 1.0e8) for i = 1:max_length],
        [(1.0e8, 1.0e8) for i = 1:max_length],
        [0],
    ) 
end

function cache_pairs!(filter::Filter) 
    for i = 1:filter.index[1]
        filter.cache[i] = filter.pairs[i]
    end
    return
end

function reset_pairs!(filter::Filter) 
    for i = 1:filter.index[1]
        filter.pairs[i] = (1.0e8, 1.0e8) 
    end
    filter.index[1] = 0
    return
end

function reset_cache!(filter::Filter) 
    for i = 1:filter.index[1]
        filter.cache[i] = (1.0e8, 1.0e8) 
    end
    return
end

function reset!(filter::Filter) 
    reset_cache!(filter) 
    reset_pairs!(filter)
    return
end

"""
    check_filter(constraint_violation, merit, f)

    Check if the constraint residual `constraint_violation` and the barrier objective `merit` are accepted by the filter.
    To be accepted, the pair must be acceptable to each pair stored in the filter.
"""
function check_filter(constraint_violation, merit, filter)
    for f in filter.pairs
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
    if filter.index[1] == 0
        filter.pairs[1] = (constraint_violation, merit)
        filter.index[1] += 1 
        return
    # remove filter points dominated by new point
    elseif check_filter(constraint_violation, merit, filter)
        # cache pairs
        n = filter.index[1] 
        cache_pairs!(filter) 

        # reset filter
        reset_pairs!(filter)

        # augment filter
        filter.index[1] += 1 
        filter.pairs[filter.index[1]] = (constraint_violation, merit)

        # restore minimal filter
        for i = 1:n 
            if !(filter.cache[i][1] >= constraint_violation && filter.cache[i][2] >= merit)
                filter.index[1] += 1
                filter.pairs[filter.index[1]] = filter.cache[i]
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

# filter = Filter(max_length=1000)

# reset!(filter)
# cache_pairs!(filter)

# filter.pairs[1] = (10.0, 1.0)