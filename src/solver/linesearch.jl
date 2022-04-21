#TODO: add reference
function switching_condition(step_size, search_direction, merit_gradient, merit_exponent, violation, violation_exponent, regularization)
    return (merit_gradient' * search_direction < 0.0 &&
            step_size * (-merit_gradient' * search_direction)^merit_exponent > regularization * violation^violation_exponent)
end

# TODO: add reference
function sufficient_progress(violation, violation_candidate, merit, merit_candidate, violation_tolerance, merit_tolerance, machine_tolerance)
    return (violation_candidate - 10.0 * machine_tolerance * abs(violation) <= (1.0 - violation_tolerance) * violation || 
            merit_candidate - 10.0 * machine_tolerance * abs(merit) <= merit - merit_tolerance * violation)
end

# TODO: add reference
function armijo(merit, merit_candidate, merit_gradient, search_direction, step_size, armijo_tolerance, machine_tolerance)
    return (merit_candidate - merit - 10.0 * machine_tolerance * abs(merit) <= armijo_tolerance * step_size * merit_gradient' * search_direction)
end
