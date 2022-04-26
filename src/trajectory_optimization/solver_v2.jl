function Solver(
    objective, 
    dynamics, 
    equality, 
    nonnegative, 
    second_order, 
    num_states, 
    num_actions, 
    num_parameters;
    parameters=[zeros(p) for p in num_parameters],
    options=Options())

    # check problem 
    check_trajectory_optimization(
        objective, 
        dynamics, 
        equality, 
        nonnegative, 
        second_order, 
        num_states, 
        num_actions, 
        num_parameters)

    # generate 
    o, oz, oθ, ozz, ozθ, OZZ, OZθ,ozz_s, ozθ_s, e, ez, eθ, EZ, Eθ, ez_s, eθ_s, ey, eyz, eyzz, eyzθ, EYZZ, EYZθ, eyzz_s, eyzθ_s, c, cz, cθ, CZ, Cθ, cz_s, cθ_s, cy, cyz, cyzz, cyzθ, CYZZ, CYZθ, cyzz_s, cyzθ_s, nn_idx, so_idx, nz, np, ne, nc = generate_trajectory_optimization(objective, dynamics, equality, nonnegative, second_order, num_states, num_actions, num_parameters; threads=options.codegen_threads, checkbounds=options.codegen_checkbounds, verbose=options.verbose)
        
    m = ProblemMethods(
        o, oz, oθ, ozz, ozθ, OZZ, OZθ,ozz_s, ozθ_s, e, ez, eθ, EZ, Eθ, ez_s, eθ_s, ey, eyz, eyzz, eyzθ, EYZZ, EYZθ, eyzz_s, eyzθ_s, c, cz, cθ, CZ, Cθ, cz_s, cθ_s, cy, cyz, cyzz, cyzθ, CYZZ, CYZθ, cyzz_s, cyzθ_s
    )

    options.verbose && println("generating trajectory optimization problem: success")
    
    # return m, nn_idx, so_idx, nz, np, ne, nc
    # CALIPSO solver 
    solver = Solver(m, nz, np, ne, nc,
        parameters=vcat(parameters...),
        nonnegative_indices=nn_idx, 
        second_order_indices=so_idx,
        options=options)

    options.verbose && println("solver generation: success")

    return solver 
end

function check_trajectory_optimization(
        objective, 
        dynamics, 
        equality, 
        nonnegative, 
        second_order, 
        num_states, 
        num_actions, 
        num_parameters)

    #TODO: implement
    return true 
end

