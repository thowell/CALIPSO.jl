function Solver(objective, dynamics, num_states::Vector{Int}, num_actions::Vector{Int};
    equality=[empty_constraint for t = 1:length(num_states)],
    nonnegative=[empty_constraint for t = 1:length(num_states)],
    second_order=[[empty_constraint] for t = 1:length(num_states)],
    parameters=[zeros(0) for t = 1:length(num_states)],
    options=Options(),
    )

    # horizon
    H = length(num_states) 

    # check inputs 
    # TODO: add more checks
    @assert length(num_actions) ==  H - 1
    @assert length(objective) ==    H 
    @assert length(dynamics) ==     H - 1
    @assert length(equality) ==     H
    @assert length(nonnegative) ==  H
    @assert length(second_order) == H
    @assert length(parameters) ==   H

    # parameters 
    num_parameters = [length(p) for p in parameters]
     
    # objective
    obj = generate_methods(objective, num_states, num_actions, num_parameters, :Cost;
        checkbounds=options.codegen_checkbounds,
        constraint_tensor=options.constraint_tensor);

    # dynamics 
    dyn = generate_methods(dynamics, num_states, num_actions, num_parameters, :Dynamics;
        checkbounds=options.codegen_checkbounds,
        constraint_tensor=options.constraint_tensor);

    # equality 
    eq = generate_methods(equality, num_states, num_actions, num_parameters, :Constraint;
        checkbounds=options.codegen_checkbounds,
        constraint_tensor=options.constraint_tensor);

    # nonnegative 
    nn = generate_methods(nonnegative, num_states, num_actions, num_parameters, :Constraint;
        checkbounds=options.codegen_checkbounds,
        constraint_tensor=options.constraint_tensor);
 
    # second order
    # TODO: more efficient generate method 
    so = [[Constraint(s, num_states[t], t == H ? 0 : num_actions[t]; 
        num_parameter=num_parameters[t],
        constraint_tensor=options.constraint_tensor) for s in second_order[t]] for t = 1:H]

    # trajectory optimization problem
    trajopt = TrajectoryOptimizationProblem(dyn, obj, eq, nn, so;
        parameters=parameters)

    # CALIPSO methods
    methods = ProblemMethods(trajopt)

    # cone indices
    nonnegative_indices, second_order_indices = cone_indices(trajopt)

    # solver 
    solver = Solver(methods, 
        trajopt.dimensions.total_variables, 
        trajopt.dimensions.total_parameters, 
        trajopt.dimensions.total_equality, 
        trajopt.dimensions.total_cone,
        parameters=vcat(parameters...),
        nonnegative_indices=nonnegative_indices, 
        second_order_indices=second_order_indices,
        custom=trajopt,
        options=options,
    )

    return solver 
end

function initialize_states!(solver::Solver, states) 
    for (t, idx) in enumerate(solver.problem.custom.indices.states)
        solver.solution.variables[idx] = states[t]
    end
end

function initialize_controls!(solver::Solver, actions) 
    for (t, idx) in enumerate(solver.problem.custom.indices.actions)
        solver.solution.variables[idx] = actions[t]
    end
end

function get_trajectory(solver::Solver) 
    states = [solver.solution.variables[idx] for (t, idx) in enumerate(solver.problem.custom.indices.states)]
    actions = [solver.solution.variables[idx] for (t, idx) in enumerate(solver.problem.custom.indices.actions)] 
    return states, actions
end

function generate_methods(func::Vector, num_states::Vector{Int}, num_actions::Vector{Int}, num_parameters::Vector{Int}, mode::Symbol;
    checkbounds=true,
    constraint_tensor=true)

    # trajectory length
    horizon = length(num_states)

    # pairs 
    if mode == :Dynamics
        pairs = [(func[t], num_states[t], num_actions[t], num_states[t+1], num_parameters[t]) for t = 1:horizon-1]
         # get indices for unique pairs
        indices_unique = unique(i -> pairs[i], 1:horizon-1)
    else
        pairs = [(func[t], num_states[t],  t == horizon ? 0 : num_actions[t], num_parameters[t]) for t = 1:horizon] 
        # get indices for unique pairs
        indices_unique = unique(i -> pairs[i], 1:horizon)
    end

    # get unique pairs
    pairs_unique = unique(pairs)
    
    # codegen unique pairs
    if mode == :Dynamics
        f_unique = [eval(mode)(pairs_unique[i][1], pairs_unique[i][4], pairs_unique[i][2], pairs_unique[i][3], 
            num_parameter=pairs_unique[i][5],
            checkbounds=checkbounds,
            constraint_tensor=constraint_tensor) for (i, t) in enumerate(indices_unique)];
    else
        f_unique = [eval(mode)(pairs_unique[i][1], pairs_unique[i][2], pairs_unique[i][3], 
            num_parameter=pairs_unique[i][4],
            checkbounds=checkbounds,
            constraint_tensor=constraint_tensor) for (i, t) in enumerate(indices_unique)];
    end

    # initialize trajectory
    f = [f_unique[1]];

    for pt in pairs[2:end]
        for (i, fu) in enumerate(pairs_unique) 
            if fu == pt
                push!(f, f_unique[i]) 
                break
            end
        end
    end

    return f 
end

empty_constraint(x, u, θ) = zeros(0) 