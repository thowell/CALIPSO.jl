 

function generate_trajectory_optimization(objective, dynamics, equality, nonnegative, second_order, num_states, num_actions, num_parameters;
    checkbounds=true)

    # dimensions
    nz = sum(num_states) + sum(num_actions)
    np = sum(num_parameters)
    T = length(num_states) 
    @assert length(num_actions) == T - 1

    # indices 
    x_idx, u_idx = state_action_indices(num_states, num_actions)
    p_idx = parameter_indices(num_parameters) 

    # variables
    @variables z[1:nz] p[1:sum(np)] 
    x = [z[idx] for idx in x_idx]
    u = [z[idx] for idx in u_idx]
    θ = [p[idx] for idx in p_idx]

    # symbolic expressions
    o = [objective[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
    d = [dynamics[t](x[t+1], x[t], u[t], θ[t]) for t = 1:T-1]
    e = [equality[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
    nn = [nonnegative[t](x[t], t == T ? zeros(0) : u[t], θ[t]) for t = 1:T]
    s = [[so(x[t], t == T ? zeros(0) : u[t], θ[t]) for so in second_order[t]] for t = 1:T]

    # constraint dimensions 
    dynamics_dimensions = [length(dt) for dt in d]
    equality_dimensions = [length(et) for et in e]
    nonnegative_dimensions = [length(nnt) for nnt in nn]
    second_order_dimensions = [[length(so) for so in st] for st in s]

    # assemble 
    os = sum(o)
    es = vcat(d..., e...)
    cs = vcat(nn..., (s...)...)

    ## derivatives 

    # objective
    oz = Symbolics.gradient(os, z)
    oθ = Symbolics.gradient(os, θ)
    ozz = Symbolics.sparsejacobian(oz, z)
    ozθ = Symbolics.sparsejacobian(oz, θ)

    # equality
    ez = Symbolics.sparsejacobian(es, z)
    eθ = Symbolics.sparsejacobian(es, θ)

    # cone
    cz = Symbolics.sparsejacobian(cs, z)
    cθ = Symbolics.sparsejacobian(cs, θ)

    ## product derivatives
    @variables ye[1:length(es)] yc[1:length(cs)]

    # equality
    ey = dot(es, ye)
    eyz = Symbolics.gradient(ey, z)
    eyzz = Symbolics.sparsejacobian(eyz, z)
    eyzθ = Symbolics.sparsejacobian(eyz, θ)

    # cone
    cy = dot(cs, yc)
    cyz = Symbolics.gradient(cy, z)
    cyzz = Symbolics.sparsejacobian(cyz, z)
    cyzθ = Symbolics.sparsejacobian(cyz, θ)

    # sparsity
    ozz_sparsity = collect(zip([findnz(ozz)[1:2]...]...))
    ozθ_sparsity = collect(zip([findnz(ozθ)[1:2]...]...))

    ez_sparsity = collect(zip([findnz(ez)[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(eθ)[1:2]...]...))
    
    cz_sparsity = collect(zip([findnz(cz)[1:2]...]...))
    cθ_sparsity = collect(zip([findnz(cθ)[1:2]...]...))

    eyzz_sparsity = collect(zip([findnz(eyzz)[1:2]...]...))
    eyzθ_sparsity = collect(zip([findnz(eyzθ)[1:2]...]...))

    cyzz_sparsity = collect(zip([findnz(cyzz)[1:2]...]...))
    cyzθ_sparsity = collect(zip([findnz(cyzθ)[1:2]...]...))

    ## build functions

    # objective
    o_func = Symbolics.build_function([os], z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    oz_func = Symbolics.build_function(oz, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    oθ_func = Symbolics.build_function(oθ, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    ozz_func = Symbolics.build_function(ozz.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    ozθ_func = Symbolics.build_function(ozθ.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    # equality
    e_func = Symbolics.build_function(es, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    ez_func = Symbolics.build_function(ez.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    eθ_func = Symbolics.build_function(eθ.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    
    ey_func = Symbolics.build_function([ey], z, p, ye, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    eyz_func = Symbolics.build_function(eyz, z, p, ye, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    eyzz_func = Symbolics.build_function(eyzz.nzval, z, p, ye, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    eyzθ_func = Symbolics.build_function(eyzθ.nzval, z, p, ye, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    # cone 
    c_func = Symbolics.build_function(cs, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cz_func = Symbolics.build_function(cz.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cθ_func = Symbolics.build_function(cθ.nzval, z, p, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    cy_func = Symbolics.build_function([cy], z, p, yc, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cyz_func = Symbolics.build_function(cyz, z, p, yc, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cyzz_func = Symbolics.build_function(cyzz.nzval, z, p, yc, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]
    cyzθ_func = Symbolics.build_function(cyzθ.nzval, z, p, yc, 
        checkbounds=checkbounds, 
        expression=Val{false})[2]

    # methods 
    methods = ProblemMethods(
        o_func,
        oz_func,
        oθ_func, 
        ozz_func, 
        ozθ_func, 
        zeros(length(ozz_sparsity)), zeros(length(ozθ_sparsity)),
        ozz_sparsity, ozθ_sparsity,
        e_func,
        ez_func, 
        eθ_func, 
        zeros(length(ez_sparsity)), zeros(length(eθ_sparsity)),
        ez_sparsity, eθ_sparsity,
        ey_func, 
        eyz_func, 
        eyzz_func, 
        eyzθ_func, 
        zeros(length(eyzz_sparsity)), zeros(length(eyzθ_sparsity)),
        eyzz_sparsity, eyzθ_sparsity,
        c_func, 
        cz_func, 
        cθ_func, 
        zeros(length(cz_sparsity)), zeros(length(cθ_sparsity)),
        cz_sparsity, cθ_sparsity,
        cy_func, 
        cyz_func,
        cyzz_func, 
        cyzθ_func, 
        zeros(length(cyzz_sparsity)), zeros(length(cyzθ_sparsity)),
        cyzz_sparsity, cyzθ_sparsity,
    )

    # cone indices 
    nn_idx = constraint_indices(nonnegative_dimensions)

    so_idx = constraint_indices(second_order_dimensions, 
        shift=sum(nonnegative_dimensions))
   
    return methods, nn_idx, so_idx, nz, np, length(es), length(cs)
end

