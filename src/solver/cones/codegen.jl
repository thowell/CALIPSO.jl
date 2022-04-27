function generate_cones(num_cone, idx_nn, idx_soc;
    checkbounds=false,
    threads=false)

    @variables a[1:num_cone] b[1:num_cone]
    # a = Symbolics.variables(:a, 1:num_cone) 
    # b = Symbolics.variables(:b, 1:num_cone) 

    Φ = cone_barrier(a, idx_nn, idx_soc)
    Φa = cone_barrier_gradient(a, idx_nn, idx_soc)

    p = cone_product(a, b, idx_nn, idx_soc) 
    pa = cone_product_jacobian(a, b, idx_nn, idx_soc) 
    pai = cone_product_jacobian_inverse(a, b, idx_nn, idx_soc)
    t = cone_target(idx_nn, idx_soc)

    Φ_func = Symbolics.build_function([Φ], a,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2]
    Φa_func = Symbolics.build_function(Φa, a,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2]

    p_func = Symbolics.build_function(p, a, b,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2]
    pa_func = Symbolics.build_function(pa, a, b,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2] 
    pai_func = Symbolics.build_function(pai, a, b,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2]
    t_func = Symbolics.build_function(t, a, b,
        checkbounds=checkbounds,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()), 
        expression=Val{false})[2]

    return Φ_func, Φa_func, p_func, pa_func, pai_func, t_func
end