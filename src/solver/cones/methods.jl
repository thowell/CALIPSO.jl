struct ConeMethods{B,BX,P,PX,PXI,E}
    barrier::B 
    barrier_gradient::BX 
    product::P 
    product_jacobian::PX 
    produce_jacobian_inverse::PXI 
    target::E
end

function ConeMethods(num_cone, idx_nn, idx_soc) 
    Φ_func, Φa_func, p_func, pa_func, pai_func, t_func = generate_cones(num_cone, idx_nn, idx_soc) 
    return ConeMethods(
        Φ_func, 
        Φa_func, 
        p_func, 
        pa_func, 
        pai_func,
        t_func,
    )
end