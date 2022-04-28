struct ConeMethods{T,B,BX,P,PX,PXI,E}
    barrier::B 
    barrier_gradient::BX 
    product::P 
    product_jacobian::PX 
    produce_jacobian_inverse::PXI 
    product_jacobian_cache::Vector{T}
    product_jacobian_inverse_cache::Vector{T}
    product_jacobian_sparsity::Vector{Tuple{Int,Int}}
    product_jacobian_inverse_sparsity::Vector{Tuple{Int,Int}}
    target::E
end

function ConeMethods(num_cone, idx_nn, idx_soc) 
    Φ_func, Φa_func, p_func, pa_func, pai_func, pa_sparsity, pai_sparsity, t_func = generate_cones(num_cone, idx_nn, idx_soc) 
    return ConeMethods(
        Φ_func, 
        Φa_func, 
        p_func, 
        pa_func, 
        pai_func,
        zeros(length(pa_sparsity)),
        zeros(length(pai_sparsity)),
        pa_sparsity,
        pai_sparsity,
        t_func,
    )
end