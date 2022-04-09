function constraint_violation!(c, g, r, h, s, idx; 
    norm_type=1.0)

    c[idx.violation_equality] = g 
    c[idx.violation_equality] -= r 
    c[idx.violation_cone] = h 
    c[idx.violation_cone] -= s

    return norm(c, norm_type)
end