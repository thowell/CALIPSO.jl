function constraint_violation!(c, g, r, h, s, idx; 
    norm_type=1.0)

    for (i, ii) in enumerate(idx.violation_equality) 
        c[ii] = g[i] - r[i] 
    end 

    for (j, jj) in enumerate(idx.violation_cone) 
        c[jj] = h[j] - s[j] 
    end

    return norm(c, norm_type) / length(c)
end