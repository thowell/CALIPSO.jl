cone_product(a, b) = [dot(a, b); a[1] * b[2:end] + b[1] * a[2:end]]

function cone_product!(a, z, s; reset=false) 
    reset && fill!(a, 0.0)
    n = length(a)
    a[1] += z' * s
    for i = 2:n
        a[i] += z[1] * s[i]
        a[i] += s[1] * z[i]
    end
    return nothing
end