function residual(g, h, fx, gx, hx, x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx + gx' * y + hx' * z 
    Mr = λ + ρ * r - y 
    Ms = -z - t 
    My = g - r 
    Mz = h - s 
    Mt = s .* t .- κ 

    return Mx, Mr, Ms, My, Mz, Mt
end

function residual_jacobian(gx, hx, fxx, gyxx, hyxx, x, r, s, y, z, t, κ, λ, ρ, n, m, p)
    x_idx = collect(1:n) 
    r_idx = collect(n .+ (1:m))
    s_idx = collect(n + m .+ (1:p))
    y_idx = collect(n + m + p .+ (1:m)) 
    z_idx = collect(n + m + p + m .+ (1:p))
    t_idx = collect(n + m + p + m + p .+ (1:p))

    num_total = n + m + p + m + p + p 

    M = zeros(num_total, num_total) 
    
    # x
    # Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    # Mxx = fxx(x)
    M[x_idx, x_idx] = fxx + gyxx + hyxx
    # Mxr = zeros(n, m)
    # Mxs = zeros(n, p)
    # Mxy = gx(x)'
    M[x_idx, y_idx] = gx'
    # Mxz = hx(x)'
    M[x_idx, z_idx] = hx'
    # Mxt = zeros(n, p)

    # r
    # Mr = λ + ρ * r - y 
    # Mrx = zeros(m, n)
    # Mrr = ρ * I(m)
    M[r_idx, r_idx] = Diagonal(ρ * ones(m))
    # Mrs = zeros(m, p)
    # Mry = -1.0 * I(m)
    M[r_idx, y_idx] = Diagonal(-1.0 * ones(m))
    # Mrz = zeros(m, p)
    # Mrt = zeros(m, p)

    # s
    # Ms = -z - t 
    # Msx = zeros(p, n)
    # Msr = zeros(p, m)
    # Mss = zeros(p, p)
    # Msy = zeros(p, m)
    # Msz = -1.0 * I(p)
    M[s_idx, z_idx] = Diagonal(-1.0 * ones(p))
    # Mst = -1.0 * I(p)
    M[s_idx, t_idx] = Diagonal(-1.0 * ones(p))


    # y
    # My = g(x) - r 
    # Myx = gx(x) 
    M[y_idx, x_idx] = gx
    # Myr = -1.0 * I(m)
    M[y_idx, r_idx] = Diagonal(-1.0 * ones(m))
    # Mys = zeros(m, p)
    # Myy = zeros(m, m)
    # Myz = zeros(m, p)
    # Myt = zeros(m, p)

    # z 
    # Mz = h(x) - s 
    # Mzx = hx(x)
    M[z_idx, x_idx] = hx 
    # Mzr = zeros(p, m)
    # Mzs = -1.0 * I(p)
    M[z_idx, s_idx] = Diagonal(-1.0 * ones(p))
    # Mzy = zeros(p, m)
    # Mzz = zeros(p, p)
    # Mzt = zeros(p, p)

    # t
    # Mt = s .* t .- κ 
    # Mtx = zeros(p, n)
    # Mtr = zeros(p, m)
    # Mts = Diagonal(t)
    M[t_idx, s_idx] = Diagonal(t)
    # Mty = zeros(p, m)
    # Mtz = zeros(p, p)
    # Mtt = Diagonal(s)
    M[t_idx, t_idx] = Diagonal(s)

    return M
   
end