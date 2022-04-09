using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

"""
    minimize   v' b
    subject to ||b|| <= μn
"""

n = 3 
m = 1 
p_nn = 0
p_soc = [3] 
p = p_nn + sum(p_soc)
idx_ineq = collect(1:0)
idx_soc = [collect(1:3)]

v = [0.0; 100.0; 100.0] 
μ = 1.0 
γ = 0.0 

obj(x) = transpose(v) * x #+ 1.0e-5 * dot(x, x)
eq(x) = [x[1] - μ * γ]
ineq(x) = x

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

# merit 
function merit(x, r, s, y, z, t, κ, λ, ρ)
    M = 0.0
    
    # objective
    M += f(x)

    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ * dot(r, r)

    # barrier  
    M -= κ * cone_barrier(s, idx_ineq, idx_soc)

    return M
end

function merit_gradient(x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mr = λ + ρ * r 
    Ms = - κ .* cone_barrier_gradient(s, idx_ineq, idx_soc)

    return Mx, Mr, Ms
end

function residual(x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mr = λ + ρ * r - y 
    Ms = -z - t 
    My = g(x) - r 
    Mz = h(x) - s 
    Mt = cone_product(s, t, idx_ineq, idx_soc) - κ .* cone_target(idx_ineq, idx_soc)

    return Mx, Mr, Ms, My, Mz, Mt
end

function residual_jacobian(x, r, s, y, z, t, κ, λ, ρ)
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
    M[x_idx, x_idx] = fxx(x)
    # Mxr = zeros(n, m)
    # Mxs = zeros(n, p)
    # Mxy = gx(x)'
    M[x_idx, y_idx] = gx(x)'
    # Mxz = hx(x)'
    M[x_idx, z_idx] = hx(x)'
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
    M[y_idx, x_idx] = gx(x) 
    # Myr = -1.0 * I(m)
    M[y_idx, r_idx] = Diagonal(-1.0 * ones(m))
    # Mys = zeros(m, p)
    # Myy = zeros(m, m)
    # Myz = zeros(m, p)
    # Myt = zeros(m, p)

    # z 
    # Mz = h(x) - s 
    # Mzx = hx(x)
    M[z_idx, x_idx] = hx(x) 
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
    M[t_idx, s_idx] = cone_product_jacobian(s, t, idx_ineq, idx_soc)
    # Mty = zeros(p, m)
    # Mtz = zeros(p, p)
    # Mtt = Diagonal(s)
    M[t_idx, t_idx] = cone_product_jacobian(t, s, idx_ineq, idx_soc)

    return M
   
end 

x = randn(n) 
r = zeros(m)
s = zeros(p)#max.(h(x), 1.0e-1)
y = zeros(m) 
z = zeros(p) 
t = zeros(p)
κ = 1.0
λ = zeros(m) 
ρ = 1.0

initialize_cone!(s, idx_ineq, idx_soc)
initialize_cone!(t, idx_ineq, idx_soc)

ϵ = 0.0 * 1.0e-5
reg = [ϵ * ones(n); ϵ * ones(m); ϵ * ones(p); - 0.0 * ϵ * ones(m); - 0.0 * ϵ * ones(p); - 0.0 * ϵ * ones(p)]

M = merit(x, r, s, y, z, t, κ, λ, ρ)
M_grad = vcat(merit_gradient(x, r, s, y, z, t, κ, λ, ρ)...)

res = vcat(residual(x, r, s, y, z, t, κ, λ, ρ)...)
jac = residual_jacobian(x, r, s, y, z, t, κ, λ, ρ)

total_iter = 1

for j = 1:10
    for i = 1:100
        println("iter: ($j, $i, $total_iter)")

        M = merit(x, r, s, y, z, t, κ, λ, ρ)
        merit_grad = vcat(merit_gradient(x, r, s, y, z, t, κ, λ, ρ)...)

        res = vcat(residual(x, r, s, y, z, t, κ, λ, ρ)...)
        jac = residual_jacobian(x, r, s, y, z, t, κ, λ, ρ)
        res_norm = norm(res, Inf)

        println("res: $(res_norm)")
        println("cond: $(cond(jac))")

        # check convergence
        if res_norm < 1.0e-4
            break 
        end

        # search direction 
        Δ = (jac + Diagonal(reg)) \ res 

        # line search
        ls_iter = 0
        c1 = 1.0e-4 
        c2 = 1.0e-4
        α = 1.0  
        αt = 1.0
        x̂ = x - α * Δ[1:n] 
        r̂ = r - α * Δ[n .+ (1:m)]
        ŝ = s - α * Δ[n + m .+ (1:p)]
        ŷ = y - α * Δ[n + m + p .+ (1:m)]
        ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
        t̂ = t - αt * Δ[n + m + p + m + p .+ (1:p)] 

        while cone_violation(ŝ, idx_ineq, idx_soc)
            α = 0.5 * α 
            ŝ = s - α * Δ[n + m .+ (1:p)]
        end

        while cone_violation(t̂, idx_ineq, idx_soc)
            αt = 0.5 * αt
            t̂ = t - αt * Δ[n + m + p + m + p .+ (1:p)]
        end

        x̂ = x - α * Δ[1:n] 
        r̂ = r - α * Δ[n .+ (1:m)]
        ŝ = s - α * Δ[n + m .+ (1:p)]
        ŷ = y - α * Δ[n + m + p .+ (1:m)]
        ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
        t̂ = t - αt * Δ[n + m + p + m + p .+ (1:p)]

        θ̂ = norm([g(x̂) - r̂; h(x̂) - ŝ], 1)
        θ = norm([g(x) - r; h(x) - s], 1)

        while merit(x̂, r̂, ŝ, y, z, t, κ, λ, ρ) > M + c1 * α * dot(Δ[1:(n+m+p)], merit_grad) && θ̂ > θ#|| -dot(Δ[1:(n + m + p + m + p)], vcat(merit_grad(x̂, r̂, ŝ, y, z, t, κ, λ, ρ)...)) > -c2 * dot(Δ[1:(n + m + p + m + p)], res_barrier)
            α = 0.5 * α
            x̂ = x - α * Δ[1:n] 
            r̂ = r - α * Δ[n .+ (1:m)]
            ŝ = s - α * Δ[n + m .+ (1:p)]
            ŷ = y - α * Δ[n + m + p .+ (1:m)]
            ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
            t̂ = t - αt * Δ[n + m + p + m + p .+ (1:p)] 
            θ̂ = norm([g(x̂) - r̂; h(x̂) - ŝ], 1)
            ls_iter += 1 
            ls_iter > 25 && error("line search failure")
        end
        println("α = $α")

        # update
        x = x̂
        r = r̂
        s = ŝ 
        y = ŷ
        z = ẑ
        t = t̂
        
        total_iter += 1
        println("con: $(norm(g(x), Inf))")
        println("")
    end

    if norm(g(x), Inf) < 1.0e-3# && norm(cone_product(s, t), Inf) < 1.0e-3 # norm(fx(x) + transpose(gx(x)) * y, Inf) < 1.0e-3
        println("solve success!")
        break 
    end

    # update
    κ = max(1.0e-8, 0.1 * κ)
    λ = λ + ρ * r
    ρ = min(1.0e8, 10 * ρ) 
end

@show x[1] - μ * γ 
@show v[2:3] ./ norm(v[2:3]) 
@show x[2:3] ./ norm(x[2:3])