using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

include("qp.jl")

# merit 
function merit(x, r, s, y, z, t, κ, λ, ρ)
    M = 0.0
    # objective
    M += f(x)
    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ * dot(r, r) 
    # equality
    M += dot(y, g(x) - r) 
    # log barrier
    M -= κ * sum(log.(s)) 
    # inequality 
    M += dot(z, h(x) - s) 
    # inequality complementarity
    # M -= dot(s, t) 
    # M -= κ * sum(log.(t)) 

    return M
end

function merit_gradient(x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mr = λ + ρ * r - y 
    Ms = -z - t 
    My = g(x) - r 
    Mz = h(x) - s 
    Mt = s .* t .- κ 

    return Mx, Mr, Ms, My, Mz, Mt
end

function merit_hessian(x, r, s, y, z, t, κ, λ, ρ)
    # x
    # Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mxx = fxx(x)
    Mxr = zeros(n, m)
    Mxs = zeros(n, p)
    Mxy = gx(x)'
    Mxz = hx(x)'
    Mxt = zeros(n, p)

    # r
    # Mr = λ + ρ * r - y 
    Mrx = zeros(m, n)
    Mrr = ρ * I(m)
    Mrs = zeros(m, p)
    Mry = -1.0 * I(m)
    Mrz = zeros(m, p)
    Mrt = zeros(m, p)

    # s
    # Ms = -z - t 
    Msx = zeros(p, n)
    Msr = zeros(p, m)
    Mss = zeros(p, p)
    Msy = zeros(p, m)
    Msz = -1.0 * I(p)
    Mst = -1.0 * I(p)

    # y
    # My = g(x) - r 
    Myx = gx(x) 
    Myr = -1.0 * I(m)
    Mys = zeros(m, p)
    Myy = zeros(m, m)
    Myz = zeros(m, p)
    Myt = zeros(m, p)

    # z 
    # Mz = h(x) - s 
    Mzx = hx(x)
    Mzr = zeros(p, m)
    Mzs = -1.0 * I(p)
    Mzy = zeros(p, m)
    Mzz = zeros(p, p)
    Mzt = zeros(p, p)

    # t
    # Mt = s .* t .- κ 
    Mtx = zeros(p, n)
    Mtr = zeros(p, m)
    Mts = Diagonal(t)
    Mty = zeros(p, m)
    Mtz = zeros(p, p)
    Mtt = Diagonal(s)

    [
        Mxx Mxr Mxs Mxy Mxz Mxt;
        Mrx Mrr Mrs Mry Mrz Mrt;
        Msx Msr Mss Msy Msz Mst;
        Myx Myr Mys Myy Myz Myt;
        Mzx Mzr Mzs Mzy Mzz Mzt;
        Mtx Mtr Mts Mty Mtz Mtt;
    ]
end 

M = merit(x, r, s, y, z, t, κ, λ, ρ)
res = vcat(merit_gradient(x, r, s, y, z, t, κ, λ, ρ)...)
jac = merit_hessian(x, r, s, y, z, t, κ, λ, ρ)

x = randn(n) 
r = g(x)
s = ones(p)#max.(h(x), 1.0e-1)
y = zeros(m) 
z = zeros(p) 
t = ones(p)
κ = 1.0
λ = zeros(m) 
ρ = 1.0 

total_iter = 1
for j = 1:10
    for i = 1:100
        println("iter: ($j, $i, $total_iter)")

        M = merit(x, r, s, y, z, t, κ, λ, ρ)
        res = vcat(merit_gradient(x, r, s, y, z, t, κ, λ, ρ)...)
        jac = merit_hessian(x, r, s, y, z, t, κ, λ, ρ)
        res_norm = norm(res, Inf)

        println("res: $(res_norm)")
        println("cond: $(cond(jac))")

        # check convergence
        if res_norm < 1.0e-4
            break 
        end

        # search direction 
        Δ = jac \ res 

        # line search
        ls_iter = 0
        c1 = 1.0e-4 
        c2 = 0.0
        α = 1.0  
        x̂ = x - α * Δ[1:n] 
        r̂ = r - α * Δ[n .+ (1:m)]
        ŝ = s - α * Δ[n + m .+ (1:p)]
        ŷ = y - α * Δ[n + m + p .+ (1:m)]
        ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
        t̂ = t - α * Δ[n + m + p + m + p .+ (1:p)] 

        while any(ŝ .<= 0) || any(t̂ .<= 0.0) 
            α = 0.5 * α 
            # x̂ = x - α * Δ[1:n] 
            # r̂ = r - α * Δ[n .+ (1:m)]
            ŝ = s - α * Δ[n + m .+ (1:p)]
            # ŷ = y - α * Δ[n + m + p .+ (1:m)]
            # ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
            t̂ = t - α * Δ[n + m + p + m + p .+ (1:p)]
        end

        x̂ = x - α * Δ[1:n] 
        r̂ = r - α * Δ[n .+ (1:m)]
        # ŝ = s - α * Δ[n + m .+ (1:p)]
        ŷ = y - α * Δ[n + m + p .+ (1:m)]
        ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
        # t̂ = t - α * Δ[n + m + p + m + p .+ (1:p)]

        while merit(x̂, r̂, ŝ, ŷ, ẑ, t̂, κ, λ, ρ) > M + c1 * α * dot(Δ, res) || -dot(Δ, vcat(merit_gradient(x̂, r̂, ŝ, ŷ, ẑ, t̂, κ, λ, ρ)...)) > -c2 * dot(Δ, res)
            α = 0.5 * α
            x̂ = x - α * Δ[1:n] 
            r̂ = r - α * Δ[n .+ (1:m)]
            ŝ = s - α * Δ[n + m .+ (1:p)]
            ŷ = y - α * Δ[n + m + p .+ (1:m)]
            ẑ = z - α * Δ[n + m + p + m .+ (1:p)] 
            t̂ = t - α * Δ[n + m + p + m + p .+ (1:p)] 
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
  
    if norm(g(x), Inf) < 1.0e-3 && norm(s .* t, Inf) < 1.0e-3 # norm(fx(x) + transpose(gx(x)) * y, Inf) < 1.0e-3
        println("solve success!")
        break 
    end
    # update
    κ = 0.1 * κ
    λ = λ + ρ * r
    ρ = 10 * ρ 
end



