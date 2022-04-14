using Pkg 
Pkg.activate(joinpath(@__DIR__, "..")) 
Pkg.instantiate()
using CALIPSO 
using LinearAlgebra

include("acrobot.jl")
n = trajopt.dimensions.total_variables 
m = trajopt.dimensions.equality
p = trajopt.dimensions.cone

# merit 
function merit(x, r, s, y, z, t, κ, λ, ρ)
    M = 0.0
    
    # objective
    M += f(x)

    # augmented lagrangian 
    M += dot(λ, r) + 0.5 * ρ * dot(r, r)

    # barrier  
    M -= κ * sum(log.(s))

    return M
end

function merit_gradient(x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mr = λ + ρ * r 
    Ms = - κ ./ s

    return Mx, Mr, Ms
end

function residual(x, r, s, y, z, t, κ, λ, ρ)
    Mx = fx(x) + gx(x)' * y + hx(x)' * z 
    Mr = λ + ρ * r - y 
    Ms = -z - t 
    My = g(x) - r 
    Mz = h(x) - s 
    Mt = s .* t .- κ 

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
    M[t_idx, s_idx] = Diagonal(t)
    # Mty = zeros(p, m)
    # Mtz = zeros(p, p)
    # Mtt = Diagonal(s)
    M[t_idx, t_idx] = Diagonal(s)

    return M
   
end 

# x = randn(n) 
r = g(x)
s = ones(p)#max.(h(x), 1.0e-1)
y = zeros(m) 
z = zeros(p) 
t = ones(p)
κ = 1.0
λ = zeros(m) 
ρ = 1.0e6

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
        Δ = jac \ res 

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

        while any(ŝ .<= 0)
            α = 0.5 * α 
            ŝ = s - α * Δ[n + m .+ (1:p)]
        end

        while any(t̂ .<= 0.0) 
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

    if norm(g(x), Inf) < 1.0e-3 && norm(s .* t, Inf) < 1.0e-3 # norm(fx(x) + transpose(gx(x)) * y, Inf) < 1.0e-3
        println("solve success!")
        break 
    end

    # update
    κ = max(1.0e-8, 0.1 * κ)
    λ = λ + ρ * r
    ρ = min(1.0e8, 10 * ρ) 
end

using Plots
x_sol = [x[idx] for idx in trajopt.indices.states]
u_sol = [x[idx] for idx in trajopt.indices.actions]

plot(hcat(x_sol...)', xlabel="time step", ylabel="state")
plot(hcat(u_sol...)', xlabel="time step", ylabel="action")

norm(g(x), Inf)