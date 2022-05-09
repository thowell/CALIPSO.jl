# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 
using Printf

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra
using RoboDojo

using RoboDojo

# ## Initial conditions
q1 = nominal_configuration(halfcheetah4)
v1 = zeros(halfcheetah4.nq)
q1[2] += 0.25
q1[3] += 0.0 * π

# ## Time
h = 0.05
T = 101

# ## policy 

# ## Simulator
s = Simulator(halfcheetah4, T, h=h)

# ## Simulate
simulate!(s, q1, v1)

# ## Visualizer
vis = Visualizer()
render(vis)

# ## Visualize
visualize!(vis, s)

# ## rollout
x_hist = [[q1; v1]]
u_hist = [zeros(s.model.nu)]
next_state
for t = 1:T
    push!(u_hist, 1.0e-3 * randn(s.model.nu))
    
    RD.dynamics(dynamics_model, y, x_hist[end], u_hist[end], zeros(nw))
    push!(x_hist, y)
end