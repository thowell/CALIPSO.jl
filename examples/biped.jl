# CALIPSO
using Pkg 
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO 

# Examples
Pkg.activate(@__DIR__) 
Pkg.instantiate()
using LinearAlgebra

# PREAMBLE

# PKG_SETUP

# ## Setup

using RoboDojo 

# ## Initial conditions
model = RoboDojo.biped
function initial_configuration(model, θ_torso, θ_thigh_1, θ_leg_1, θ_thigh_2)
    q1 = zeros(model.nq)
    q1[3] = θ_torso
    q1[4] = θ_thigh_1
    q1[5] = θ_leg_1
    z1 = model.l_thigh1 * cos(q1[4]) + model.l_calf1 * cos(q1[5])
    q1[7] = θ_thigh_2
    q1[8] = 1.0 * acos((z1 - model.l_thigh2 * cos(q1[7])) / model.l_calf2)
	q1[2] = z1
	q1[6] = 0.0 
	q1[9] = 0.0

    return q1
end

q1 = initial_configuration(model, -π / 50.0, -π / 6.5, π / 10.0 , -π / 10.0)
v1 = zeros(model.nq)

# ## Time
h = 0.01
T = 100

# ## Simulator
s = Simulator(model, T, h=h) 

# ## Simulate
status = simulate!(s, q1, v1)

# ## Visualizer
vis = Visualizer()
render(vis)

# ## Visualize
visualize!(vis, s)