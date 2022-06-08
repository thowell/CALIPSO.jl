# Solver

CALIPSO is a solver for non-convex optimization problems designed for robotics applications with conic and complementarity constraints. Additionally, solutions are differentiable with respect to problem data provided to the solver.

## Standard form
Problems:

```math
\begin{align*}
\underset{x}{\text{minimize}} & \quad c(x; \theta) \\
\text{subject to} & \quad  g(x; \theta) = 0, \\
                  & \quad  h(x; \theta) \in \mathcal{K}, \\
\end{align*}
```

with 

- ``x \in \mathbf{R}^n``: decision variables  
- ``\theta \in \mathbf{R}^d``: problem data  

are optimized for 

- ``c : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}``: objective 
- ``g : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}^m``: equality constraints
- ``h : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}^p``:  cone constraints 

The cone,

```math 
\mathcal{K} = \mathbf{R}_{++}^q \times Q_{l_1}^{(1)} \times \dots \times Q_{l_j}^{(j)},
```

is the Cartesian product of the ``q``-dimensional nonnegative orthant and ``j`` second-order cones, each of dimension ``l_i``.

### Non-convex problem example 
In the following example, we formulate and solve the [Wächter problem](http://users.iems.northwestern.edu/~andreasw/pubs/waechter_thesis.pdf) that motivated many of the algorithms and heuristics developed for [Ipopt](https://coin-or.github.io/Ipopt/):

```julia
using CALIPSO

# problem
objective(x) = x[1]
equality(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
cone(x) = x[2:3]

# variables 
num_variables = 3

# solver
solver = Solver(objective, equality, cone, num_variables);

# initialize
x0 = [-2.0, 3.0, 1.0]
initialize!(solver, x0)

# solve 
solve!(solver)

# solution 
solver.solution.variables # x* = [1.0, 0.0, 0.5]
```

## Trajectory optimization 

Deterministic Markov Decision Processes,

```math
\begin{align*}
		\underset{X_{1:T}, \phantom{\,} U_{1:T-1}}{\text{minimize }} & C_T(X_T; \theta_T) + \sum \limits_{t = 1}^{T-1} C_t(X_t, U_t; \theta_t) \\
		\text{subject to } & F_t(X_t, U_t; \theta_t) = X_{t+1}, \quad t = 1,\dots,T-1, \\
		& E_t(X_t, U_t; \theta_t) = 0, \phantom{\, _{t+1}} \quad t = 1, \dots, T, \\
		& H_t(X_t, U_t; \theta_t) \in \mathcal{K}_t, \phantom{X} \quad t = 1, \dots, T,
\end{align*}
```

with 

- ``X_t \in \mathbf{R}^{n_t}``: state
- ``U_t \in \mathbf{R}^{m_t}``: action
- ``\theta_t \in \mathbf{R}^{d_t}``: problem data 

and 
- ``C_t : \mathbf{R}^{n_t} \times \mathbf{R}^{m_t} \times \mathbf{R}^{d_t} \rightarrow \mathbf{R}``: stage cost
- ``F_t : \mathbf{R}^{n_t} \times \mathbf{R}^{m_t} \times \mathbf{R}^{d_t} \rightarrow \mathbf{R}^{n_{t+1}}``: discrete-time dynamics 
- ``E_t : \mathbf{R}^{n_t} \times \mathbf{R}^{m_t} \times \mathbf{R}^{d_t} \rightarrow \mathbf{R}^{e_t}``: equality constraint
- ``H_t : \mathbf{R}^{n_t} \times \mathbf{R}^{m_t} \times \mathbf{R}^{d_t} \rightarrow \mathbf{R}^{h_t}``: cone constraints 

are automatically transcribed and optimized with CALIPSO.

### Pendulum swing-up example
In the following example, we optimize state and action trajectories in order to perform a swing-up with a pendulum system:

```julia
using CALIPSO 
using LinearAlgebra 

# horizon 
horizon = 11 

# dimensions 
num_states = [2 for t = 1:horizon]
num_actions = [1 for t = 1:horizon-1] 

# dynamics
function pendulum_continuous(x, u)
   mass = 1.0
   length_com = 0.5
   gravity = 9.81
   damping = 0.1

   [
      x[2],
      (u[1] / ((mass * length_com * length_com))
            - gravity * sin(x[1]) / length_com
            - damping * x[2] / (mass * length_com * length_com))
   ]
end

function pendulum_discrete(y, x, u)
   h = 0.05 # timestep 
   y - (x + h * pendulum_continuous(0.5 * (x + y), u))
end

dynamics = [pendulum_discrete for t = 1:horizon-1] 

# states
state_initial = [0.0; 0.0] 
state_goal = [π; 0.0] 

# objective 
objective = [
   [(x, u) -> 0.1 * dot(x[1:2], x[1:2]) + 0.1 * dot(u, u) for t = 1:horizon-1]..., 
   (x, u) -> 0.1 * dot(x[1:2], x[1:2]),
];

# constraints 
equality = [
      (x, u) -> x - state_initial, 
      [empty_constraint for t = 2:horizon-1]..., 
      (x, u) -> x - state_goal,
];

# solver 
solver = Solver(objective, dynamics, num_states, num_actions; 
   equality=equality);

# initialize
state_guess = linear_interpolation(state_initial, state_goal, horizon)
action_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
initialize_states!(solver, state_guess) 
initialize_actions!(solver, action_guess)

# solve 
solve!(solver)

# solution
state_solution, action_solution = get_trajectory(solver);
```

## Solution gradients

The solutions returned by CALIPSO are differentiable with respect to problem data provided to the solver. 

```math
    \frac{\partial w}{\partial \theta} = -\Big(\frac{\partial R}{\partial w}\Big)^{-1} \frac{\partial R}{\partial \theta}
```

Sensitivities are efficienctly computing via the [implicit-function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem). This functionality can be utilized by setting the solver option: **differentiate = true**.

### Differentiable trajectory-optimization example 

In the following example we demonstrate how problem data can be provided to the solver in order to get gradients of the solution with respect to these values. For more advanced usage, see our [auto-tuning examples](https://github.com/thowell/CALIPSO.jl/tree/main/examples/autotuning).

```julia 
using CALIPSO
using LinearAlgebra

# horizon 
horizon = 5

# dimensions 
num_states = [2 for t = 1:horizon]
num_actions = [1 for t = 1:horizon-1] 

# dynamics
function double_integrator(y, x, u, w)
	A = reshape(w[1:4], 2, 2) 
	B = w[4 .+ (1:2)] 

	return y - (A * x + B * u[1])
end

# model
dynamics = [double_integrator for t = 1:horizon-1]

# parameters
state_initial = [0.0; 0.0] 
state_goal = [1.0; 0.0] 

A = [1.0 1.0; 0.0 1.0]
B = [0.0; 1.0]
Qt = [1.0 0.0; 0.0 1.0] 
Rt = [0.1]
QT = [10.0 0.0; 0.0 10.0] 
θ1 = [vec(A); B; diag(Qt); Rt; state_initial]
θt = [vec(A); B; diag(Qt); Rt]  
θT = [diag(QT); state_goal] 
parameters = [θ1, [θt for t = 2:horizon-1]..., θT]

# objective 
function obj1(x, u, w) 
	Q1 = Diagonal(w[6 .+ (1:2)])
	R1 = w[8 + 1]
	return 0.5 * transpose(x) * Q1 * x + 0.5 * R1 * transpose(u) * u
end

function objt(x, u, w) 
	Qt = Diagonal(w[6 .+ (1:2)])
	Rt = w[8 + 1]
	return 0.5 * transpose(x) * Qt * x + 0.5 * Rt * transpose(u) * u
end

function objT(x, u, w) 
	QT = Diagonal(w[0 .+ (1:2)])
	return 0.5 * transpose(x) * QT * x
end

objective = [
				obj1,
				[objt for t = 2:horizon-1]...,
				objT,
]

# constraints 
equality = [
		(x, u, w) -> 1 * (x - w[9 .+ (1:2)]),
		[empty_constraint for t = 2:horizon-1]...,
		(x, u, w) -> 1 * (x - w[2 .+ (1:2)]),
]

# options 
options = Options(
		residual_tolerance=1.0e-12, 
		equality_tolerance=1.0e-8,
		complementarity_tolerance=1.0e-8,
		differentiate=true) # <--- setting to get solution gradients

# solver 
solver = Solver(objective, dynamics, num_states, num_actions;
	parameters=parameters,
	equality=equality,
	options=options);

# initialize
state_guess = linear_interpolation(state_initial, state_goal, horizon)
action_guess = [1.0 * randn(num_actions[t]) for t = 1:horizon-1]
initialize_states!(solver, state_guess) 
initialize_actions!(solver, action_guess)

# solve 
solve!(solver)

# solution
state_solution, action_solution = get_trajectory(solver);
```