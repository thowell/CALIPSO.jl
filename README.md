[![CI](https://github.com/thowell/CALIPSO.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/CALIPSO.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/CALIPSO.jl/branch/main/graph/badge.svg?token=RNX4943S70)](https://codecov.io/gh/thowell/CALIPSO.jl)

# CALIPSO.jl
Conic Augmented Lagrangian Interior-Point SOlver: A solver for contact-implicit trajectory optimization

The CALIPSO algorithm is an infeasible-start, primal-dual augmented-Lagrangian interior-point solver for non-convex optimization problems. 
An augmented Lagrangian is employed for equality constraints and cones are handled by interior-point methods.

## Standard form
Problems of the following form:
```
minimize     f(x; p)
   x
subject to   g(x; p)  = 0,
             h(x; p) in K = R+ x Q^1 x ... x Q^k
```
can be optimized for 

- x: decision variables 
- p: problem parameters 
- K: Cartesian product of convex cones; nonnegative orthant R+ and second-order cones Q are currently implemented

## Trajectory optimization
Additionally, trajectory optimization problems of the form:
```
minimize        cost_T(state_T; parameter_T) + sum(cost_t(state_t, action_t; parameter_t))
states, actions
subject to      dynamics_t(state_t+1, state_t, action_t; parameter_t)  = 0,        t = 1,...,T-1  
                equality_t(state_t, action_t; parameter_t)             = 0,        t = 1,...,T
                inequality_t(state_t, action_t; parameter_t)          >= 0,        t = 1,...,T
                second_order_t(state_t, action_t; parameter_t)        in Q,        t = 1,...,T
``` 
are automatically formulated, and fast gradients generated, for CALIPSO.

## Solution gradients
The solver is differentiable, and gradients of the solution (including internal solver variables) with respect to the problem parameters are efficiently computed.

## Quick start 
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

