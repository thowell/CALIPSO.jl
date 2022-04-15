[![CI](https://github.com/thowell/CALIPSO.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/thowell/CALIPSO.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/thowell/CALIPSO.jl/branch/main/graph/badge.svg?token=RNX4943S70)](https://codecov.io/gh/thowell/CALIPSO.jl)

# CALIPSO.jl
Conic Augmented Lagrangian Interior-Point SOlver: A solver for contact-implicit trajectory optimization

The CALIPSO algorithm is an infeasible-start, primal-dual augmented-Lagrangian interior-point solver for non-convex optimization problems. 

Problems of the following form:
```
minimize     f(x; p)
   x
subject to   g(x; p)  = 0,
             h(x; p) in K
```
can be optimized for 

- x: decision variables 
- p: problem parameters 
- K: Cartesian product of non-negative orthant and second-order cones 

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

