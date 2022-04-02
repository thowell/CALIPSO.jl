# CALIPSO.jl
Conic Augmented Lagrangian Interior-Point SOlver: A solver for contact-implicit trajectory optimization

CALIPSO is an infeasible-start, primal-dual augmented-Lagrangian interior-point solver for non-convex optimization problems. 

Problems of the following form:
```
minimize     f(x)
   x
subject to   g(x)  = 0,
             h(x) >= 0
```
can be optimized. 

Additionally, trajectory optimization problems of the form:
```
minimize        cost_T(state_T; parameter_T) + sum(cost_t(state_t, action_t; parameter_t))
states, actions
subject to      dynamics_t(state_t, action_t, state_t+1; parameter_t) = 0,        t = 1,...,T-1  
                equality_t(state_t, action_t; parameter_t)            = 0,        t = 1,...,T
                inequality_t(state_t, action_t; parameter_t)          = 0,        t = 1,...,T
``` 
are automatically formulated, and fast gradients generated, for CALIPSO.

