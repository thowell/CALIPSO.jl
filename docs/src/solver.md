# Solver

CALIPSO is a solver for non-convex optimization problems. 

## problem formulation
Standard form problems:

```math
\begin{align*}
\underset{x}{\text{minimize}} & \quad c(x; \theta) \\
\text{subject to} & \quad  g(x; \theta) = 0, \\
                  & \quad  h(x; \theta) \in \mathcal{K}, \\
\end{align*}
```

with decision variables ``x \in \mathbf{R}^n`` and problem data ``\theta \in \mathbf{R}^d`` are optimized for an objective ``c : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}`` subject to equality ``g : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}^m`` and cone ``h : \mathbf{R}^n \times \mathbf{R}^d \rightarrow \mathbf{R}^p`` constraints are optimized.

The cone,

```math 
\mathcal{K} = \mathbf{R}_{++}^q \times Q_{l_1}^{(1)} \times \dots \times Q_{l_j}^{(j)},
```

is the Cartesian product of the ``q``-dimensional nonnegative orthant and ``j`` second-order cones, each of dimension ``l_i``.

## trajectory optimization 

```math
\begin{align*}
		\underset{X_{1:T}, U_{1:T-1}}{\mbox{minimize }} & C_T(X_T) + \sum \limits_{t = 1}^{T-1} C_t(X_t, U_t)\\
		\mbox{subject to } & F_t(X_t, U_t) = X_{t+1},\phantom{\mathcal{K}} \quad t = 1,\dots,T-1,\\
		& E_t(X_t, U_t) = 0,\phantom{\,_{t+1}\mathcal{K}_t}\quad t = 1, \dots, T,\\
		& H_t(X_t, U_t) \in \mathcal{K}_t,\phantom{\,x_{t+1}}\quad t = 1, \dots, T,
\end{align*}
```

## solution gradients

```math
    \frac{\partial w}{\partial \theta} = -\Big(\frac{\partial R}{\partial w}\Big)^{-1} \frac{\partial R}{\partial \theta}
```