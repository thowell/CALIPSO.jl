# API Documentation

Docstrings for CALIPSO.jl interface members can be [accessed through Julia's built-in documentation system](https://docs.julialang.org/en/v1/manual/documentation/index.html#Accessing-Documentation-1) or in the list below.


```@meta
CurrentModule = CALIPSO
```

## Contents

```@contents
Pages = ["api.md"]
```

## Index

```@index
Pages = ["api.md"]
```

## Solver

```@docs
    Solver
    solve!
    initialize!
    Options
    empty_constraint
    callback_inner
    callback_outer
```

## Trajectory Optimization

```@docs
    Cost
    Constraint
    Constraints
    Dynamics
    initialize_states!
    initialize_actions!
    get_trajectory
    linear_interpolation
```
