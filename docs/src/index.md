# Get Started 

__[CALIPSO](https://github.com/thowell/CALIPSO.jl) is a differentiable solver for trajectory optimization with conic and complementarity constraints__. The solver is written in pure Julia in order to be both performant and easy to use.

## Features
* __Differentiable__: Solutions are efficiently differentiable with respect to problem data provided to the solver
* __Trajectory Optimization__: Problems formulated as deterministic Markov Decision Processes are automatically transcribed for the solver
* __Complementarity Constraints__: Complementarity constraints can be provided to the solver without reformulation
* __Second-Order-Cone Constraints__: Cone constraints are natively supported in the non-convex problem setting
* __Codegen for Derivatives__: User-provided functions (e.g., objective, constraints) are symbolically differentiated and fast code is autogenerated using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
* __Open Source__: Code is available on [GitHub](https://github.com/thowell/CALIPSO.jl) and distributed under the MIT Licence

## Installation
CALIPSO can be installed using the Julia package manager for Julia `v1.7` and higher. Inside the Julia REPL, type `]` to enter the Pkg REPL mode then run:

`pkg> add CALIPSO`

If you want to install the latest version from Github run:

`pkg> add CALIPSO#main`

## Citation 
If this project is useful for your work please consider:
* [Citing](citing.md) the relevant paper
* Leaving a star on the [GitHub repository](https://github.com/thowell/CALIPSO.jl)

## Licence
CALIPSO is licensed under the MIT License. For more details click [here](https://github.com/thowell/CALIPSO.jl/blob/main/LICENSE.md).
