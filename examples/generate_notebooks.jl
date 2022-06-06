using Literate

function preprocess(str)
    str = replace(str, "# PREAMBLE" => "")
    str = replace(str, "# PKG_SETUP" => "")
    return str
end

exampledir = @__DIR__

# non-convex
Literate.notebook(joinpath(exampledir, "nonconvex/wachter.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "nonconvex/maratos.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "nonconvex/complementarity.jl"), exampledir, execute=false, preprocess=preprocess)

# contact-implicit trajectory optimization
Literate.notebook(joinpath(exampledir, "contact_implicit/ball_in_cup.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "contact_implicit/bunnyhop.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "contact_implicit/quadruped_gait.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "contact_implicit/drifting.jl"), exampledir, execute=false, preprocess=preprocess)

# state-triggered constraints 
Literate.notebook(joinpath(exampledir, "state_triggered/rocket_landing.jl"), exampledir, execute=false, preprocess=preprocess)

# auto-tuning
Literate.notebook(joinpath(exampledir, "autotuning/cartpole.jl"), exampledir, execute=false, preprocess=preprocess)
Literate.notebook(joinpath(exampledir, "autotuning/acrobot.jl"), exampledir, execute=false, preprocess=preprocess)










