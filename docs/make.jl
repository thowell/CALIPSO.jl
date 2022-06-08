push!(LOAD_PATH, "../src/")

# copy animations to docs/src/assets/animations
path_assets_animations = joinpath(@__DIR__, "src/assets/animations")
!isdir(path_assets_animations) && mkdir(path_assets_animations)
path_animations = joinpath(@__DIR__, "../examples/animations")
files = readdir(path_animations)
filter!(x -> endswith(x, ".gif"), files)
for file in files 
    cp(joinpath(path_animations, file), joinpath(path_assets_animations, file), force=true)
end

using Documenter#, CALIPSO

makedocs(
    modules = [CALIPSO],
    format = Documenter.HTML(prettyurls=false),
    sitename = "CALIPSO",
    pages = [
        ##############################################
        ## MAKE SURE TO SYNC WITH docs/src/index.md ##
        ##############################################
        "index.md",
        "solver.md",
        "options.md",
        "examples.md",
        "api.md",
        "contributing.md",
        "citing.md",
    ]
)

# deploydocs(
#     repo = "github.com/thowell/CALIPSO.jl.git",
# )
