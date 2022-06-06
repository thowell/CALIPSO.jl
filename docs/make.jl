push!(LOAD_PATH, "../src/")

using Documenter, CALIPSO

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
        "examples.md",
        "api.md",
        "citing.md",
    ]
)

# deploydocs(
#     repo = "github.com/thowell/CALIPSO.jl.git",
# )
