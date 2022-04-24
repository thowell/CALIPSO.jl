abstract type LinearSolver end

"""
    QDLDL inplace functionality
"""
mutable struct LDLSolver{Tf<:AbstractFloat,Ti<:Integer} <: LinearSolver
    # QDLDL Factorization
    F::QDLDLFactorisation{Tf,Ti}
    A_sparse::SparseMatrixCSC{Tf,Ti}
    inertia::Inertia
end

function LDLSolver(A::SparseMatrixCSC{Tv,Ti}, F::QDLDLFactorisation{Tv,Ti}) where {Tv<:AbstractFloat,Ti<:Integer}
    inertia = Inertia(0, 0, 0)
    return LDLSolver{Tv,Ti}(
        F, 
        copy(A), 
        inertia
    )
end

function factorize!(s::LDLSolver{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}; 
    update=false) where {Tv<:AbstractFloat, Ti<:Integer}
     
    if update
        triu!(A)
        update_values!(s.F, 1:length(A.nzval), A.nzval)
        refactor!(s.F)
    else 
        s.F = qdldl(A)
    end

    return nothing
end

function compute_inertia!(ls::LDLSolver)
    ls.inertia.positive = ls.F.workspace.positive_inertia.x 
    n = 0 
    z = 0
    for d in ls.F.workspace.D 
        d <= 0.0 && (n += 1) 
        d == 0.0 && (z += 1)
    end
    ls.inertia.negative = n
    ls.inertia.zero = z
    return nothing
end

"""
    LDL solver
"""
function ldl_solver(A::SparseMatrixCSC{T,Int}) where T
    LDLSolver(A, qdldl(A))
end

ldl_solver(A::Array{T, 2}) where T = ldl_solver(sparse(A))

function linear_solve!(solver::LDLSolver{Tv,Ti}, x::Vector{Tv}, A::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv};
    fact=true,
    update=true) where {Tv<:AbstractFloat,Ti<:Integer}

    fact && factorize!(solver, A;
        update=update) # factorize
    x .= b
    solve!(solver.F, x) # solve
end

# function linear_solve!(solver::LDLSolver{Tv,Ti}, x::Vector{Tv}, A::AbstractMatrix{Tv}, b::Vector{Tv};
#     fact=true,
#     update=true) where {Tv<:AbstractFloat,Ti<:Integer}

#     # fill sparse_matrix
#     n, m = size(A) 
#     for i = 1:n 
#         for j = 1:m 
#             solver.A_sparse[i, j] = A[i, j]
#         end
#     end
    
#     linear_solve!(solver, x, solver.A_sparse, b, 
#         fact=fact,
#         update=update)
# end

function linear_solve!(s::LDLSolver{T}, x::Matrix{T}, A::Matrix{T},
    b::Matrix{T}; 
    fact=true,
    update=true) where T

    fill!(x, 0.0)
    n, m = size(x) 
    r_idx = 1:n
    fact && factorize!(s, A;
        update=update)

    x .= b 
    for j = 1:m
        xv = @views x[r_idx, j]
        solve!(solver.F, xv)
    end
end
