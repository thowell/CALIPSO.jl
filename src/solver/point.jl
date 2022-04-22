struct Point{T} 
    all::Vector{T}
    variables::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    equality_slack::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    cone_slack::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false} 
    equality_dual::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    cone_dual::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false} 
    cone_slack_dual::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    primals::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
end

function Point(dims::Dimensions, idx::Indices)
    w = zeros(dims.total)
    x = @views w[idx.variables]
    r = @views w[idx.equality_slack]
    s = @views w[idx.cone_slack]
    y = @views w[idx.equality_dual]
    z = @views w[idx.cone_dual]
    t = @views w[idx.cone_slack_dual]
    p = @views w[idx.primals]
    return Point(w, x, r, s, y, z, t, p)
end


