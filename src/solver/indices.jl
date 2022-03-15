################################################################################
# Indices
################################################################################

struct IndicesOptimization 
	# Set the residual to 0
	# r(z) = 0
	# z <- z + Δz

	# Dimensions
	nz::Int # dimension of the optimization variable z
	nΔ::Int # dimension of the optimization variable Δz and of the residual r

	# Variables
	ortz::Vector{Vector{Int}} # indices of the variables associated with the positive ORThant constraints in z
	ortΔ::Vector{Vector{Int}} # indices of the variables associated with the positive ORThant constraints in Δz
	socz::Vector{Vector{Vector{Int}}} # indices of the variables associated with the Second Order Cone constraints in z
	socΔ::Vector{Vector{Vector{Int}}} # indices of the variables associated with the Second Order Cone constraints in Δz

	# Residual
	equr::Vector{Int} # indices of the residual associated with the EQUality constraints in r
	ortr::Vector{Int} # indices of the residual associated with the positive ORThant constraints in r
	socr::Vector{Int} # indices of the residual associated with the Second-Order Constraints in r
	socri::Vector{Vector{Int}} # indices of the residual associated with individual Second-Order Cone constraints in r
	bil::Vector{Int} # indices of the residual associated with the bilinear constraints in r
end

function IndicesOptimization()
	v1 = Vector{Int}()
	v2 = Vector{Vector{Int}}()
	v3 = Vector{Vector{Vector{Int}}}()

	s = IndicesOptimization(
		0, 0,
		v2, v2, v3, v3,
		v1, v1, v1, v2, v1)
	return s
end

a = Vector{Vector{Int}}()
b = Vector{Vector{Vector{Int}}}()
isempty(a)
isempty(b)

c = [[collect(1:0), collect(1:0)], [collect(1:0), collect(1:0)]]
isempty(c)
isempty(c[1])
length(c[1][1])
c[1][1]
for bi in b 
	@show bi 
end

