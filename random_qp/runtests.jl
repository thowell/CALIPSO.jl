x = randn(num_variables)
y = randn(num_equality)
z = randn(num_inequality)
s = rand(num_inequality)
t = rand(num_inequality) 

w = [x; y; z; s; t]
κ = [1.0]
ρ = [1.0e6]
λ = zeros(num_equality)

evaluate_problem!(p_data, idx, w)

matrix!(s_data, p_data, idx, w, κ, ρ, λ)

matrix_symmetric!(s_data, p_data, idx, w, κ, ρ, λ)

residual!(s_data, p_data, idx, w, κ, ρ, λ)

residual_symmetric!(s_data, p_data, idx, w, κ, ρ, λ)

# KKT matrix 
@test rank(s_data.matrix) == num_total
@test norm(s_data.matrix[idx.primal, idx.primal] - (p_data.objective_hessian + p_data.equality_hessian + p_data.inequality_hessian)) < 1.0e-6
@test norm(s_data.matrix[idx.equality, idx.primal] - p_data.equality_jacobian) < 1.0e-6
@test norm(s_data.matrix[idx.primal, idx.equality] - p_data.equality_jacobian') < 1.0e-6
@test norm(s_data.matrix[idx.equality, idx.equality] + 1.0 / ρ[1] * I(num_equality)) < 1.0e-6
@test norm(s_data.matrix[idx.inequality, idx.primal] - p_data.inequality_jacobian) < 1.0e-6
@test norm(s_data.matrix[idx.primal, idx.inequality] - p_data.inequality_jacobian') < 1.0e-6
@test norm(s_data.matrix[idx.slack_primal, idx.inequality] + I(num_inequality)) < 1.0e-6
@test norm(s_data.matrix[idx.inequality, idx.slack_primal] + I(num_inequality)) < 1.0e-6
@test norm(s_data.matrix[idx.slack_primal, idx.slack_dual] + I(num_inequality)) < 1.0e-6
@test norm(s_data.matrix[idx.slack_dual, idx.slack_primal] - Diagonal(w[idx.slack_dual])) < 1.0e-6
@test norm(s_data.matrix[idx.slack_dual, idx.slack_dual] - Diagonal(w[idx.slack_primal])) < 1.0e-6

# KKT matrix (symmetric)
@test rank(s_data.matrix_symmetric) == num_variables + num_equality + num_inequality
@test norm(s_data.matrix_symmetric[idx.primal, idx.primal] - (p_data.objective_hessian + p_data.equality_hessian + p_data.inequality_hessian)) < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.equality, idx.primal] - p_data.equality_jacobian) < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.primal, idx.equality] - p_data.equality_jacobian') < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.equality, idx.equality] + 1.0 / ρ[1] * I(num_equality)) < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.inequality, idx.primal] - p_data.inequality_jacobian) < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.primal, idx.inequality] - p_data.inequality_jacobian') < 1.0e-6
@test norm(s_data.matrix_symmetric[idx.inequality, idx.inequality] + Diagonal(w[idx.slack_primal] ./ w[idx.slack_dual])) < 1.0e-6

# residual 
@test norm(s_data.residual[idx.primal] - (p_data.objective_gradient + p_data.equality_jacobian' * w[idx.equality] + p_data.inequality_jacobian' * w[idx.inequality])) < 1.0e-6
@test norm(s_data.residual[idx.equality] - (p_data.equality - 1.0 / ρ[1] * (λ - w[idx.equality]))) < 1.0e-6
@test norm(s_data.residual[idx.inequality] - (p_data.inequality - w[idx.slack_primal])) < 1.0e-6
@test norm(s_data.residual[idx.slack_primal] - (-w[idx.inequality] - w[idx.slack_dual])) < 1.0e-6
@test norm(s_data.residual[idx.slack_dual] - (w[idx.slack_primal] .* w[idx.slack_dual] .- κ[1])) < 1.0e-6

# residual symmetric
@test norm(s_data.residual_symmetric[idx.primal] - (p_data.objective_gradient + p_data.equality_jacobian' * w[idx.equality] + p_data.inequality_jacobian' * w[idx.inequality])) < 1.0e-6
@test norm(s_data.residual_symmetric[idx.equality] - (p_data.equality - 1.0 / ρ[1] * (λ - w[idx.equality]))) < 1.0e-6
@test norm(s_data.residual_symmetric[idx.inequality] - (p_data.inequality - w[idx.slack_primal] - w[idx.slack_primal] .* w[idx.inequality] ./ w[idx.slack_dual] - κ[1] * ones(num_inequality) ./ w[idx.slack_dual])) < 1.0e-6

# step
step!(s_data)
Δ = deepcopy(s_data.step)
step_symmetric!(s_data, idx, w, κ)
Δ_symmetric = deepcopy(s_data.step)
@test norm(Δ - Δ_symmetric) < 1.0e-6