@benchmark evaluate_problem!($p_data, $idx, $w)

@benchmark matrix!($s_data, $p_data, $idx, $w, $κ, $ρ, $λ)

@benchmark matrix_symmetric!($s_data, $p_data, $idx, $w, $κ, $ρ, $λ)

@benchmark residual!($s_data, $p_data, $idx, $w, $κ, $ρ, $λ)

@benchmark residual_symmetric!($s_data, $p_data, $idx, $w, $κ, $ρ, $λ)

@benchmark step!($s_data)

@benchmark step_symmetric!($s_data, $idx, $w, $κ)

@benchmark initialize!($w, $x, $idx)

