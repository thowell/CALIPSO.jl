struct ProblemMethods
    objective::Any                                    # f
    objective_gradient_variables::Any                 # fx
    objective_gradient_parameters::Any                # fθ
    objective_jacobian_variables_variables::Any       # fxx 
    objective_jacobian_variables_parameters::Any      # fxθ
    equality_constraint::Any                          # g
    equality_jacobian_variables::Any                  # gx 
    equality_jacobian_parameters::Any                 # gθ
    equality_dual::Any                                # g'y
    equality_dual_jacobian_variables::Any             # (g'y)x
    equality_dual_jacobian_variables_variables::Any   # (g'y)xx
    equality_dual_jacobian_variables_parameters::Any  # (g'y)xθ
    cone_constraint::Any                              # h
    cone_jacobian_variables::Any                      # hx 
    cone_jacobian_parameters::Any                     # hθ
    cone_dual::Any                                    # h'z
    cone_dual_jacobian_variables::Any                 # (h'y)x
    cone_dual_jacobian_variables_variables::Any       # (h'y)xx
    cone_dual_jacobian_variables_parameters::Any      # (h'y)xθ
end

function ProblemMethods(num_variables::Int, num_parameters::Int, objective::Function, equality::Function, cone::Function)
    # generate methods
    f, fx!, fθ!, fxx!, fxθ! = generate_gradients(objective, num_variables, num_parameters, :scalar)
    g, gx, gθ, gᵀy, gᵀyx, gᵀyxx, gᵀyxθ = generate_gradients(equality, num_variables, num_parameters, :vector)
    h, hx, hθ, hᵀy, hᵀyx, hᵀyxx, hᵀyxθ = generate_gradients(cone, num_variables, num_parameters, :vector)

    ProblemMethods(
        f, fx!, fθ!, fxx!, fxθ!,
        g, gx, gθ, gᵀy, gᵀyx, gᵀyxx, gᵀyxθ,
        h, hx, hθ, hᵀy, hᵀyx, hᵀyxx, hᵀyxθ,
    )
end