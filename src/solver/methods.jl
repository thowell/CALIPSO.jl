struct ProblemMethods{O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP}
    objective::O                                       # f
    objective_gradient_variables::OX                   # fx
    objective_gradient_parameters::OP                  # fθ
    objective_jacobian_variables_variables::OXX        # fxx 
    objective_jacobian_variables_parameters::OXP       # fxθ
    equality_constraint::E                             # g
    equality_jacobian_variables::EX                    # gx 
    equality_jacobian_parameters::EP                   # gθ
    equality_dual::ED                                  # g'y
    equality_dual_jacobian_variables::EDX              # (g'y)x
    equality_dual_jacobian_variables_variables::EDXX   # (g'y)xx
    equality_dual_jacobian_variables_parameters::EDXP  # (g'y)xθ
    cone_constraint::C                                 # h
    cone_jacobian_variables::CX                        # hx 
    cone_jacobian_parameters::CP                       # hθ
    cone_dual::CD                                      # h'z
    cone_dual_jacobian_variables::CDX                  # (h'y)x
    cone_dual_jacobian_variables_variables::CDXX       # (h'y)xx
    cone_dual_jacobian_variables_parameters::CDXP      # (h'y)xθ
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