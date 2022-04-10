struct ProblemMethods
    objective::Any 
    objective_gradient::Any 
    objective_hessian::Any 
    equality_constraint::Any 
    equality_jacobian::Any 
    equality_hessian::Any
    cone_constraint::Any 
    cone_jacobian::Any 
    cone_hessian::Any
end

function ProblemMethods(num_variables::Int, objective::Function, equality::Function, cone::Function)
    # generate methods
    obj, obj_grad!, obj_hess! = generate_gradients(objective, num_variables, :scalar)
    eq_con!, eq_jac!, eq_hess! = generate_gradients(equality, num_variables, :vector)
    cone_con!, cone_jac!, cone_hess! = generate_gradients(cone, num_variables, :vector)

    ProblemMethods(
        obj, 
        obj_grad!, 
        obj_hess!,
        eq_con!, 
        eq_jac!, 
        eq_hess!, 
        cone_con!, 
        cone_jac!, 
        cone_hess!,
    )
end