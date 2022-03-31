struct ProblemMethods
    objective::Any 
    objective_gradient::Any 
    objective_hessian::Any 
    equality::Any 
    equality_jacobian::Any 
    equality_hessian::Any
    inequality::Any 
    inequality_jacobian::Any 
    inequality_hessian::Any
end

function ProblemMethods(num_variables::Int, objective::Function, equality::Function, inequality::Function )
    # generate methods
    obj, obj_grad!, obj_hess! = generate_gradients(objective, num_variables, :scalar)
    eq!, eq_jac!, eq_hess! = generate_gradients(equality, num_variables, :vector)
    ineq!, ineq_jac!, ineq_hess! = generate_gradients(inequality, num_variables, :vector)

    ProblemMethods(
        obj, 
        obj_grad!, 
        obj_hess!,
        eq!, 
        eq_jac!, 
        eq_hess!, 
        ineq!, 
        ineq_jac!, 
        ineq_hess!,
    )
end