struct Dimensions 
    variables::Int 
    equality_slack::Int
    cone_slack::Int
    equality_dual::Int 
    cone_dual::Int 
    cone_slack_dual::Int 
    primal::Int 
    dual::Int 
    symmetric::Int
    total::Int
    cone_nonnegative::Int 
    cone_second_order::Vector{Int}
end

function Dimensions(num_variables, num_equality, num_cone;
    nonnegative=num_cone,
    second_order=[0,]) 
    num_total = num_variables + num_equality + num_cone # primal 
    num_total += num_equality + 2 * num_cone

    Dimensions(
        num_variables, 
        num_equality,
        num_cone,
        num_equality, 
        num_cone, 
        num_cone,
        num_variables + num_equality + num_cone, 
        num_equality + 2 * num_cone, 
        num_variables + num_equality + num_cone,
        num_total,
        nonnegative,
        second_order,
    )
end
