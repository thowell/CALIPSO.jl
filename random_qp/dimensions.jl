struct Dimensions 
    variables::Int 
    equality_slack::Int
    inequality_slack::Int
    equality_dual::Int 
    inequality_dual::Int 
    inequality_slack_dual::Int 
    primal::Int 
    dual::Int 
    symmetric::Int
    total::Int
end

function Dimensions(num_variables, num_equality, num_inequality) 
    num_total = num_variables + num_equality + num_inequality # primal 
    num_total += num_equality + 2 * num_inequality

    Dimensions(
        num_variables, 
        num_equality,
        num_inequality,
        num_equality, 
        num_inequality, 
        num_inequality,
        num_variables + num_equality + num_inequality, 
        num_equality + 2 * num_inequality, 
        num_variables + num_equality + num_inequality,
        num_total,
    )
end
