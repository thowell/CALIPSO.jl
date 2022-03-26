struct Dimensions 
    variables::Int 
    slack::Int 
    equality::Int 
    inequality::Int 
    slack_dual::Int 
    primal::Int 
    dual::Int 
    symmetric::Int
    total::Int
end

function Dimensions(num_variables, num_equality, num_inequality) 
    num_total = num_variables + num_equality + 3 * num_inequality

    Dimensions(
        num_variables, 
        num_inequality,
        num_equality, 
        num_inequality, 
        num_inequality,
        num_variables + num_inequality, 
        num_equality + 2 * num_inequality, 
        num_variables + num_equality + num_inequality,
        num_total,
    )
end
