struct Dimensions 
    variables::Int 
    equality::Int 
    inequality::Int 
    slack_primal::Int 
    slack_dual::Int 
    total::Int
end

function Dimensions(num_variables, num_equality, num_inequality) 
    num_total = num_variables + num_equality + 3 * num_inequality

    Dimensions(
        num_variables, 
        num_equality, 
        num_inequality, 
        num_inequality, 
        num_inequality, 
        num_total
    )
end
