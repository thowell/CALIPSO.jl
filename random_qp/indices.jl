struct Indices 
    variables::Vector{Int} 
    slack_primal::Vector{Int} 
    equality::Vector{Int} 
    inequality::Vector{Int} 
    slack_dual::Vector{Int} 
    symmetric::Vector{Int}
    symmetric_equality::Vector{Int}
    symmetric_inequality::Vector{Int}
end 

function Indices(num_variables, num_equality, num_inequality)
    # variables, slack_primal, equality, inequality, slack_dual
    variables = collect(1:num_variables)
    slack_primal = collect(num_variables .+ (1:num_inequality))    

    equality = collect(num_variables + num_inequality .+ (1:num_equality))
    inequality = collect(num_variables + num_inequality + num_equality .+ (1:num_inequality))
    slack_dual = collect(num_variables + num_inequality + num_equality + num_inequality .+ (1:num_inequality))
    
    symmetric = collect(1:(num_variables + num_inequality + num_equality))
    symmetric_equality = collect(num_variables .+ (1:num_equality))
    symmetric_inequality = collect(num_variables + num_equality .+ (1:num_inequality))

    return Indices(
        variables, 
        slack_primal,
        equality,
        inequality,
        slack_dual,
        symmetric,
        symmetric_equality,
        symmetric_inequality
    )
end