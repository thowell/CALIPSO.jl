struct Indices 
    primal::Vector{Int} 
    equality::Vector{Int} 
    inequality::Vector{Int} 
    slack_primal::Vector{Int} 
    slack_dual::Vector{Int} 
    symmetric::Vector{Int}
end 

function Indices(num_variables, num_equality, num_inequality)
    # primal, equality, inequality, slack_primal, slack_dual
    primal = collect(1:num_variables)
    equality = collect(num_variables .+ (1:num_equality))
    inequality = collect(num_variables + num_equality .+ (1:num_inequality))
    slack_primal = collect(num_variables + num_equality + num_inequality .+ (1:num_inequality))    
    slack_dual = collect(num_variables + num_equality + 2 * num_inequality .+ (1:num_inequality))
    symmetric = collect(1:(num_variables + num_equality + num_inequality))

    return Indices(
        primal, 
        equality,
        inequality,
        slack_primal,
        slack_dual,
        symmetric,
    )
end