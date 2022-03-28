struct Indices 
    variables::Vector{Int} 
    equality_slack::Vector{Int} 
    inequality_slack::Vector{Int}
    equality_dual::Vector{Int} 
    inequality_dual::Vector{Int} 
    inequality_slack_dual::Vector{Int} 
    symmetric::Vector{Int}
    symmetric_equality::Vector{Int}
    symmetric_inequality::Vector{Int}
end 

function Indices(num_variables, num_equality, num_inequality)
    # variables, equality_slack, inequality_slack, equality_dual, inequality_dual, inequality_slack_dual
    variables = collect(1:num_variables)
    equality_slack = collect(num_variables .+ (1:num_equality))
    inequality_slack = collect(num_variables + num_equality .+ (1:num_inequality))

    equality_dual = collect(num_variables + num_equality + num_inequality .+ (1:num_equality))
    inequality_dual = collect(num_variables + num_equality + num_inequality + num_equality .+ (1:num_inequality))
    inequality_slack_dual = collect(num_variables + num_equality + num_inequality + num_equality + num_inequality .+ (1:num_inequality))
    
    symmetric = collect(1:(num_variables + num_equality + num_inequality))
    symmetric_equality = collect(num_variables .+ (1:num_equality))
    symmetric_inequality = collect(num_variables + num_equality .+ (1:num_inequality))

    return Indices(
        variables, 
        equality_slack, 
        inequality_slack,
        equality_dual,
        inequality_dual,
        inequality_slack_dual,
        symmetric,
        symmetric_equality,
        symmetric_inequality
    )
end