struct Indices 
    variables::Vector{Int} 
    equality_slack::Vector{Int} 
    cone_slack::Vector{Int}
    equality_dual::Vector{Int} 
    cone_dual::Vector{Int} 
    cone_slack_dual::Vector{Int} 
    primals::Vector{Int} 
    duals::Vector{Int}
    symmetric::Vector{Int}
    symmetric_equality::Vector{Int}
    symmetric_cone::Vector{Int}
    violation_equality::Vector{Int} 
    violation_cone::Vector{Int}
    cone_nonnegative::Vector{Int} 
    cone_second_order::Vector{Vector{Int}}
end 

function Indices(num_variables, num_parameters, num_equality, num_cone;
    nonnegative=collect(1:num_cone),
    second_order=[collect(1:0)])

    # variables, equality_slack, cone_slack, equality_dual, cone_dual, cone_slack_dual
    variables = collect(1:num_variables)
    equality_slack = collect(num_variables .+ (1:num_equality))
    cone_slack = collect(num_variables + num_equality .+ (1:num_cone))

    equality_dual = collect(num_variables + num_equality + num_cone .+ (1:num_equality))
    cone_dual = collect(num_variables + num_equality + num_cone + num_equality .+ (1:num_cone))
    cone_slack_dual = collect(num_variables + num_equality + num_cone + num_equality + num_cone .+ (1:num_cone))
    
    symmetric = collect(1:(num_variables + num_equality + num_cone))
    symmetric_equality = collect(num_variables .+ (1:num_equality))
    symmetric_cone = collect(num_variables + num_equality .+ (1:num_cone))

    primals = collect(1:(num_variables + num_equality + num_cone))
    duals = collect(num_variables + num_equality + num_cone .+ (1:(num_equality + num_cone + num_cone)))

    violation_equality = collect(1:num_equality) 
    violation_cone = collect(num_equality .+ (1:num_cone))

    return Indices(
        variables, 
        equality_slack, 
        cone_slack,
        equality_dual,
        cone_dual,
        cone_slack_dual,
        primals, 
        duals,
        symmetric,
        symmetric_equality,
        symmetric_cone,
        violation_equality, 
        violation_cone,
        nonnegative, 
        second_order,
    )
end