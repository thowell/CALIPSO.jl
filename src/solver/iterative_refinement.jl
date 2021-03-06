function iterative_refinement!(step::Point{T}, solver::Solver{T,O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP,B,BX,P,PX,PXI,K}) where {T,O,OX,OP,OXX,OXP,E,EX,EP,ED,EDX,EDXX,EDXP,C,CX,CP,CD,CDX,CDXX,CDXP,B,BX,P,PX,PXI,K}
    # reset 
    fill!(solver.data.step_correction.all, 0.0) 
    fill!(solver.data.residual_error.all, 0.0)
    iteration = 0 

    # residual error
    solver.data.residual_error.all .= solver.data.residual.all
    mul!(solver.data.residual_error.all, solver.data.jacobian_variables, step.all, -1.0, 1.0)
    
    residual_norm = norm(solver.data.residual_error.all, Inf)
    residual_norm_initial = residual_norm 

    while iteration <= solver.options.max_iterative_refinement  
        if residual_norm <= solver.options.iterative_refinement_tolerance && iteration >= solver.options.min_iterative_refinement
            return true
        end
        
        # correction
        if solver.options.linear_solver == :QDLDL
            search_direction_symmetric!(solver.data.step_correction, solver.data.residual_error, solver.data.jacobian_variables, 
                solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.jacobian_variables_symmetric, 
                solver.indices, 
                solver.data.step_correction_second_order, solver.data.residual_error_second_order,
                solver.linear_solver)
        else 
            search_direction_nonsymmetric!(
                solver.data.step_correction, 
                solver.data.jacobian_variables, 
                solver.data.residual_error,
                solver.lu_factorization)
        end
        
        # update
        step.all .+= solver.data.step_correction.all 
        
        # residual error
        solver.data.residual_error.all .= solver.data.residual.all
        mul!(solver.data.residual_error.all, solver.data.jacobian_variables, step.all, -1.0, 1.0)
        
        residual_norm = norm(solver.data.residual_error.all, Inf)

        iteration += 1
    end

    if residual_norm <= residual_norm_initial 
        return true
    else
        # failure
        @warn "iterative refinement failure: $(residual_norm)"
        return false
    end
end