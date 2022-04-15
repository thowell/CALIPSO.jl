function residual_jacobian_parameters!(data::SolverData, problem::ProblemData, idx::Indices, κ, ρ, λ, ϵp, ϵd;
    constraint_hessian=true)
  
    # # reset
    # H = data.jacobian 
    # fill!(H, 0.0)

    # # Hessian of Lagrangian
    # for i in idx.variables 
    #     for j in idx.variables 
    #         H[i, j]  = problem.objective_jacobian_variables_variables[i, j] 
    #         constraint_hessian && (H[i, j] += problem.equality_dual_jacobian_variables_variables[i, j])
    #         constraint_hessian && (H[i, j] += problem.cone_dual_jacobian_variables_variables[i, j])
    #     end
    # end

    # for (i, ii) in enumerate(idx.equality_slack) 
    #     for (j, jj) in enumerate(idx.equality_dual) 
    #         if i == j
    #             H[ii, jj] = -1.0 
    #             H[jj, ii] = -1.0
    #         end 
    #     end
    # end

    # for (i, ii) in enumerate(idx.cone_slack) 
    #     for (j, jj) in enumerate(idx.cone_dual) 
    #         if i == j
    #             H[ii, jj] = -1.0 
    #             H[jj, ii] = -1.0 
    #         end
    #     end
    # end

    # for (i, ii) in enumerate(idx.cone_slack) 
    #     for (j, jj) in enumerate(idx.cone_slack_dual) 
    #         if i == j
    #             H[ii, jj] = -1.0 
    #         end
    #     end
    # end

    # # equality Jacobian 
    # for (i, ii) in enumerate(idx.equality_dual) 
    #     for (j, jj) in enumerate(idx.variables)
    #         H[ii, jj] = problem.equality_jacobian_variables[i, j]
    #         H[jj, ii] = problem.equality_jacobian_variables[i, j]
    #     end
    # end

    # # cone Jacobian 
    # for (i, ii) in enumerate(idx.cone_dual) 
    #     for (j, jj) in enumerate(idx.variables)
    #         H[ii, jj] = problem.cone_jacobian_variables[i, j]
    #         H[jj, ii] = problem.cone_jacobian_variables[i, j]
    #     end
    # end

    # # augmented Lagrangian block 
    # for (i, ii) in enumerate(idx.equality_slack) 
    #     H[ii, ii] = ρ[1]
    # end
    
    # # cone block (non-negative)
    # for i in idx.cone_nonnegative
    #     H[idx.cone_slack_dual[i], idx.cone_slack[i]] = problem.cone_product_jacobian_primal[i, i] 
    #     H[idx.cone_slack_dual[i], idx.cone_slack_dual[i]] = problem.cone_product_jacobian_dual[i, i]  
    # end

    # # cone block (second-order)
    # for idx_soc in idx.cone_second_order
    #     Cs = @views problem.cone_product_jacobian_primal[idx_soc, idx_soc] 
    #     Ct = @views problem.cone_product_jacobian_dual[idx_soc, idx_soc] 
    #     H[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]] = Cs 
    #     H[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]] = Ct 
    # end

    # # regularization 
    # for i in idx.variables 
    #     H[i, i] += ϵp 
    # end

    # for i in idx.equality_slack
    #     H[i, i] += ϵp
    # end 

    # for i in idx.cone_slack
    #     H[i, i] += ϵp
    # end 

    # for i in idx.equality_dual
    #     H[i, i] -= ϵd
    # end

    # for i in idx.cone_dual
    #     H[i, i] -= ϵd 
    # end

    # for i in idx.cone_slack_dual 
    #     H[i, i] -= ϵd 
    # end 

    # return
    nothing
end

