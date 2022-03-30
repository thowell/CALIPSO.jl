# ### second order correction 
# step_copy = deepcopy(solver.data.step)
# for i = 1:solver.options.max_second_order_correction
#     problem!(solver.problem, solver.methods, solver.indices, solver.candidate,
#         gradient=false,
#         constraint=true,
#         jacobian=false,
#         hessian=false)

#     solver.data.residual[solver.indices.equality] .+= solver.problem.equality + 1.0 / solver.penalty[1] * (solver.dual - solver.candidate[solver.indices.equality])
#     solver.data.residual[solver.indices.inequality] .+= (solver.problem.inequality - solver.candidate[solver.indices.slack_primal])
                    
#     search_direction_symmetric!(solver.data.step, solver.data.residual, solver.data.matrix, 
#         solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.matrix_symmetric, 
#         solver.indices, solver.linear_solver)

#     solver.options.iterative_refinement && iterative_refinement!(solver.data.step, solver)
    
#     solver.candidate .= solver.variables - step_size * solver.data.step
# end
# @show norm(solver.data.step - step_copy)
# ### 