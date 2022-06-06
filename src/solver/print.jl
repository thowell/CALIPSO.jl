function solver_info(solver) 
    println(crayon"bold red", 
    "           _ _                 
          | (_)                
  ___ __ _| |_ _ __  ___  ___  
 / __/ _` | | | '_ \\/ __|/ _ \\ 
| (_| (_| | | | |_) \\__ \\ (_) |
 \\___\\__,_|_|_| .__/|___/\\___/ 
              | |              
              |_|
              
    ")
    println(crayon"reset bold black", 
    "Taylor Howell & Kevin Tracy & Simon Le Cleac'h")
    println("Robotic Exploration Lab")
    println("Stanford University & Carnegie Mellon University\n")
    print(crayon"reset")
end

function iteration_status(
        total_iterations, 
        outer_iterations, 
        inner_iterations, 
        residual_violation, 
        equality_violation, 
        cone_product_violation, 
        slack_violation, 
        central_path, 
        penalty, 
        step_size,
        options,
        ) 

    # header
    if rem(total_iterations - 1, options.print_frequency) == 0
        @printf "------------------------------------------------------------------------------------------------\n"
        @printf "total  outer  inner |residual| |equality|  |comp|    |slack|  central path   penalty      step  \n"
        @printf "------------------------------------------------------------------------------------------------\n"
    end
    
    # iteration information
    @printf("%3d     %2d    %3d   %9.2e  %9.2e %9.2e %9.2e   %9.2e   %9.2e   %9.2e \n", 
        total_iterations,
        outer_iterations,
        inner_iterations,
        residual_violation, 
        equality_violation, 
        cone_product_violation, 
        slack_violation,
        central_path, 
        penalty,
        step_size)
end

function solver_status(solver, status)
    @printf "------------------------------------------------------------------------------------------------\n"
    println("solution gradients: $(solver.options.differentiate)")
    println("solve status:       $(status ? "success" : "failure")")
    solver.dimensions.variables < 10 && println("solution:           $(round.(solver.solution.variables, sigdigits=3))")
    @printf "------------------------------------------------------------------------------------------------\n"
end