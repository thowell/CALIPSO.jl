"""
    Solver options for Ipopt
        
    https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation
"""
Base.@kwdef mutable struct Options{T} 
    tol::T = 1e-6
    s_max::T = 100.0
    max_iter::Int = 1000
    # max_wall_time = 300.0
    max_cpu_time = 300.0
    dual_inf_tol::T = 1.0
    constr_viol_tol::T = 1.0e-3
    compl_inf_tol::T = 1.0e-3
    acceptable_tol::T = 1.0e-6
    acceptable_iter::Int = 15
    acceptable_dual_inf_tol::T = 1.0e10
    acceptable_constr_viol_tol::T = 1.0e-2
    acceptable_compl_inf_tol::T = 1.0e-2
    acceptable_obj_change_tol::T = 1.0e-5
    diverging_iterates_tol::T = 1.0e8
    mu_target::T = 1.0e-4
    print_level::Int = 5
    output_file = "output.txt"
    print_user_options = "no"
    # print_options_documentation = "no"
    # print_timing_statistics = "no"
    print_options_mode = "text"
    # print_advanced_options = "no"
    print_info_string = "no"
    inf_pr_output = "original"
    print_frequency_iter = 1
    print_frequency_time = 0.0
    skip_finalize_solution_call = "no"
    # timing_statistics = :no
end
