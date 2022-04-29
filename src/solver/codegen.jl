function generate_gradients(func::Function, num_variables::Int, num_parameters::Int, mode::Symbol;
    checkbounds=false,
    threads=false)
    @variables x[1:num_variables] θ[1:num_parameters]

    if mode == :scalar 
        f = num_parameters > 0 ? [func(x, θ)] : [func(x)]
        
        fx = Symbolics.gradient(f[1], x)
        fθ = Symbolics.gradient(f[1], θ)
        fxx = Symbolics.sparsejacobian(fx, x)
        fxθ = Symbolics.sparsejacobian(fx, θ)

        f_expr = Symbolics.build_function(f, x, θ,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fx_expr = Symbolics.build_function(fx, x, θ,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fθ_expr = Symbolics.build_function(fθ, x, θ,
            parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fxx_expr = Symbolics.build_function(fxx.nzval, x, θ,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fxθ_expr = Symbolics.build_function(fxθ.nzval, x, θ,
            parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]

        fxx_sparsity = collect(zip([findnz(fxx)[1:2]...]...))
        fxθ_sparsity = collect(zip([findnz(fxθ)[1:2]...]...))

        return f_expr, fx_expr, fθ_expr, fxx_expr, fxθ_expr, fxx_sparsity, fxθ_sparsity
    elseif mode == :vector 
        f = num_parameters > 0 ? func(x, θ) : func(x)
        
        fx = Symbolics.sparsejacobian(f, x)
        fθ = Symbolics.sparsejacobian(f, θ)

        fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
        fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

        @variables y[1:length(f)]
        # y = Symbolics.variables(:y, 1:length(f))

        fᵀy = length(f) == 0 ? 0.0 : sum(transpose(f) * y)
        fᵀyx = Symbolics.gradient(fᵀy, x)
        fᵀyxx = Symbolics.sparsejacobian(fᵀyx, x) 
        fᵀyxθ = Symbolics.sparsejacobian(fᵀyx, θ) 

        f_expr = Symbolics.build_function(f, x, θ,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fx_expr = Symbolics.build_function(fx.nzval, x, θ,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
            parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fᵀy_expr = Symbolics.build_function([fᵀy], x, θ, y,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fᵀyx_expr = Symbolics.build_function(fᵀyx, x, θ, y,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fᵀyxx_expr = Symbolics.build_function(fᵀyxx.nzval, x, θ, y,
            parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]
        fᵀyxθ_expr = Symbolics.build_function(fᵀyxθ.nzval, x, θ, y,
            parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
            checkbounds=checkbounds, 
            expression=Val{false})[2]

        fᵀyxx_sparsity = collect(zip([findnz(fᵀyxx)[1:2]...]...))
        fᵀyxθ_sparsity = collect(zip([findnz(fᵀyxθ)[1:2]...]...))

        return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity, fᵀy_expr, fᵀyx_expr, fᵀyxx_expr, fᵀyxθ_expr, fᵀyxx_sparsity, fᵀyxθ_sparsity, length(f)
    end
end

empty_constraint(x) = zeros(0) 
empty_constraint(x, θ) = zeros(0) 