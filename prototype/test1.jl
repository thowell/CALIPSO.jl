n = 50
m = 30
p = 3

x0 = ones(n)

obj(x) = dot(x, x)
eq(x) = x[1:30].^2 .- 1.2
ineq(x) = [x[1] + 10.0; x[2] + 5.0; 20.0 - x[5]]

@variables x[n]

f = obj(x)
fx = Symbolics.gradient(f, x) 
fxx = Symbolics.jacobian(fx, x) 

f_func = eval(Symbolics.build_function(f, x))
fx_func = eval(Symbolics.build_function(fx, x)[1])
fxx_func = eval(Symbolics.build_function(fxx, x)[1])

f_func(x0)
fx_func(x0)
fxx_func(x0)

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)


f(x0)
fx(x0) 
fxx(x0)
