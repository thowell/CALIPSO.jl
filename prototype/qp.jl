# problem 
n = 10 
m = 5 
p = n 

A = rand(m, n) 
x̄ = max.(randn(n), 1.0e-2) 
b = A * x̄

# equations
obj(x) = dot(x, x) + dot(ones(n), x)
eq(x) = A * x - b 
ineq(x) = x

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

