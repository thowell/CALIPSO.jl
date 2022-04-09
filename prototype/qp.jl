# problem 
n = 10 
m = 5 
p_ineq = n 
p_soc = [0] 
p = p_ineq + sum(p_soc) 

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

idx_ineq = collect(1:p) 
idx_soc = [Int[]]

