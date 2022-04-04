n = 3
m = 0
p = 1
x0 = rand(n)
obj(x) = x[1] - 2*x[2] + x[3] + sqrt(6)
eq(x) = zeros(0)
ineq(x) = [1 - x[1]^2 - x[2]^2 - x[3]^2]

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

