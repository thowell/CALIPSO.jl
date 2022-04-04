n = 2
m = 1
p = 0
x0 = [2.0; 1.0]

obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
eq(x) = [x[1]^2 + x[2]^2 - 1.0]
ineq(x) = zeros(0)

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

