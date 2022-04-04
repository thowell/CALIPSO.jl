n = 3
m = 2
p = 2
x0 = [-2.0, 3.0, 1.0]

obj(x) = x[1]
eq(x) = [x[1]^2 - x[2] - 1.0; x[1] - x[3] - 0.5]
ineq(x) = x[2:3]

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

