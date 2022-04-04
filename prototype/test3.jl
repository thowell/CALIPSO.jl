n = 2
m = 0
p = 2
x0 = rand(n)

obj(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
eq(x) = zeros(0)
ineq(x) = [-(x[1] -1)^3 + x[2] - 1;
            -x[1] - x[2] + 2]

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

