n = 8
m = 7
p = 8
x0 = zeros(n)

obj(x) = (x[1] - 5)^2 + (2*x[2] + 1)^2
eq(x) = [2*(x[2] - 1) - 1.5*x[2] + x[3] - 0.5*x[4] + x[5];
            3*x[1] - x[2] - 3.0 - x[6];
            -x[1] + 0.5*x[2] + 4.0 - x[7];
            -x[1] - x[2] + 7.0 - x[8];
            x[3]*x[6];
            x[4]*x[7];
            x[5]*x[8];]
ineq(x) = x

# gradients 
f, fx, fxx = CALIPSO.generate_gradients(obj, n, :scalar, output=:out)
g, gx, gyxx = CALIPSO.generate_gradients(eq, n, :vector, output=:out)
h, hx, hyxx = CALIPSO.generate_gradients(ineq, n, :vector, output=:out)

