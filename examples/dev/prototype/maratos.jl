n = 2
m = 1
p = 0
x0 = [2.0; 1.0]

obj(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
eq(x) = [x[1]^2 + x[2]^2 - 1.0]
ineq(x) = zeros(0)

# gradients 
# gradients 
f, _fx, _fxx = CALIPSO.generate_gradients(obj, n, :scalar)
_g, _gx, _gyxx = CALIPSO.generate_gradients(eq, n, :vector)
_h, _hx, _hyxx = CALIPSO.generate_gradients(ineq, n, :vector)

x0 = ones(n)
y0 = randn(m) 
z0 = randn(p)

function fx(x) 
    grad = zeros(n) 
    _fx(grad, x) 
    return grad 
end

function fxx(x) 
    hess = zeros(n, n) 
    _fxx(hess, x) 
    return hess 
end

function g(x) 
    con = zeros(m) 
    _g(con, x) 
    return con 
end

function gx(x) 
    jac = zeros(m, n) 
    _gx(jac, x) 
    return jac 
end

function gyxx(x, y) 
    hess = zeros(n, n) 
    _gyxx(hess, x, y) 
    return hess 
end

function h(x) 
    con = zeros(p) 
    _h(con, x) 
    return con 
end

function hx(x) 
    jac = zeros(p, n) 
    _hx(jac, x) 
    return jac 
end

function hyxx(x, y) 
    hess = zeros(n, n) 
    _hyxx(hess, x, y) 
    return hess 
end

f(x0)
fx(x0)
rank(fxx(x0))
g(x0)
gx(x0)
gyxx(x0, y0)
h(x0)
rank(hx(x0))
hyxx(x0, y0)



