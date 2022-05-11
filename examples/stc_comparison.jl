using LinearAlgebra, ForwardDiff, Convex, Mosek
using MosekTools
const cvx = Convex
nx = 6
nu = 3

N = 51

A = Matrix([zeros(3,3) I(3); zeros(3,6)])
B = Matrix([zeros(3,3); I(3)])
dt = .05
H = exp(dt*[A B; zeros(nu,nx + nu)])
Ad = H[1:nx,1:nx]
Bd = H[1:nx,(nx+1):end]


X = cvx.Variable(nx,N)
U = cvx.Variable(nu,N)
x1 = [-5.0;0;5;0;0;0]
xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
prob = cvx.minimize(.1*sumsquares(vec(U)) + sumsquares(vec(X)))
prob.constraints += X[:,1] == x1
prob.constraints += X[:,N] == xT
for i = 1:N-1
    prob.constraints += X[:,i+1] == Ad*X[:,i] + Bd*U[:,i]
end

# solve!(prob,()->Mosek.Optimizer())
# using SCS
using Gurobi
cvx.solve!(prob, Gurobi.Optimizer())

using MATLAB

Xm = X.value


a = -0.5
b = 3.0
c = 0.3
d = 3.0

function stc(x)
    g = -x[1] + a
    c = x[3] - b
    min(0,g)*c
end
