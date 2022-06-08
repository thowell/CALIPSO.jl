# CALIPSO
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using CALIPSO

# Examples
Pkg.activate(@__DIR__)
Pkg.instantiate()
using LinearAlgebra

# ## horizon

function L_fx(u)

    return [u[1] -u[2] -u[3]  u[4];
                     u[2]  u[1] -u[4] -u[3];
                     u[3]  u[4]  u[1]  u[2];
                     u[4] -u[3]  u[2] -u[1]]

end
function uprime_from_xdot(xdot,u)
    # given a u and xdot, return u prime
    # xdot = float(xdot)
    # u = float(u)

    u1 = u[1]
    u2 = u[2]
    u3 = u[3]
    u4 = u[4]

    xd1 = xdot[1]
    xd2 = xdot[2]
    xd3 = xdot[3]

    up1 = (1/2)*( u1*xd1  +  u2*xd2  +  u3*xd3  )
    up2 = (1/2)*(-u2*xd1  +  u1*xd2  +  u4*xd3  )
    up3 = (1/2)*(-u3*xd1  -  u4*xd2  +  u1*xd3  )
    up4 = (1/2)*( u4*xd1  -  u3*xd2  +  u2*xd3  )

    uprime = [up1;up2;up3;up4]

    return uprime

    # return SVector((1/2)*( u[1]*xd[1]  +  u[2]*xd[2]  +  u[3]*xd3  ))

end


function u_from_x(x)
    # get KS u from x, u1 is always 1
    x = float(x)
    r = norm(x)

    u1 = .1
    u4 = sqrt(.5*(r+x[1]) - u1^2)
    u2 = (x[2]*u1 + x[3]*u4)/(r+x[1])
    u3 = (x[3]*u1 - x[2]*u4)/(r+x[1])

    u = [u1;u2;u3;u4]
    return u
end


function x_from_u(u)
    # gets x from u

    return [ u[1]^2 - u[2]^2 - u[3]^2 + u[4]^2,
                     2*(u[1]*u[2] - u[3]*u[4]),
                     2*(u[1]*u[3] + u[2]*u[4])]

end

function xdot_from_u(u,u_prime)
    up = u_prime;
    u1 = u[1];u2 = u[2];u3 = u[3];u4 = u[4]
    up1 = up[1];up2 = up[2];up3 = up[3];up4 = up[4]
    r = dot(u,u)
    x1d = (2/r)*(u1*up1 - u2*up2 - u3*up3 + u4*up4)
    x2d = (2/r)*(u2*up1 + u1*up2 - u4*up3 - u3*up4)
    x3d = (2/r)*(u3*up1 + u4*up2 + u1*up3 + u2*up4)

    return [x1d;x2d;x3d]
end
# ## rocket

num_state = 14
num_action = 3 # [tx;ty;tz;g+;g-;c+;c-]
num_parameter = 0

function rocket(x, u_dot, w)
    # unpack state
    p = [x[1],x[2],x[3],x[4]]
    p_prime = [x[5],x[6],x[7],x[8]]
    h = x[9]
    #r = x[10]
    #z = x[11]
    u = [x[12],x[13],x[14]]


    P_j2 = [u[1],u[2],u[3]]/uscale
    # Levi-Civita transformation matrix
    L = L_fx(p)

    # append 0 to end of P_j2 acceleration vector
    P_j2 = [P_j2;0]*tscale^2/dscale

    p_dp = -(h/2)*p + ((dot(p,p))/(2))*L'*P_j2

    return [p_prime[1],p_prime[2],p_prime[3],p_prime[4], # 4
                   p_dp[1],p_dp[2],p_dp[3],p_dp[4],    # 4
                   -2*dot(p_prime,L'*P_j2),  # 1
                   2*p'*p_prime,   # 1
                   2*(p_prime[1]*p[3] + p[1]*p_prime[3] + p_prime[2]*p[4] + p[2]*p_prime[4]), # 1
                   u_dot[1],u_dot[2],u_dot[3]] # 3

end


function midpoint_implicit(y, x, u, w)
    h = 3e-2 # timestep
    y - (x + h * rocket(0.5 * (x + y), u, w))
end
dscale = 1e7 # m
tscale = 20000 # s
uscale = 10000.0
μ = 3.986004418e14 *(tscale^2)/(dscale^3)

# initial conditions
r_eci = [6.578137000000001e6,0,0]
v_eci = [0,9111.210131355681,4642.393437617197]
R_0 = r_eci/dscale
V_0 = v_eci*tscale/dscale
p_0 = u_from_x(R_0)                          # u
p_prime_0 = uprime_from_xdot(V_0,p_0)     # du/ds
h_0 = μ/norm(R_0) - .5*norm(V_0)^2         # m^2/s^2
t_0 = 0                                    # seconds
x1 = [p_0; p_prime_0; h_0; norm(R_0); R_0[3];zeros(3)]
# ## model
dt = Dynamics(
    midpoint_implicit,
    num_state,
    num_state,
    num_action,
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1]

# ## initialization
# x1 = [-5.0;0;5;0;0;0]
# xT = [0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

# ## objective
ot = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)]) + 0.1 * dot(u, u)
oT = (x, u, w) -> 1.0 * dot(x[1:3] - xT[1:3], x[1:3] - xT[1:3]) + 0.1 * dot(x[3 .+ (1:3)], x[3 .+ (1:3)])
ct = Cost(ot, num_state, num_action,
    num_parameter=num_parameter)
cT = Cost(oT, num_state, 0,
    num_parameter=num_parameter)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
Fx_min = -10.0#-8.0
Fx_max = 10.0#8.0
Fy_min = -10.0#-8.0
Fy_max = 10.0#8.0
Fz_min = 0.0#6.0
Fz_max = 20.0#12.5


# state triggered constraints
# if g(x) > 0, then c(x) ≧ 0
# g(x) = -x[1] + a > 0
# c(x) = x[3] - b ≧ 0
#
# this is handled by introducing 4 new variables at each time step as controls
# g+,g-,c+,c- all ≥ 0
# g+ - g- = g(x)
# c+ - c- = c(x)
# g+ ⋅ c- = 0

a = -0.5
b = 3.0

function stc_con(x,u)
    tx,ty,tz,gp,gm,cp,cm = u

    g = -x[1] + a # ≧ 0
    c = x[3] - b  # ≧ 0

    [gp - gm - g;
     cp - cm - c;
     gp*cm]
end
eq1 = Constraint((x, u, w) -> x - x1, num_state, num_action)
eqT = Constraint((x, u, w) -> x - xT, num_state, 0)
eqSTC = Constraint((x,u,w) -> stc_con(x,u),                 # g+ ⋅ c- = 0
                               num_state,num_action)
eq = [eq1, [eqSTC for t = 2:T-1]...,eqT]


ineq = [[Constraint((x, u, w) ->
    [
        u[1] - Fx_min; Fx_max - u[1];
        u[2] - Fy_min; Fy_max - u[2];
        u[3] - Fz_min; Fz_max - u[3];
        u[4:7]                         # g+,g-,c+,c- all ≥ 0
    ], num_state, num_action
) for t = 1:T-1]..., Constraint()]

so = [[Constraint()] for t = 1:T]

# ## problem
trajopt = CALIPSO.TrajectoryOptimizationProblem(dyn, obj, eq, ineq, so)

# ## initialize
x_interpolation = linear_interpolation(x1, xT, T)
for i = 1:T
    h = 0.05
    x_interpolation[i][4:6] = (xT[1:3]-x1[1:3])/(h*T)
end
u_guess = [zeros(num_action) for t = 1:T-1]
for i = 1:T-1
    u_guess[i][1:3] = [0;0;9.8]
    g = -x_interpolation[i][1] + a
    if g >= 0
        u_guess[i][4] = g
        u_guess[i][5] = 0
    else
        u_guess[i][4] = 0
        u_guess[i][5] = -g
    end
    c =  x_interpolation[i][3] - b
    if c >= 0
        u_guess[i][6] = c
        u_guess[i][7] = 0
    else
        u_guess[i][6] = 0
        u_guess[i][7] = -c
    end
end

methods = ProblemMethods(trajopt)

solver = Solver(methods, trajopt.dimensions.total_variables, trajopt.dimensions.total_parameters, trajopt.dimensions.total_equality, trajopt.dimensions.total_cone,
    options=Options(verbose=true,penalty_initial=1.0e3))
initialize_states!(solver, trajopt, x_interpolation)
initialize_actions!(solver, trajopt, u_guess)

# ## solve
solve!(solver)
norm(solver.data.residual, Inf) < 1.0e-5

# ## solution
x_sol, u_sol = get_trajectory(solver, trajopt)

@show x_sol[1]
@show x_sol[T]

Xm = hcat(x_sol...)
Um = hcat(u_sol...)

using MATLAB
mat"
figure
hold on
plot($Xm(1,:),$Xm(3,:),'bo')
s = 10;
for i = 1:$T-1
    quiver($Xm(1,i),$Xm(3,i),$Um(1,i)/s,$Um(3,i)/s,'r')
end
patch([-5 $a $a -5],[0 0 $b $b],'b')
hold off
"

mat"
figure
hold on
plot($Um')
hold off
"
