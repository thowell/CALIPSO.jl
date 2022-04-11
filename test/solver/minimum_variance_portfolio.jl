""" 
Create a portfolio optimization problem with p dimensions 
https://arxiv.org/pdf/1705.00772.pdf
"""
temp = np.random.randn(p, p)
Sigma = temp.T.dot(temp)


Sigma_sqrt = sp.linalg.sqrtm(Sigma)
o = np.ones((p, 1))

# Create standard form cone problem
Zp1 = np.zeros((p,1))

# setup for cone problem
c = np.vstack([Zp1, np.array([[1.0]])]).ravel()

G1 = sp.linalg.block_diag(2.0*Sigma_sqrt, -1.0)
q = np.vstack([Zp1, np.array([[1.0]])])
G2 = np.hstack([o.T, np.array([[0.0]])])
G3 = np.hstack([-o.T, np.array([[0.0]])])

h = np.vstack([Zp1, np.array([[1.0]])])
z = 1.0

A = np.vstack([G2, G3, -q.T, -G1 ])
b = np.vstack([1.0, -1.0, np.array([[z]]), h]).ravel()

betahat = cp.Variable(p)
return (betahat, cp.Problem(cp.Minimize(cp.quad_form(betahat, Sigma)),
                            [o.T * betahat == 1]), {
    'A' : A, 
    'b' : b,
    'c' : c, 
    'dims' : {
        'l' : 2,
        'q' : [p+2]
    },
    'beta_from_x' : lambda x: x[:p]
})