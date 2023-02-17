import numpy as np
import cvxpy as cp

# number of people
n = 8
dim = n*n

debt = np.random.randint(-250, 250, n)
debt[-1] = (debt.sum() - debt[-1]) * -1

# total equil
A = np.zeros((n, dim))
for i in range(n):
    A[i, (i*n):(i*n)+n] = 1

# not paying self
#B = np.zeros((n, dim))
#for i in range(n):
#    B[i, i*n+i] = 1

# c
C = np.zeros((int(n*(n-1)/2), dim))
k = 0
for i in range(n):
    for j in range(i+1, n):
        d = np.zeros((n, n))
        d[i,j] = 1
        d[j,i] = -1
        C[k, :] = d.flatten()
        k += 1


x = cp.Variable(n*n)
cons = [
    A@x == debt*-1,
  #  B@x == 0,
    C@x == 0
]

obj = cp.Minimize(cp.norm(x,1))

#prob = cp.Problem(obj, cons)
#prob.solve()

#print(x.value.reshape(n,n).round())
#print(debt)


NUM_RUNS = 15
nnzs_log = np.array(())

# Store W as a positive parameter for simple modification of the problem.
W = cp.Parameter(shape=dim, nonneg=True);
x_log = cp.Variable(shape=dim)

# Initial weights.
W.value = np.ones(dim);
delta = 1e-5
# Setup the problem.
obj = cp.Minimize(W.T@cp.abs(x)) # sum of elementwise product
prob = cp.Problem(obj, cons)
best = np.zeros(dim)
# Do the iterations of the problem, solving and updating W.
for k in range(1, NUM_RUNS+1):
    # Solve problem.
    # The ECOS solver has known numerical issues with this problem
    # so force a different solver.
    prob.solve()
    # Check for error.
    #if prob.status != cp.OPTIMAL:
    #    raise Exception("Solver did not converge!")

    # Display new number of nonzeros in the solution vector.
    nnz = (np.absolute(x.value) > delta).sum()
    print(x.value.reshape(n,n).round())
    if k == 1:
        best = x.value.copy()
    if np.any(nnz < nnzs_log):
        best = x.value
    nnzs_log = np.append(nnzs_log, nnz);
    print('Iteration {}: Found a feasible x in R^{}'
          ' with {} nonzeros...'.format(k, dim, nnz))
    # Adjust the weights elementwise and re-iterate
    W.value = np.ones(dim)/(delta*np.ones(dim) + np.absolute(x.value))