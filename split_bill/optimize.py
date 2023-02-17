import cvxpy as cp
import numpy as np


def debt_constraint_matrix(n):
    dim = n * n
    A = np.zeros((n, dim))
    for i in range(n):
        A[i, (i * n) : (i * n) + n] = 1
    return A


def payment_flow_matrix(n):
    B = np.zeros((int(n * (n - 1) / 2), n * n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.zeros((n, n))
            d[i, j] = 1
            d[j, i] = 1
            B[k, :] = d.flatten()
            k += 1
    return B


def payself_matrix(n):
    C = np.zeros((n, n * n))
    for i in range(n):
        C[i, i * n + i] = 1
    return C


def optimize(balance):
    n = len(balance)
    m = n * n
    balance_arr = np.array(list(balance.values()))
    participants = list(balance.keys())
    A = debt_constraint_matrix(n)
    B = payment_flow_matrix(n)
    C = payself_matrix(n)
    W = cp.Parameter(m, nonneg=True)
    W.value = np.ones(m)
    x = cp.Variable(m)
    delta = 1e-5
    cons = [A @ x == balance_arr * -1, B @ x == 0, C @ x == 0]
    obj = cp.Minimize(W.T @ cp.abs(x))
    non_zeros = np.Inf
    n_iter = 25
    n_start = 10
    x_mat_best = np.zeros((n, n))
    for l in range(n_start):
        print(l)
        if "prob" in vars():
            del prob
        prob = cp.Problem(obj, cons)
        for k in range(n_iter):
            prob.solve()
            W.value = 1 / (delta + np.abs(x.value)) + (
                np.abs(np.random.randn(n * n) + 1e-5)
            )
            x_mat = x.value.reshape(n, n).round()
            non_zeros_new = np.sum(x_mat != 0)
            if non_zeros_new < non_zeros:
                non_zeros = non_zeros
                x_mat_best = x_mat.copy()
    for i, p in enumerate(participants):
        for j in np.argwhere(x_mat_best[i, :] > 0).flatten():
            print(f"{p} pays {x_mat_best[i,j]} to {participants[j]}")
    return {
        p: {p2: x_mat[i, j] for j, p2 in enumerate(participants)}
        for i, p in enumerate(participants)
    }
