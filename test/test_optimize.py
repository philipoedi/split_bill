import numpy as np
from split_bill import optimize


def test_debt_constraint_matrix():
    n = 2
    A = optimize.debt_constraint_matrix(n)
    A_test = np.array([[1,1,0,0],[0,0,1,1]])
    assert np.all(A == A_test)

def test_payment_flow_matrix():
    n = 2
    B = optimize.payment_flow_matrix(n)
    assert np.all(B@np.ones(n**2) == 0)


def test_optimize():
    balance = {
        "a": 20,
        "b": 10,
        "c": -30,
        "d": 10,
        "e": -10
    }
    transactions = optimize.optimize(balance)
    assert transactions["a"]["c"] == 20


