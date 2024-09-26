import numpy as np
import torch
from cvxopt import matrix, solvers


def mean_var(returns):
    solvers.options['show_progress'] = False
    returns = (returns - returns.mean(dim=0)) / returns.std(dim=0)

    expected_returns = returns.mean(dim=0)
    cov_matrix = np.cov(returns, rowvar=False)
    cond_number = np.linalg.cond(cov_matrix)

    risk_aversion = 0.5
    n = len(expected_returns)
    P = matrix(risk_aversion * cov_matrix)
    a = np.float64(expected_returns)
    b = -a
    q = matrix(b)

    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()

    weights = torch.from_numpy(weights).type(torch.Tensor)

    return weights


