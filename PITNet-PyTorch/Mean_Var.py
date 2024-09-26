import numpy as np
import torch
from cvxopt import matrix, solvers


def mean_var(returns):
    solvers.options['show_progress'] = False

    # 标准化收益数据
    returns = (returns - returns.mean(dim=0)) / returns.std(dim=0)

    # 计算预期回报和协方差矩阵
    expected_returns = returns.mean(dim=0)
    cov_matrix = np.cov(returns, rowvar=False)

    # 检查协方差矩阵的条件数
    cond_number = np.linalg.cond(cov_matrix)
    # print("Condition number of the covariance matrix:", cond_number)

    # 定义风险厌恶系数
    risk_aversion = 0.5

    # 定义问题
    n = len(expected_returns)
    P = matrix(risk_aversion * cov_matrix)
    a = np.float64(expected_returns)
    b = -a
    q = matrix(b)

    # 约束条件 Gx <= h
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))

    # 等式约束 Ax = b
    A = matrix(1.0, (1, n))
    b = matrix(1.0)

    # 求解问题
    sol = solvers.qp(P, q, G, h, A, b)
    weights = np.array(sol['x']).flatten()

    # 将结果转换为 PyTorch 张量
    weights = torch.from_numpy(weights).type(torch.Tensor)

    return weights


