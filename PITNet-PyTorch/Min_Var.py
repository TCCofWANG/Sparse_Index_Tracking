import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import torch


def min_var(returns):
    solvers.options['show_progress'] = False

    # 计算协方差矩阵
    # 标准化收益数据
    returns = (returns - returns.mean(dim=0)) / returns.std(dim=0)
    cov_matrix = np.cov(returns, rowvar=False)

    # 将数据转换为 cvxopt 格式
    n = cov_matrix.shape[1]
    P = opt.matrix(cov_matrix)
    q = opt.matrix(np.zeros(n))

    # 约束条件 Gx <= h
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(np.zeros(n))

    # 约束条件 Ax = b
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)


    # 求解二次规划问题
    sol = solvers.qp(P, q, G, h, A, b)

    # 获取最优权重
    w = np.array(sol['x'])
    w = torch.from_numpy(w).type(torch.Tensor)
    w = w.reshape(-1)
    return w

