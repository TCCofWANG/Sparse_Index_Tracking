import torch
import pandas as pd
import numpy as np
import time
import math
import cvxpy as cp


# stocks data csv read
df = pd.read_csv('data.csv')
df = df.set_index('Date')


def top_k_projection(z, K):
    _, idx = torch.topk(torch.abs(z), K)
    z_proj = torch.zeros_like(z)
    z_proj[idx] = z[idx]
    return z_proj


def l0_MFSIT(A, r, q_t, K=50, rho=1e-3, iterations=500,  tol=1e-4, z_update_max_iter=100, z_update_tol=1e-7):

    r = r.squeeze()
    N = A.shape[0]
    w = torch.ones(471) / 471
    z = torch.zeros(w.shape)
    u = torch.zeros(w.shape)
    ones = torch.ones(471)

    residual_norms = []
    dual_residual_norms = []
    lagrangian_values = []

    def compute_lagrangian(theta, z, u, A, r, w, q_t, rho, lamda2, N):
        fit_term = torch.norm(torch.matmul(A, z) - r, p=2) ** 2
        sum1_constraint = (torch.matmul(z, ones) - 1) ** 2
        nonneg_constraint = torch.sum(torch.max(torch.zeros_like(z), -z) ** 2)  # ∑ max(0, -z_n)^2
        I_z = lamda2 * (sum1_constraint + nonneg_constraint)

        qt_theta = (1 / N) * torch.matmul(q_t.T, theta)
        lagrange_term = torch.matmul(u, (z - (w + qt_theta)))

        penalty_term = (rho / 2) * torch.norm(z - (w + qt_theta), p=2) ** 2

        # 拉格朗日函数的总值
        L_value = fit_term + I_z + lagrange_term + penalty_term
        return L_value.item()

    for k in range(iterations):
        # 更新theta
        lamda1 = 1e1
        lamda2 = 1e2
        step_size = 1e-6  # 步长
        v = u + rho * (z - w)
        term1 = torch.pinverse(torch.matmul(q_t, q_t.t()))
        term2 = torch.matmul(term1, q_t)
        theta = lamda1 * torch.matmul(term2, v)

        # 更新z
        z_update_iter = 0

        while z_update_iter < z_update_max_iter:
            z_old = z.clone()
            b = w + (1 / N) * torch.matmul(q_t.t(), theta)
            Az_r = torch.matmul(A, z) -r
            grad_z = 2 * torch.matmul(A.T, Az_r) + rho * (z - b) + u
            grad_constraint_sum1 = 2 * lamda2 * (torch.matmul(z, ones) - 1) * ones
            grad_nonneg = 2 * lamda2 * torch.min(torch.zeros_like(z), z)
            grad_z += grad_constraint_sum1 + grad_nonneg
            z = z - step_size * grad_z

            grad_norm = torch.norm(grad_z)
            z_change = torch.norm(z - z_old)

            if grad_norm < z_update_tol or z_change < z_update_tol:
                # print(f"z 更新在 {z_update_iter} 次迭代后收敛")
                break
            z_update_iter += 1

        z = top_k_projection(z, K)
        # 更新 u
        u = u + rho * (z - b)

        primal_residual = torch.norm(z - b)
        dual_residual = torch.norm(-rho * (z - b))
        residual_norms.append(primal_residual.item())
        dual_residual_norms.append(dual_residual.item())

        if primal_residual < tol and dual_residual < tol:
            break

        # 计算当前的拉格朗日函数值
        L_value = compute_lagrangian(theta, z, u, A, r, w, q_t, rho, lamda2, N)
        lagrangian_values.append(L_value)

    epsilon = 1e-8
    z = z / (torch.sum(z) + epsilon)
    return z



