import torch
import pandas as pd
import time

# stocks data csv read
df = pd.read_csv('data.csv')
df = df.set_index('Date')


def avg_return(df, K):
    # Calculate daily returns
    df = df.pct_change()
    df = df.tail(-1)  # Remove the first row containing NaN values due to pct_change

    # Calculate the average daily return for each stock
    avg_daily_returns = df.mean()

    # Convert the result to a numpy array, then to a torch tensor
    avg_daily_returns = avg_daily_returns.to_numpy()
    avg_daily_returns = torch.from_numpy(avg_daily_returns).type(torch.Tensor)

    # Select the top K average daily returns
    top_K_indices = torch.topk(avg_daily_returns, K).indices
    result = torch.zeros_like(avg_daily_returns)
    result[top_K_indices] = avg_daily_returns[top_K_indices]

    return result


def Sharpe_ratios(df, K, risk_free_rate=0):
    # Calculate daily returns
    daily_returns = df.pct_change()
    daily_returns = daily_returns.tail(-1)  # Remove the first row containing NaN values due to pct_change
    # Calculate the average daily return for each stock
    avg_daily_returns = daily_returns.mean()

    # Calculate the standard deviation (volatility) of daily returns for each stock
    volatilities = daily_returns.std()
    # Calculate the Sharpe ratio for each stock
    sharpe_ratios = (avg_daily_returns - risk_free_rate) / volatilities
    # Convert the result to a numpy array, then to a torch tensor
    sharpe_ratios = sharpe_ratios.to_numpy()
    sharpe_ratios = torch.from_numpy(sharpe_ratios).type(torch.Tensor)

    # Select the top K average daily returns
    top_K_indices = torch.topk(sharpe_ratios, K).indices
    result = torch.zeros_like(sharpe_ratios)
    result[top_K_indices] = sharpe_ratios[top_K_indices]
    return result


def lagrangian_function(A, r, z, theta, w, q_t, rho):
    term1 = torch.norm(torch.matmul(A, z) - r, p=2) ** 2
    term2 = (rho / 2) * torch.norm(z - (w + (1 / A.shape[0]) * torch.matmul(theta, q_t)), p=2) ** 2
    term3 = torch.sum(z) - 1
    return term1 + term2 + term3


def top_k_projection(z, K):
    _, idx = torch.topk(torch.abs(z), K)
    z_proj = torch.zeros_like(z)
    z_proj[idx] = z[idx]
    return z_proj


def non_negative_projection(z):
    return torch.maximum(torch.tensor(0.0), z)


def unit_sum_projection(z):
    return z / torch.sum(z)


def PIT_admm(A, r, q_t, K=50, rho=0.01, iterations=1000, tol=1e-4):
    # 初始化theta, z, u, w
    w = Sharpe_ratios(df, 50)
    # w = avg_return(df, 50)
    # w = torch.ones(471) / 471
    z = torch.zeros(w.shape)
    u = torch.zeros(w.shape)
    ones = torch.ones(471)
    eta = 0.003


    ATA = torch.matmul(A.T, A)
    I = torch.eye(A.shape[1])
    N = A.shape[0]
    for k in range(iterations):
        # 更新theta
        lamda = 1e1
        v = z - w + u
        term1 = torch.pinverse(torch.matmul(q_t, q_t.t()))
        term2 = torch.matmul(term1, q_t)
        theta = lamda * torch.matmul(term2, v.T)

        # 更新z
        v = w + (1 / N) * torch.matmul(theta, q_t) - u
        temp = torch.matmul(A.T, r)
        temp1 = temp.squeeze()
        z_prev = z.clone()
        z = torch.linalg.solve(ATA + rho * I + torch.outer(ones, ones), temp1 + rho * v + ones)

        # 投影步骤
        z = non_negative_projection(z)
        z = top_k_projection(z, K)
        z = unit_sum_projection(z)

        # 更新u
        u = u + z - (w + (1 / N) * torch.matmul(theta, q_t))

    return z



