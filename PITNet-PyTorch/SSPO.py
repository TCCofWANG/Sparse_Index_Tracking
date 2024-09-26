import torch
import torch.nn.functional as F
import time



def SSPO_fun(b_t_hat):
    # Parameter Setting
    max_iter = 1e3
    zeta = 500
    lambda_ = 0.5
    gamma = 0.01
    eta = 0.005
    x = -b_t_hat
    tao = lambda_ / gamma
    # Main

    stock_num = b_t_hat.shape[0]
    #date_num = b_t_hat.shape[1]
    prim_res = []

    g = b_t_hat.clone()
    b = b_t_hat.clone()
    rho = 0
    I = torch.eye(stock_num)
    YI = torch.ones((stock_num, stock_num))
    yi = torch.ones_like(b_t_hat)


    for iter in range(int(max_iter)):
        #A = tao * I + eta * YI
        K = tao * g + (eta - rho) * yi - x
        b = torch.linalg.solve(tao * I + eta * YI, tao * g + (eta - rho) * yi - x)
        g = F.threshold(b, gamma, 0)
        prim_res_tmp = yi * b - 1
        rho = rho + eta * prim_res_tmp

    b_tplus1_hat = zeta * b
    w_opt = simplex_projection_selfnorm2(b_tplus1_hat, 1)
    return w_opt


def simplex_projection_selfnorm2(v, b):
    # Ensure v is a tensor
    v = v.clone()

    v = torch.maximum(torch.tensor(0.0), v)
    w = v / torch.sum(v)

    return w

