import torch.nn as nn
import torch
import math
import torch.nn.init as init


class l0_MFSIT_Net(nn.Module):
    def __init__(self):
        super(l0_MFSIT_Net, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1e1]), requires_grad=True)
        self.lamda = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.rho = nn.Parameter(torch.tensor([1e-3]), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor([1e-3]), requires_grad=True)
        self.w = nn.Parameter(torch.randn(471))
        self.b1 = nn.Parameter(torch.randn(471))  # 大小为 471 的向量
        self.linear = nn.Linear(471, 471)
        # 初始化参数
        init.constant_(self.alpha, 1e1)
        init.constant_(self.lamda, 0.1)
        init.constant_(self.rho, 1e-3)

        init.normal_(self.w, mean=0.0021, std=0.0005)

        self.First_layer = First_Layer(self.alpha, self.w, self.rho)
        self.theta_update_layer = theta_Update_Layer(self.alpha, self.w, self.rho)
        self.z_update_layer = z_Update_Layer(self.rho, self.b1, self.w, self.lamda, self.mu, self.linear)
        self.u_update_layer = u_Update_Layer(self.w, self.rho)
        self.final_layer = final_Layer(self.rho, self.b1, self.w, self.lamda, self.mu, self.linear)
        self.layers = nn.ModuleList()

        self.layers.append(self.First_layer)
        self.layers.append(self.z_update_layer)
        self.layers.append(self.u_update_layer)
        for i in range(8):
            self.layers.append(self.theta_update_layer)
            self.layers.append(self.z_update_layer)
            self.layers.append(self.u_update_layer)
        self.layers.append(self.theta_update_layer)
        self.layers.append(self.final_layer)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, q_t):
        out = (x, q_t)
        for layer in self.layers:
            out = layer(*out)
        return out


class First_Layer(nn.Module):
    def __init__(self, alpha, w, rho):
        super(First_Layer, self).__init__()
        self.alpha = alpha
        self.w = w
        self.rho = rho

    def forward(self, x, q_t):

        z = torch.zeros(471)
        u = torch.zeros(471)
        v = u - self.rho * (z - self.w)
        theta = self.alpha * torch.matmul(torch.pinverse(torch.matmul(q_t, q_t.t())), torch.matmul(q_t, v))
        MFSIT_data = dict()
        ATA = torch.matmul(x.T, x)
        MFSIT_data['A_Transfer'] = ATA
        MFSIT_data['z'] = torch.zeros(471)
        MFSIT_data['input'] = x
        MFSIT_data['theta'] = theta
        MFSIT_data['u'] = torch.zeros(471)

        return MFSIT_data, q_t


class theta_Update_Layer(nn.Module):
    def __init__(self, alpha, w, rho):
        super(theta_Update_Layer, self).__init__()
        self.alpha = alpha
        self.w = w
        self.rho = rho

    def forward(self, MFSIT_data, q_t):
        z = MFSIT_data['z']
        u = MFSIT_data['u']
        v = u - self.rho * (z - self.w)
        term1 = torch.matmul(q_t, q_t.t())
        theta = self.alpha * torch.matmul(torch.pinverse(term1), torch.matmul(q_t, v))
        MFSIT_data['theta'] = theta

        return MFSIT_data, q_t


class z_Update_Layer(nn.Module):
    def __init__(self, rho, b1, w, lamda, mu, linear, topk=50):
        super(z_Update_Layer, self).__init__()
        self.rho = rho
        self.linear = linear
        self.topk = topk
        self.b1 = b1
        self.w = w
        self.lamda = lamda
        self.mu = mu

    def forward(self, MFSIT_data, q_t):

        theta = MFSIT_data['theta']
        z = MFSIT_data['z']
        u = MFSIT_data['u']
        A = MFSIT_data['input']
        N = 471
        ones = torch.ones(471)
        b = self.w + (1 / N) * torch.matmul(q_t.t(), theta)

        term1 = self.linear(A)

        W2 = term1.mean(dim=0)

        grad_z = W2 + self.rho * (z - b) + u
        grad_constraint_sum1 = 2 * self.lamda * (torch.matmul(z, ones) - 1) * ones
        grad_nonneg = 2 * self.lamda * torch.min(torch.zeros_like(z), z)
        grad_z += grad_constraint_sum1 + grad_nonneg
        z = z - torch.mul(self.mu, grad_z)
        z = torch.relu(z)
        _, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=z.dtype))
        z = z * mask

        MFSIT_data['z'] = z
        return MFSIT_data, q_t


class u_Update_Layer(nn.Module):
    def __init__(self, w, rho):
        super(u_Update_Layer, self).__init__()
        self.w = w
        self.rho = rho

    def forward(self, MFSIT_data, q_t):
        u = MFSIT_data['u']
        theta = MFSIT_data['theta']
        N = 471
        b = self.w + (1 / N) * torch.matmul(q_t.t(), theta)
        z = MFSIT_data['z']
        output = u + self.rho * (z - b)
        MFSIT_data['u'] = output
        return MFSIT_data, q_t


class final_Layer(nn.Module):
    def __init__(self, rho, b1, w, lamda, mu, linear, topk=50):
        super(final_Layer, self).__init__()
        self.rho = rho
        self.linear = linear
        self.topk = topk
        self.b1 = b1
        self.w = w
        self.lamda = lamda
        self.mu = mu

    def forward(self, MFSIT_data, q_t):
        theta = MFSIT_data['theta']
        z = MFSIT_data['z']
        u = MFSIT_data['u']
        A = MFSIT_data['input']
        N = 471
        ones = torch.ones(471)
        b = self.w + (1 / N) * torch.matmul(q_t.t(), theta)

        term1 = self.linear(A)

        W2 = term1.mean(dim=0)

        grad_z = W2 + self.rho * (z - b) + u
        grad_constraint_sum1 = 2 * self.lamda * (torch.matmul(z, ones) - 1) * ones
        grad_nonneg = 2 * self.lamda * torch.min(torch.zeros_like(z), z)
        grad_z += grad_constraint_sum1 + grad_nonneg
        z = z - torch.mul(self.mu, grad_z)
        z = torch.relu(z)

        _, indices = torch.topk(z, self.topk, dim=-1)
        mask = torch.zeros_like(z)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=z.dtype))
        z = z * mask
        epsilon = 1e-8
        z = z / (torch.sum(z) + epsilon)
        return z







